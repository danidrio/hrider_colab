import fitz
from pathlib import Path

from hrider.anonymizer.anonymizer import Anonymizer


class PDFAnonymizer:
    """
    Anonimiza PDFs con texto real aplicando redacciones sobre el PDF original.
    """

    def __init__(self, anonymizer=None):
        self.anonymizer = anonymizer or Anonymizer()

    def anonymize_pdf(
        self,
        input_path,
        output_path,
        people=None,
        store_page_text=False,
        store_matches=True
    ):
        """
        Lee un PDF, anonimiza el texto de cada página y guarda un PDF nuevo
        preservando el documento original tanto como sea posible.

        Parámetros:
        - input_path: ruta del PDF de entrada.
        - output_path: ruta del PDF anonimizado generado.
        - people: lista opcional de personas conocidas.
        - store_page_text:
            False por defecto para reducir memoria en PDFs grandes.
            Si True, cada página del resultado incluirá original_text y
            anonymized_text.
        - store_matches:
            True por defecto para conservar el detalle de coincidencias y poder
            generar informes. Si False, cada página solo conserva estadísticas
            agregadas por tipo.

        Retorna un diccionario con:
        - input_path: ruta del PDF de entrada.
        - output_path: ruta del PDF anonimizado generado.
        - pages: lista con el resultado por página. Por defecto no incluye el
          texto completo de cada página.
        - manual_review_required: indica si alguna página requiere revisión
          manual.
        """
        doc = fitz.open(str(input_path))
        page_results = []

        try:
            for page_index, page in enumerate(doc, start=1):
                original_text = page.get_text("text") or ""
                result = self.anonymizer.anonymize(original_text, people=people)

                # Para aplicar redacciones sobre el PDF se usa una lista separada
                # de matches. Las posiciones start/end de regex, people y LLM
                # pueden pertenecer a textos intermedios distintos, por lo que
                # PDFAnonymizer localiza visualmente por matched_fragment.
                page_matches = self._build_pdf_matches(result.get("matches", []))
                self._apply_matches_to_page(page, page_matches)
                page_result = {
                    "page_number": page_index,
                    "manual_review_required": result["manual_review_required"],
                    "llm_detection_skipped": result.get(
                        "llm_detection_skipped",
                        False
                    ),
                    "llm_error": result.get("llm_error")
                }

                if store_matches:
                    page_result["matches"] = page_matches
                else:
                    page_result["matches_count"] = len(page_matches)
                    page_result["automatic_replacements_count"] = sum(
                        1
                        for match in page_matches
                        if match.get("auto_redact", True)
                    )
                    page_result["manual_review_matches_count"] = sum(
                        1
                        for match in page_matches
                        if match.get("manual_review_required", False)
                    )
                    page_result["replacement_types"] = (
                        self._count_matches_by_entity_type(page_matches)
                    )

                if store_page_text:
                    page_result["original_text"] = original_text
                    page_result["text_after_regex"] = result.get("text_after_regex")
                    page_result["text_after_people"] = result.get("text_after_people")
                    page_result["anonymized_text"] = result["anonymized_text"]

                page_results.append(page_result)

            doc.save(
                str(output_path),
                garbage=4,
                deflate=True,
                clean=True
            )
        finally:
            doc.close()

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "pages": page_results,
            "manual_review_required": any(
                page_result["manual_review_required"]
                for page_result in page_results
            )
        }

    def _build_pdf_matches(self, matches):
        """
        Construye la lista de matches que se intentarán aplicar sobre el PDF.

        - Aquí solo se filtran matches automáticos y se deduplican por fragmento,
          replacement, tipo y fuente.
        """
        pdf_matches = []

        for match in matches:
            fragment = str(match.get("matched_fragment", "")).strip()
            if not fragment:
                continue

            if not match.get("auto_redact", True):
                continue

            pdf_matches.append(match)

        return self._deduplicate_pdf_matches(pdf_matches)


    def _deduplicate_pdf_matches(self, matches):
        """
        Reglas:
        - No usa posiciones porque los matches vienen de fases distintas.
        - Elimina duplicados equivalentes aunque vengan de fuentes distintas.
        - Prioriza reemplazos específicos de persona, como [PERSONA:EMP001],
        frente a reemplazos genéricos como [PERSONA].
        - Prioriza detecciones de people sobre LLM cuando representan personas.
        """
        if not matches:
            return []

        def replacement_priority(match):
            replacement = str(match.get("replacement", ""))
            source = str(match.get("source", ""))
            entity_type = str(match.get("entity_type", "")).upper()

            if entity_type == "PERSONA" and replacement.startswith("[PERSONA:"):
                return 100

            if entity_type == "PERSONA" and source.startswith("people"):
                return 90

            if entity_type == "PERSONA" and replacement == "[PERSONA]":
                return 50

            if source == "regex":
                return 80

            if source == "llm":
                return 40

            return 10

        sorted_matches = sorted(
            matches,
            key=lambda match: (
                -replacement_priority(match),
                -len(str(match.get("matched_fragment", ""))),
                str(match.get("entity_type", "")),
                str(match.get("replacement", "")),
            )
        )

        unique = []

        for match in sorted_matches:
            fragment = str(match.get("matched_fragment", "")).strip()
            replacement = str(match.get("replacement", ""))
            entity_type = str(match.get("entity_type", "")).upper()
            normalized = str(
                match.get("matched_fragment_normalized")
                or self._normalize_text(fragment)
            )

            if not fragment:
                continue

            discard = False

            for chosen in unique:
                chosen_fragment = str(chosen.get("matched_fragment", "")).strip()
                chosen_replacement = str(chosen.get("replacement", ""))
                chosen_entity_type = str(chosen.get("entity_type", "")).upper()
                chosen_normalized = str(
                    chosen.get("matched_fragment_normalized")
                    or self._normalize_text(chosen_fragment)
                )

                same_entity = entity_type == chosen_entity_type
                same_fragment = fragment == chosen_fragment
                same_normalized = normalized and normalized == chosen_normalized

                # Duplicado real aunque venga de otra fuente.
                if same_entity and (same_fragment or same_normalized):
                    discard = True
                    break

                # Caso específico:
                # si ya tenemos [PERSONA:EMP001], no conservar un [PERSONA]
                # genérico para el mismo texto normalizado o incluido.
                if entity_type == "PERSONA" and chosen_entity_type == "PERSONA":
                    current_is_generic = replacement == "[PERSONA]"
                    chosen_is_specific = chosen_replacement.startswith("[PERSONA:")

                    if current_is_generic and chosen_is_specific:
                        if (
                            same_normalized
                            or normalized in chosen_normalized
                            or chosen_normalized in normalized
                        ):
                            discard = True
                            break

            if not discard:
                unique.append(match)

        return sorted(
            unique,
            key=lambda match: (
                -len(str(match.get("matched_fragment", ""))),
                str(match.get("entity_type", "")),
                str(match.get("replacement", ""))
            )
        )

    def _normalize_text(self, text):
        normalize_method = getattr(self.anonymizer, "_normalize_text", None)
        if callable(normalize_method):
            return str(normalize_method(text))
        return str(text).strip().lower()


    def _count_matches_by_entity_type(self, matches):
        """
        Cuenta sustituciones automáticas por tipo de entidad.

        Se usa cuando store_matches=False para devolver estadísticas útiles
        sin guardar cada match completo en memoria.
        """
        counts = {}

        for match in matches:
            if not match.get("auto_redact", True):
                continue

            entity_type = str(match.get("entity_type", "OTRO")).upper() or "OTRO"
            counts[entity_type] = counts.get(entity_type, 0) + 1

        return dict(sorted(counts.items()))


    def print_pdf_result(self, pdf_result):
        print("=" * 100)
        print("PDF original: ", pdf_result["input_path"])
        print("PDF anonimizado: ", pdf_result["output_path"])
        print("Requiere revisión manual: ", pdf_result["manual_review_required"])
        print("\nCambios en PDF:\n")
        for page_result in pdf_result["pages"]:
            self._print_pdf_page_result(page_result)
        

    def _print_pdf_page_result(self, page_result):
        page_matches = [
            match for match in page_result["matches"]
            if match.get("auto_redact", True)
        ]
        if not page_matches:
            return

        print(f"- Página {page_result['page_number']}")
        for match in page_matches:
            print(
                f"  {match['entity_type']} | "
                f"fragment='{match['matched_fragment']}' | "
                f"replacement='{match['replacement']}' | "
                f"source={match['source']}"
            )

        if page_result.get("llm_detection_skipped"):
            print(f"  LLM OMITIDO: {page_result.get('llm_error')}")


    def _apply_matches_to_page(self, page, matches):
        """
        Busca fragmentos detectados en la página y aplica redacciones.

        Nota:
            La búsqueda se hace por texto con page.search_for(), no por las
            posiciones start/end del Anonymizer. Por tanto, se redactan todas
            las apariciones del fragmento encontradas en la página.

        Importante:
            No se usa rect.intersects() para evitar duplicados. En PDFs reales,
            dos entidades distintas pueden tener cajas que se solapan ligeramente
            o incluso comparten altura de línea. Se evitan solo duplicados reales
            mediante una clave geométrica aproximada.
        """
        redactions_added = False
        processed_fragments = set()
        redacted_entries = []

        sorted_matches = sorted(
            matches,
            key=lambda match: (
                -(match.get("end", 0) - match.get("start", 0)),
                -len(str(match.get("matched_fragment", "")))
            )
        )

        for match in sorted_matches:
            if not match.get("auto_redact", True):
                continue

            fragment = str(match.get("matched_fragment", "")).strip()

            if not fragment:
                continue

            replacement = str(match.get("replacement", "[REDACTED]"))
            fragment_key = (
                fragment,
                replacement,
                str(match.get("entity_type", "")),
                str(match.get("source", ""))
            )

            if fragment_key in processed_fragments:
                continue

            processed_fragments.add(fragment_key)
            rects = self._search_fragment_rects(page, fragment)

            if not rects:
                continue

            for rect in rects:
                entity_type = str(match.get("entity_type", "OTRO")).upper()

                if self._is_duplicate_redaction(
                    rect=rect,
                    replacement=replacement,
                    entity_type=entity_type,
                    redacted_entries=redacted_entries
                ):
                    continue

                # Escribimos la etiqueta en cada caja redactada para mantener
                # el contexto visual cuando hay varias ocurrencias.
                page.add_redact_annot(
                    rect,
                    text=replacement,
                    fill=(1, 1, 1),
                    text_color=(0, 0, 0),
                    cross_out=False,
                    fontname="helv",
                    fontsize=max(min(rect.height * 0.8, 12), 6),
                    align=fitz.TEXT_ALIGN_LEFT
                )
                redacted_entries.append({
                    "rect": rect,
                    "replacement": replacement,
                    "entity_type": entity_type
                })                
                redactions_added = True

        if redactions_added:
            page.apply_redactions()


    def _is_duplicate_redaction(
        self,
        rect,
        replacement,
        entity_type,
        redacted_entries,
        overlap_threshold=0.75
    ):
        """
        Considera duplicada una redacción solo cuando:
        - tiene el mismo replacement;
        - tiene el mismo tipo de entidad;
        - y el área se solapa de forma sustancial.

        Esto evita duplicados visuales como dos [PERSONA:EMP001] encima,
        pero no bloquea redacciones distintas que solo se pisan ligeramente.
        """
        for entry in redacted_entries:
            if entry["replacement"] != replacement:
                continue

            if entry["entity_type"] != entity_type:
                continue

            existing_rect = entry["rect"]

            if self._rect_overlap_ratio(rect, existing_rect) >= overlap_threshold:
                return True

            if self._rect_centers_are_close(rect, existing_rect):
                return True

        return False


    def _rect_overlap_ratio(self, a, b):
        """
        Devuelve cuánto se solapan dos rectángulos respecto al menor de los dos.

        1.0 significa que el menor está completamente cubierto.
        0.0 significa que no hay solape.
        """
        intersection = a & b

        if intersection.is_empty:
            return 0.0

        intersection_area = intersection.get_area()
        min_area = min(a.get_area(), b.get_area())

        if min_area <= 0:
            return 0.0

        return intersection_area / min_area


    def _rect_centers_are_close(self, a, b, tolerance=2.0):
        """
        Detecta duplicados casi idénticos aunque las cajas no coincidan exactamente.

        Útil cuando PyMuPDF devuelve rectángulos con pequeñas variaciones.
        """
        ax = (a.x0 + a.x1) / 2
        ay = (a.y0 + a.y1) / 2
        bx = (b.x0 + b.x1) / 2
        by = (b.y0 + b.y1) / 2

        return abs(ax - bx) <= tolerance and abs(ay - by) <= tolerance

    def _rect_key(self, rect, precision=1):
        """
        Devuelve una clave geométrica aproximada para detectar duplicados reales.
        No debe usarse para bloquear solapes entre entidades distintas.
        """
        return (
            round(rect.x0, precision),
            round(rect.y0, precision),
            round(rect.x1, precision),
            round(rect.y1, precision),
        )

    def _search_fragment_rects(self, page, fragment):
        """
        Busca un fragmento en el PDF usando variantes conservadoras.

        Criterio:
        - Primero intenta buscar el fragmento exactamente como viene.
        - Después intenta buscarlo con espacios normalizados.
        """
        fragment = str(fragment or "").strip()

        if not fragment:
            return []

        candidates = []

        def add_candidate(value):
            value = str(value or "").strip()
            if value and value not in candidates:
                candidates.append(value)

        # 1. Fragmento original.
        add_candidate(fragment)

        # 2. Fragmento con espacios, tabs y saltos de línea normalizados.
        normalized_spaces = " ".join(fragment.split())
        add_candidate(normalized_spaces)

        for candidate in candidates:
            rects = page.search_for(candidate)
            if rects:
                return rects

        return []

    def anonymize_pdf_directory(
        self,
        input_dir,
        output_dir,
        people=None,
        recursive=False,
        overwrite=True,
        output_suffix="_anonimizado",
        store_page_text=False,
        store_matches=True
    ):
        """
        Anonimiza todos los PDFs de un directorio de entrada y guarda las copias
        anonimizadas en un directorio de salida.

        Parámetros:
        - input_dir: directorio donde están los PDFs originales.
        - output_dir: directorio donde se guardarán los PDFs anonimizados.
        - people: lista opcional de personas conocidas.
        - recursive:
            False por defecto. Si True, busca PDFs también en subdirectorios y
            conserva la estructura relativa dentro de output_dir.
        - overwrite:
            True por defecto. Si False, no sobrescribe PDFs ya existentes.
        - output_suffix:
            Sufijo añadido al nombre del PDF de salida. Por defecto:
            documento.pdf -> documento_anonimizado.pdf.
            Si quieres conservar el mismo nombre en output_dir, usa "".
        - store_page_text:
            False por defecto para reducir memoria en PDFs grandes.
        - store_matches:
            True por defecto para conservar el detalle de coincidencias.

        Retorna un diccionario resumen con el resultado por fichero.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"input_dir no existe: {input_dir}")

        if not input_dir.is_dir():
            raise NotADirectoryError(f"input_dir no es un directorio: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_paths = (
            sorted(input_dir.rglob("*.pdf"))
            if recursive
            else sorted(input_dir.glob("*.pdf"))
        )

        file_results = []

        for input_path in pdf_paths:
            print(input_path)
            relative_path = input_path.relative_to(input_dir)
            output_parent = output_dir / relative_path.parent
            output_parent.mkdir(parents=True, exist_ok=True)

            output_name = (
                f"{input_path.stem}{output_suffix}{input_path.suffix}"
                if output_suffix
                else input_path.name
            )
            output_path = output_parent / output_name

            if output_path.exists() and not overwrite:
                file_results.append({
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "success": False,
                    "skipped": True,
                    "error": "output_path ya existe y overwrite=False",
                    "manual_review_required": False
                })
                continue

            try:
                pdf_result = self.anonymize_pdf(
                    input_path=input_path,
                    output_path=output_path,
                    people=people,
                    store_page_text=store_page_text,
                    store_matches=store_matches
                )

                pdf_result["success"] = True
                pdf_result["skipped"] = False
                pdf_result["error"] = None
                file_results.append(pdf_result)

            except Exception as exc:
                file_results.append({
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "success": False,
                    "skipped": False,
                    "error": str(exc),
                    "manual_review_required": False
                })

        processed_files = [
            file_result
            for file_result in file_results
            if file_result.get("success")
        ]
        failed_files = [
            file_result
            for file_result in file_results
            if not file_result.get("success") and not file_result.get("skipped")
        ]
        skipped_files = [
            file_result
            for file_result in file_results
            if file_result.get("skipped")
        ]

        return {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "recursive": bool(recursive),
            "total_files": len(pdf_paths),
            "processed_files": len(processed_files),
            "failed_files": len(failed_files),
            "skipped_files": len(skipped_files),
            "manual_review_required": any(
                file_result.get("manual_review_required", False)
                for file_result in processed_files
            ),
            "files": file_results
        }
