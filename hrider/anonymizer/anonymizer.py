import re
import json
import unicodedata
from copy import deepcopy
from rapidfuzz import fuzz


class Anonymizer:
    """
    Anonimizador de texto para detección y sustitución de datos sensibles.

    La clase aplica un pipeline de anonimización en tres fases:

    1. Detección por expresiones regulares:
       Detecta entidades estructuradas como emails, teléfonos, DNI/NIE, URLs,
       direcciones postales e importes. Estas entidades se consideran de alta
       confianza y se sustituyen automáticamente.

    2. Detección de personas conocidas:
       Detecta menciones a personas a partir de una lista de empleados/personas
       proporcionada por el usuario. La detección usa variantes normalizadas y
       comparación fuzzy.

       Las variantes fuertes, como nombre completo, inicial + apellido o usuario
       de email, pueden sustituirse automáticamente si superan sus umbrales.

       Las variantes débiles, como nombre de pila o apellido aislado, se detectan
       para revisión manual, pero no se sustituyen automáticamente.

    3. Detección opcional mediante LLM:
       Si está activado, se usa un cliente LLM externo para detectar posibles
       entidades sensibles que no hayan sido cubiertas por regex o por la lista
       de personas conocidas.

       El LLM debe devolver únicamente JSON válido con fragmentos literales del
       texto. En esta fase, los hallazgos marcados con manual_review_required=True
       no se sustituyen automáticamente; los hallazgos con
       manual_review_required=False sí se sustituyen.

    Uso principal:
        result = anonymizer.anonymize(text, people)

    El parámetro people debe ser una lista de diccionarios con esta estructura
    esperada:

        [
            {
                "employee_id": "EMP001",
                "name": "María",
                "lastname": "Gómez",
                "email": "maria.gomez@empresa.com"
            }
        ]

    El método anonymize() devuelve un diccionario con el texto original, los textos
    intermedios, el texto anonimizado final y la lista completa de matches.

    Cada match puede incluir, entre otros, estos campos:

        - entity_type:
            Tipo de entidad detectada. Por ejemplo: PERSONA, EMAIL, TELEFONO, 
            DNI_NIE, IBAN, DIRECCION, ROL, IMPORTE, URL, OTRO.

        - matched_fragment:
            Fragmento exacto detectado en el texto.

        - matched_fragment_normalized:
            Versión normalizada del fragmento usada internamente para comparar.

        - score:
            Confianza numérica de la coincidencia. En regex y LLM suele ser 100.
            En detección fuzzy de personas representa la similitud calculada.

        - auto_redact:
            Indica si el fragmento debe sustituirse automáticamente en el texto.

            True:
                El fragmento se incluye en los spans que serán reemplazados.

            False:
                El fragmento se informa como hallazgo, pero no se reemplaza
                automáticamente.

        - manual_review_required:
            Indica si el hallazgo requiere revisión humana.

            True:
                El hallazgo debe revisarse, ya sea porque no se ha sustituido
                automáticamente o porque la detección es ambigua/fuzzy.

            False:
                El hallazgo se considera suficientemente claro y no requiere
                validación manual.

        - replacement:
            Texto usado como sustitución cuando auto_redact=True.

        - risk_level:
            Nivel de riesgo estimado: low, medium o high.

        - start / end:
            Posiciones del fragmento dentro del texto procesado en esa fase.

        - source:
            Fuente de la detección: regex, people_exact, people_fuzzy o llm.

    Relación entre auto_redact y manual_review_required:

        Regex:
            auto_redact=True
            manual_review_required=False

        Personas conocidas:
            - full_name, initial_lastname y email_user:
                auto_redact=True.
                manual_review_required=True solo si el score fuzzy está por debajo
                de fuzzy_review_threshold.

            - first_name y last_name:
                auto_redact=False.
                manual_review_required=True siempre.

        LLM:
            manual_review_required lo decide el LLM.
            auto_redact se calcula como el inverso:
                auto_redact = not manual_review_required

            Por tanto, un hallazgo LLM dudoso se informa pero no se sustituye
            automáticamente.

    Nota:
        Las posiciones start/end de cada match corresponden al texto de la fase en
        la que se detectó el match, no necesariamente al texto original completo.    
    """

    _NON_ALLOWED_RE = re.compile(r"[^a-z0-9\s\.]")
    _MULTISPACE_RE = re.compile(r"\s+")

    LLM_ALLOWED_ENTITY_TYPES = {
        "PERSONA",
        "EMAIL",
        "TELEFONO",
        "DNI_NIE",
        "IBAN",
        "DIRECCION",
        "ROL",
        "IMPORTE",
        "URL",
        "OTRO"
    }

    DEFAULT_LLM_BATCH_SIZE = 10

    DEFAULT_LLM_DETECTION_PROMPT = """
Eres un detector de datos sensibles en texto en español.

Tu tarea es identificar fragmentos del texto que puedan revelar la identidad
de una persona o exponer información sensible que deba anonimizarse.

Este detector se ejecuta después de una primera fase de anonimización automática
por expresiones regulares y por una lista de personas conocidas. Por tanto, tu
objetivo principal es encontrar posibles fugas de información sensible que sigan
apareciendo en el texto.

Instrucciones:
- No reescribas el texto.
- No resumas.
- No expliques tus decisiones.
- Devuelve únicamente JSON válido.
- Vas a recibir una lista JSON de elementos con esta forma:
  [{"id":"...", "text":"..."}, ...]
- Detecta solo fragmentos que aparezcan literalmente en el campo text de cada
  elemento.
- Cada hallazgo debe corresponder a un substring exacto del text recibido para
  ese id, que ya puede contener marcadores de anonimización de fases anteriores.
- Conserva exactamente espacios, signos de puntuación y saltos de línea del
  fragmento detectado.
- No normalices, traduzcas, corrijas ni completes fragmentos.
- No devuelvas entidades inferidas si no aparecen literalmente en el texto.
- Este texto ya ha pasado por una primera anonimización automática. No devuelvas
  como hallazgo ningún fragmento que ya haya sido sustituido por un marcador de
  anonimización.
- Ignora siempre cualquier fragmento que coincida total o parcialmente con:
  [PERSONA], [PERSONA:...], [EMAIL], [DNI_NIE], [IBAN], [URL], [IMPORTE],
  [TELEFONO], [DIRECCION], [ROL], [OTRO] o [REVISAR_LLM].
- Tu tarea es detectar únicamente datos sensibles residuales que todavía
  aparezcan en claro después de esa primera anonimización.
- Si no encuentras nada en un elemento, devuelve "matches": [] para ese id.
- Cuando haya varios datos sensibles dentro de una misma frase, devuelve cada dato 
  como un match separado. No devuelvas la frase completa si contiene varias entidades.    

Tipos de entidad:
- PERSONA: nombres de personas o referencias claramente personales.
- EMAIL: direcciones de email.
- TELEFONO: números de teléfono.
- DNI_NIE: documentos de identidad españoles.
- IBAN: cuentas bancarias IBAN.
- DIRECCION: direcciones postales.
- ROL: cargos, puestos, departamentos o referencias profesionales que puedan
  identificar indirectamente a una persona.
- IMPORTE: cantidades económicas, salarios, compensaciones, precios u otros importes sensibles.
- URL: enlaces, dominios o páginas web que puedan revelar información sensible.
- OTRO: otros datos sensibles o identificativos.

Marca "manual_review_required": false cuando:
- El fragmento sea un nombre completo de persona.
- El fragmento sea un nombre y apellido, aunque haya pequeñas dudas.
- El fragmento sea una dirección de email.
- El fragmento sea un teléfono completo.
- El fragmento sea un DNI, NIE, pasaporte u otro identificador personal.
- El fragmento sea un IBAN o cuenta bancaria.
- El fragmento sea una dirección postal suficientemente concreta.
- El fragmento sea una URL personal, privada o corporativa específica.
- El fragmento sea un importe económico asociado a una persona, salario, compensación, bonus, indemnización o factura.
- El fragmento sea claramente identificativo dentro del contexto del texto.

Marca "manual_review_required": true solo cuando:
- El fragmento sea una palabra común que podría no ser una entidad sensible.
- El fragmento sea únicamente un nombre de pila muy común sin apellido ni contexto identificativo.
- El fragmento sea únicamente un apellido aislado sin contexto identificativo.
- El fragmento sea un cargo, rol, departamento, equipo o ubicación que solo podría identificar indirectamente a alguien.
- El fragmento esté incompleto o sea demasiado ambiguo para sustituirlo automáticamente.

Regla de decisión:
- Ante un dato personal directo, prioriza anonimizar automáticamente.
- Ante una referencia indirecta o ambigua, prioriza revisión manual.
- No marques nombres completos como revisión manual salvo que claramente no sean personas.

Importante:
- Si "manual_review_required" es false, el sistema sustituirá automáticamente el
  fragmento.
- Si "manual_review_required" es true, el sistema no sustituirá automáticamente el
  fragmento y lo dejará para revisión humana.

Direcciones postales:
- Detecta direcciones postales completas como calles, avenidas, plazas, caminos
  o similares cuando incluyan datos suficientemente identificativos como número,
  piso, portal, código postal o localidad.
- Las direcciones completas deben marcarse con "manual_review_required": false.
- Las direcciones parciales o ambiguas solo deben detectarse si el contexto sugiere
  que podrían identificar a una persona, y en ese caso deben marcarse con
  "manual_review_required": true.

Ejemplos de direcciones completas:
- "Avenida de la Albufera 114, portal 2, bajo A, 28038"
- "Calle Mayor 12, 3o izquierda, 28013 Madrid"
- "Plaza de España, 5"
    
Formato de salida:
{{
  "results": [
    {{
      "id": "id_original",
      "matches": [
        {{
          "matched_fragment": "texto exacto encontrado",
          "entity_type": "PERSONA",
          "risk_level": "high",
          "manual_review_required": false
        }}
      ]
    }}
  ]
}}

Valores permitidos:
- entity_type: PERSONA, EMAIL, TELEFONO, DNI_NIE, IBAN, DIRECCION, IMPORTE, URL, ROL u OTRO.
- risk_level: low, medium o high.
- manual_review_required: true o false. Debe ser booleano JSON, no texto.

Items JSON:
<<<
{items_json}
>>>
""".strip()

    DEFAULT_REGEX_PATTERNS = [
        {
            "type": "EMAIL",
            "pattern": r"(?<![\w\.-])[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}(?![\w\.-])",
            "replacement": "[EMAIL]",
            "risk_level": "medium"
        },
        {
            "type": "TELEFONO",
            "pattern": (
                r"(?<!\w)(?:\+\d{1,3}[\s-]?)?"
                r"(?:(?:6|7|8|9)\d{8}|(?:6|7|8|9)\d{2}(?:[\s-]?\d{3}){2})"
                r"(?!\w)"
            ),
            "replacement": "[TELEFONO]",
            "risk_level": "medium"
        },
        {
            "type": "DNI_NIE",
            "pattern": r"\b(?:\d{8}[A-Za-z]|[XYZxyz]\d{7}[A-Za-z])\b",
            "replacement": "[DNI_NIE]",
            "risk_level": "high"
        },
        {
            "type": "URL",
            "pattern": r"https?://\S+|www\.\S+",
            "replacement": "[URL]",
            "risk_level": "low"
        },
        {
            "type": "DIRECCION",
            "pattern": (
                r"(?<!\w)(?:calle|c/|avenida|avda\.?|plaza|paseo|camino|"
                r"carretera|ronda|travesia|trav\.?|via|gran via)\s+"
                r"[A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ./-]+"
                r"(?:\s+[A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ./-]+)*"
                r"(?:\s+|,\s*)\d+[A-Za-z]?"
                r"(?:,\s*[^,\n]+){0,4}"
                r"(?:,\s*\d{5})?"
                r"(?!\w)"
            ),
            "replacement": "[DIRECCION]",
            "risk_level": "high"
        },
        {
            "type": "IMPORTE",
            "pattern": (
                r"(?<!\w)(?:\d{1,3}(?:[\.\s]\d{3})+|\d+)"
                r"(?:,\d{2})?\s?(?:€|EUR|euros?)"
                r"(?!\w)"
            ),
            "replacement": "[IMPORTE]",
            "risk_level": "medium"
        }
    ]

    def __init__(
        self,
        regex_patterns = None,
        full_name_threshold=90,
        email_user_threshold=90,
        initial_lastname_threshold=95,
        first_name_threshold=100,
        last_name_threshold=95,
        fuzzy_review_threshold=95,
        enable_llm_detection=False,
        llm_client=None,
        llm_detection_prompt=None,
        llm_batch_size=DEFAULT_LLM_BATCH_SIZE
    ):
        self.regex_patterns = deepcopy(
            regex_patterns
            if regex_patterns is not None
            else self.DEFAULT_REGEX_PATTERNS
        )        
        self.full_name_threshold = full_name_threshold
        self.email_user_threshold = email_user_threshold
        self.initial_lastname_threshold = initial_lastname_threshold
        self.first_name_threshold = first_name_threshold
        self.last_name_threshold = last_name_threshold
        self.fuzzy_review_threshold = fuzzy_review_threshold
        self.enable_llm_detection = bool(enable_llm_detection)
        self.llm_client = llm_client
        self.llm_detection_prompt = (
            str(llm_detection_prompt).strip()
            if llm_detection_prompt
            else self.DEFAULT_LLM_DETECTION_PROMPT
        )
        self.set_llm_batch_size(llm_batch_size)

    def add_regex_pattern(self, entity_type, regex_pattern, replacement, risk_level="high"):
        pattern_config = {
            "type": str(entity_type),
            "pattern": str(regex_pattern),
            "replacement": str(replacement),
            "risk_level": str(risk_level)
        }

        for index, existing_pattern in enumerate(self.regex_patterns):
            if existing_pattern["type"] == pattern_config["type"]:
                self.regex_patterns[index] = pattern_config
                return

        self.regex_patterns.append(pattern_config)

    def del_regex_pattern(self, entity_type):
        pattern_type = str(entity_type)
        original_count = len(self.regex_patterns)
        self.regex_patterns = [
            pattern_config
            for pattern_config in self.regex_patterns
            if pattern_config["type"] != pattern_type
        ]
        return len(self.regex_patterns) < original_count

    def set_llm_detection_prompt(self, prompt):
        prompt = str(prompt).strip()

        if not prompt:
            raise ValueError("prompt vacio")

        self.llm_detection_prompt = prompt

    def set_llm_client(self, llm_client):
        self.llm_client = llm_client

    def enable_llm_step(self, enabled=True):
        self.enable_llm_detection = bool(enabled)

    def get_llm_batch_size(self):
        return self._llm_batch_size

    def set_llm_batch_size(self, llm_batch_size):
        if not isinstance(llm_batch_size, int) or llm_batch_size <= 0:
            raise ValueError("llm_batch_size must be an integer greater than 0")
        self._llm_batch_size = llm_batch_size


    def anonymize(self, text, people=None):
        """
        Pipeline principal:

        1. Sustituye primero patrones estructurados.
        2. Sobre el texto resultante, detecta y sustituye personas conocidas.
        3. Opcionalmente, llama a un detector LLM para encontrar fugas.

        people debe ser una lista de diccionarios:
        [
            {
                "employee_id": "EMP001",
                "name": "María",
                "lastname": "Gómez",
                "email": "maria.gomez@empresa.com"
            }
        ]
        """
        batch_results = self.anonymize_batch(
            [{"id": "0", "text": str(text)}],
            people=people
        )
        if not batch_results:
            return {
                "original_text": str(text),
                "text_after_regex": str(text),
                "text_after_people": str(text),
                "anonymized_text": str(text),
                "regex_matches": [],
                "people_matches": [],
                "llm_matches": [],
                "llm_detection_skipped": True,
                "llm_error": None,
                "matches": [],
                "manual_review_required": False
            }
        return batch_results[0]

    def anonymize_batch(self, items, people=None):
        if people is None:
            people = []
        if items is None:
            items = []
        if not isinstance(items, list):
            raise ValueError("items debe ser una lista")

        prepared = []
        llm_input_items = []

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            item_id = str(item.get("id", index))
            text = str(item.get("text", ""))

            regex_result = self.anonymize_regex_entities(text)
            text_after_regex = regex_result["anonymized_text"]

            people_result = self.anonymize_people(text_after_regex, people)
            text_after_people = people_result["anonymized_text"]

            prepared_item = {
                "id": item_id,
                "original_text": text,
                "text_after_regex": text_after_regex,
                "text_after_people": text_after_people,
                "regex_matches": regex_result["matches"],
                "people_matches": people_result["matches"],
                "_base_manual_review_required": (
                    regex_result["manual_review_required"]
                    or people_result["manual_review_required"]
                ),
            }
            prepared.append(prepared_item)
            llm_input_items.append({"id": item_id, "text": text_after_people})

        llm_results_by_id = self.anonymize_llm_entities_batch(llm_input_items)

        results = []
        for prepared_item in prepared:
            llm_result = llm_results_by_id.get(prepared_item["id"], {
                "anonymized_text": prepared_item["text_after_people"],
                "matches": [],
                "manual_review_required": False,
                "llm_detection_skipped": True,
                "llm_error": None
            })

            all_matches = (
                prepared_item["regex_matches"]
                + prepared_item["people_matches"]
                + llm_result["matches"]
            )

            results.append({
                "id": prepared_item["id"],
                "original_text": prepared_item["original_text"],
                "text_after_regex": prepared_item["text_after_regex"],
                "text_after_people": prepared_item["text_after_people"],
                "anonymized_text": llm_result["anonymized_text"],
                "regex_matches": prepared_item["regex_matches"],
                "people_matches": prepared_item["people_matches"],
                "llm_matches": llm_result["matches"],
                "llm_detection_skipped": llm_result.get("llm_detection_skipped", False),
                "llm_error": llm_result.get("llm_error"),
                "matches": all_matches,
                "manual_review_required": (
                    prepared_item["_base_manual_review_required"]
                    or llm_result["manual_review_required"]
                )
            })

        return results

    def anonymize_regex_entities(self, text):
        """
        Detecta y sustituye entidades estructuradas mediante regex.
        """
        text = str(text)
        matches = self.detect_regex_entities(text)

        spans_to_redact = [
            {
                "start": match["start"],
                "end": match["end"],
                "replacement": match["replacement"]
            }
            for match in matches
            if match.get("auto_redact", True)
        ]

        anonymized_text = self._replace_spans(text, spans_to_redact)

        return {
            "original_text": text,
            "anonymized_text": anonymized_text,
            "matches": matches,
            "manual_review_required": any(
                match.get("manual_review_required", False)
                for match in matches
            )
        }

    def anonymize_people(self, text, people):
        """
        Detecta y sustituye personas conocidas en el texto recibido.

        Este método asume que antes ya se han sustituido las expresiones
        regulares de emails, teléfonos, DNI/NIE, URLs, etc. 
        Para el pipeline completo, usa anonymize().
        """
        text = str(text)

        people_matches = self.detect_people(text, people)
        people_matches = self._deduplicate_overlapping_matches(people_matches)

        spans_to_redact = [
            {
                "start": match["start"],
                "end": match["end"],
                "replacement": match["replacement"]
            }
            for match in people_matches
            if match.get("auto_redact", True)
        ]

        anonymized_text = self._replace_spans(text, spans_to_redact)

        manual_review_required = any(
            match.get("manual_review_required", False)
            for match in people_matches
        )

        return {
            "original_text": text,
            "anonymized_text": anonymized_text,
            "matches": people_matches,
            "manual_review_required": manual_review_required
        }

    def anonymize_llm_entities(self, text):
        """
        Detecta entidades sensibles adicionales mediante un LLM.

        Este paso es opcional. Los hallazgos marcados por el LLM con
        manual_review_required=False se sustituyen automáticamente. Los hallazgos
        marcados con manual_review_required=True se devuelven para revisión manual
        y no se sustituyen.        
        """
        text = str(text)
        batch_result = self.anonymize_llm_entities_batch([{"id": "0", "text": text}])
        return batch_result.get("0", {
            "original_text": text,
            "anonymized_text": text,
            "matches": [],
            "manual_review_required": False,
            "llm_detection_skipped": True,
            "llm_error": None
        })

    def anonymize_llm_entities_batch(self, items):
        if items is None:
            items = []

        prepared = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", index))
            text = str(item.get("text", ""))
            prepared.append({"id": item_id, "text": text})

        if not prepared:
            return {}

        if not self.enable_llm_detection:
            return {
                item["id"]: {
                    "original_text": item["text"],
                    "anonymized_text": item["text"],
                    "matches": [],
                    "manual_review_required": False,
                    "llm_detection_skipped": True,
                    "llm_error": None
                }
                for item in prepared
            }

        if self.llm_client is None:
            raise ValueError("llm_client no configurado")

        try:
            detected_by_id = self.detect_llm_entities_batch(prepared)
            llm_error = None
            llm_detection_skipped = False
        except Exception as exc:
            detected_by_id = {}
            llm_error = str(exc)
            llm_detection_skipped = True

        results = {}
        for item in prepared:
            item_id = item["id"]
            text = item["text"]
            llm_matches = self._deduplicate_overlapping_matches(
                detected_by_id.get(item_id, [])
            )

            spans_to_redact = [
                {
                    "start": match["start"],
                    "end": match["end"],
                    "replacement": match["replacement"]
                }
                for match in llm_matches
                if match.get("auto_redact", True)
            ]
            anonymized_text = self._replace_spans(text, spans_to_redact)

            results[item_id] = {
                "original_text": text,
                "anonymized_text": anonymized_text,
                "matches": llm_matches,
                "llm_detection_skipped": llm_detection_skipped,
                "llm_error": llm_error,
                "manual_review_required": any(
                    match.get("manual_review_required", False)
                    for match in llm_matches
                )
            }

        return results

    def detect_regex_entities(self, text):
        """
        Detecta entidades estructuradas mediante regex.
        """
        text = str(text)
        matches = []

        for pattern_config in self.regex_patterns:
            for match in re.finditer(
                pattern_config["pattern"],
                text,
                flags=re.IGNORECASE
            ):
                matches.append({
                    "entity_type": pattern_config["type"],
                    "matched_fragment": match.group(),
                    "matched_fragment_normalized": self._normalize_text(match.group()),
                    "score": 100,
                    "auto_redact": True,
                    "manual_review_required": False,
                    "replacement": pattern_config["replacement"],
                    "risk_level": pattern_config["risk_level"],
                    "start": match.start(),
                    "end": match.end(),
                    "source": "regex"
                })

        return self._deduplicate_overlapping_matches(matches)

    def detect_people(self, text, people):
        """
        Detecta todas las menciones de personas conocidas en un texto.
        """
        text = str(text)
        ngrams = self._generate_ngrams_with_positions(text)
        matches = []

        for person in people:
            variants = self._build_person_variants(person)
            employee_id = str(person.get("employee_id", "")).strip()

            replacement = (
                f"[PERSONA:{employee_id}]"
                if employee_id
                else "[PERSONA]"
            )

            for variant in variants:
                variant_matches = self._find_all_matches(
                    variant["variant"],
                    ngrams,
                    threshold=variant["threshold"]
                )

                for match in variant_matches:
                    is_fuzzy = match["score"] < 100
                    matches.append({
                        "entity_type": "PERSONA",
                        "employee_id": person.get("employee_id"),
                        "name": person.get("name"),
                        "lastname": person.get("lastname"),
                        "variant": variant["variant"],
                        "variant_type": variant["type"],
                        "matched_fragment": match["matched_fragment"],
                        "matched_fragment_normalized": match["matched_fragment_normalized"],
                        "score": match["score"],
                        "auto_redact": variant["auto_redact"],
                        "manual_review_required": (
                            not variant["auto_redact"]
                            or match["score"] < self.fuzzy_review_threshold
                        ),
                        "replacement": replacement,
                        "risk_level": "medium",
                        "start": match["start"],
                        "end": match["end"],
                        "source": "people_fuzzy" if is_fuzzy else "people_exact"
                    })
                    
        return self._deduplicate_overlapping_matches(matches)

    def detect_llm_entities(self, text):
        """
        Pide a un LLM entidades sensibles adicionales sobre el texto recibido.
        """
        text = str(text)
        results = self.detect_llm_entities_batch([{"id": "0", "text": text}])
        return self._deduplicate_overlapping_matches(results.get("0", []))

    def detect_llm_entities_batch(self, items):
        prepared = []
        for index, item in enumerate(items or []):
            if not isinstance(item, dict):
                continue
            prepared.append({
                "id": str(item.get("id", index)),
                "text": str(item.get("text", ""))
            })

        if not prepared:
            return {}

        matches_by_id = {item["id"]: [] for item in prepared}
        text_by_id = {item["id"]: item["text"] for item in prepared}

        for batch in self._iter_batches(prepared, self._llm_batch_size):
            items_json = json.dumps(batch, ensure_ascii=False)
            prompt = str(self.llm_detection_prompt).replace("{items_json}", items_json)
            print(".", end="")
            response = self.llm_client.generate(prompt)
            payload = self._parse_llm_batch_json_response(response)

            for result_item in payload.get("results", []):
                item_id = str(result_item.get("id", ""))

                if item_id not in text_by_id:
                    continue

                text = text_by_id[item_id]
                for candidate in result_item.get("matches", []):
                    if not isinstance(candidate, dict):
                        continue

                    matched_fragment = str(candidate.get("matched_fragment", ""))
                    if not matched_fragment.strip():
                        continue

                    manual_review_required = candidate.get("manual_review_required", True)
                    if not isinstance(manual_review_required, bool):
                        manual_review_required = True

                    entity_type = str(
                        candidate.get("entity_type", "OTRO")
                    ).strip().upper() or "OTRO"
                    if entity_type not in self.LLM_ALLOWED_ENTITY_TYPES:
                        entity_type = "OTRO"

                    replacement = (
                        "[REVISAR_LLM]"
                        if entity_type == "OTRO"
                        else f"[{entity_type}]"
                    )

                    risk_level = str(
                        candidate.get("risk_level", "medium")
                    ).strip().lower() or "medium"
                    if risk_level not in {"low", "medium", "high"}:
                        risk_level = "medium"

                    positions = self._find_literal_occurrences(text, matched_fragment)
                    for start, end in positions:
                        matches_by_id[item_id].append({
                            "entity_type": entity_type,
                            "matched_fragment": matched_fragment,
                            "matched_fragment_normalized": self._normalize_text(
                                matched_fragment
                            ),
                            "score": 100,
                            "auto_redact": not manual_review_required,
                            "manual_review_required": manual_review_required,
                            "replacement": replacement,
                            "risk_level": risk_level,
                            "start": start,
                            "end": end,
                            "source": "llm"
                        })

        return {
            item_id: self._deduplicate_overlapping_matches(item_matches)
            for item_id, item_matches in matches_by_id.items()
        }

    def _find_all_matches(self, person_name, ngrams, threshold=90):
        """
        Devuelve todas las coincidencias cuyo score supera el umbral.
        """
        matches = []

        for fragment in ngrams:
            score = self._fuzzy_score(
                person_name,
                fragment["normalized"]
            )

            if score >= threshold:
                matches.append({
                    "person_name": person_name,
                    "matched_fragment": fragment["original"],
                    "matched_fragment_normalized": fragment["normalized"],
                    "score": score,
                    "start": fragment["start"],
                    "end": fragment["end"]
                })

        return matches

    def _iter_batches(self, values, batch_size):
        for start_index in range(0, len(values), batch_size):
            yield values[start_index:start_index + batch_size]

    def _find_literal_occurrences(self, text, fragment):
        """
        Devuelve todas las posiciones en las que aparece literalmente fragment.
        """
        positions = []
        start = 0

        index = text.find(fragment, start)
        while (index != -1):
            positions.append((index, index + len(fragment)))
            start = index + len(fragment)
            index = text.find(fragment, start)

        return positions

    def _fuzzy_score(self, a, b):
        return max(
            fuzz.ratio(a, b),
            fuzz.token_sort_ratio(a, b)
        )

    def _tokenize_with_positions(self, text):
        """
        Tokeniza conservando posiciones en el texto original.
        Incluye letras con tildes, ñ, números, puntos y guiones.
        """
        tokens = []
        token_pattern = r"(?<![\w.-])\w+(?:[.-]\w+)*(?![\w.-])"

        for match in re.finditer(token_pattern, text):
            original = match.group()
            normalized = self._normalize_text(original)

            if normalized:
                tokens.append({
                    "original": original,
                    "normalized": normalized,
                    "start": match.start(),
                    "end": match.end()
                })

        return tokens

    def _generate_ngrams_with_positions(self, text, min_n=1, max_n=4):
        """
        Genera ngrams con:
        - texto original
        - texto normalizado
        - posición inicial
        - posición final
        """
        tokens = self._tokenize_with_positions(text)
        ngrams = []

        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                selected = tokens[i:i + n]

                ngrams.append({
                    "original": text[selected[0]["start"]:selected[-1]["end"]],
                    "normalized": " ".join(
                        token["normalized"] for token in selected
                    ),
                    "start": selected[0]["start"],
                    "end": selected[-1]["end"]
                })

        return ngrams

    def _replace_spans(self, text, spans):
        """
        Reemplaza fragmentos por posición.
        Se ordena de derecha a izquierda para no romper los índices.
        """
        if not spans:
            return text

        spans = self._merge_duplicate_spans(spans)
        spans = sorted(spans, key=lambda span: span["start"], reverse=True)

        result = text

        for span in spans:
            result = (
                result[:span["start"]]
                + span["replacement"]
                + result[span["end"]:]
            )

        return result

    def _merge_duplicate_spans(self, spans):
        """
        Elimina spans duplicados exactos.
        """
        seen = set()
        unique = []

        for span in spans:
            key = (span["start"], span["end"], span["replacement"])

            if key not in seen:
                seen.add(key)
                unique.append(span)

        return unique

    def _deduplicate_overlapping_matches(self, matches):
        """
        Elimina coincidencias solapadas.

        Prioridad:
        1. Mayor longitud del fragmento.
        2. Mayor score.
        3. Regex antes que fuzzy.
        4. full_name/email_user antes que first_name/last_name.
        5. auto_redact=True.
        """
        if not matches:
            return []

        def source_priority(match):
            if match.get("entity_type") == "EMAIL":
                return 5
            if match.get("source") == "regex":
                return 4
            if match.get("variant_type") == "full_name":
                return 3
            if match.get("variant_type") == "email_user":
                return 3
            if match.get("variant_type") == "initial_lastname":
                return 2
            return 1

        sorted_matches = sorted(
            matches,
            key=lambda m: (
                -(m["end"] - m["start"]),
                -m.get("score", 0),
                -source_priority(m),
                not m.get("auto_redact", True),
                m["start"]
            )
        )

        selected = []

        for match in sorted_matches:
            overlaps = any(
                match["end"] > chosen["start"]
                and match["start"] < chosen["end"]
                for chosen in selected
            )

            if not overlaps:
                selected.append(match)

        return sorted(selected, key=lambda m: m["start"])

    def _normalize_text(self, text):
        text = unicodedata.normalize("NFD", str(text).lower())
        text = "".join(
            char for char in text
            if not unicodedata.combining(char)
        )
        text = self._NON_ALLOWED_RE.sub(" ", text)
        return self._MULTISPACE_RE.sub(" ", text).strip()

    def _build_person_variants(self, person):
        """
        Genera variantes de búsqueda para una persona.

        name + lastname:
            Se anonimiza automáticamente si supera el umbral.

        email_user:
            Ejemplo: maria.gomez -> maria gomez.
            Se anonimiza automáticamente si supera el umbral.

        name:
            Solo se informa, no se anonimiza automáticamente.

        last_name:
            Solo se informa, no se anonimiza automáticamente.
        """
        name = self._normalize_text(person.get("name", ""))
        lastname = self._normalize_text(person.get("lastname", ""))

        raw_email = str(person.get("email", "")).strip()
        email_user = raw_email.split("@")[0] if "@" in raw_email else ""
        email_user = self._normalize_text(email_user)

        variants = []

        if name and lastname:
            full_name = f"{name} {lastname}"

            variants.append({
                "variant": full_name,
                "type": "full_name",
                "threshold": self.full_name_threshold,
                "auto_redact": True
            })

            initial_lastname = f"{name[0]} {lastname}"

            variants.append({
                "variant": initial_lastname,
                "type": "initial_lastname",
                "threshold": self.initial_lastname_threshold,
                "auto_redact": True
            })

        if email_user:
            variants.append({
                "variant": email_user.replace(".", " "),
                "type": "email_user",
                "threshold": self.email_user_threshold,
                "auto_redact": True
            })

        if name:
            variants.append({
                "variant": name,
                "type": "first_name",
                "threshold": self.first_name_threshold,
                "auto_redact": False
            })

        if lastname:
            variants.append({
                "variant": lastname,
                "type": "last_name",
                "threshold": self.last_name_threshold,
                "auto_redact": False
            })

        return variants

    def _parse_llm_json_response(self, response):
        """
        Extrae el JSON de la respuesta del LLM.
        """
        response = str(response).strip()

        if not response:
            return {"matches": []}

        fenced_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            response,
            flags=re.DOTALL
        )

        if fenced_match:
            response = fenced_match.group(1)

        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("respuesta LLM no es JSON valido") from exc

        if not isinstance(payload, dict):
            raise ValueError("respuesta LLM invalida")

        matches = payload.get("matches", [])

        if not isinstance(matches, list):
            raise ValueError("respuesta LLM invalida: matches debe ser lista")

        return payload

    def _parse_llm_batch_json_response(self, response):
        payload = self._parse_llm_json_response(response)

        if "results" not in payload:
            # Compatibilidad defensiva con payload legacy de un único texto.
            if "matches" in payload:
                return {
                    "results": [
                        {
                            "id": "0",
                            "matches": payload.get("matches", [])
                        }
                    ]
                }
            raise ValueError("respuesta LLM invalida: falta results")

        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError("respuesta LLM invalida: results debe ser lista")

        normalized_results = []
        for result_item in results:
            if not isinstance(result_item, dict):
                continue
            result_id = str(result_item.get("id", "")).strip()
            result_matches = result_item.get("matches", [])
            if not isinstance(result_matches, list):
                continue
            if not result_id:
                continue
            normalized_results.append({
                "id": result_id,
                "matches": result_matches
            })

        return {"results": normalized_results}
    
