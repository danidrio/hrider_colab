from collections import Counter
from html import escape


class ExcelAnonymizationReport:
    """
    Genera informes para el resultado de ExcelAnonymizer.anonymize_excel().
    """

    DEFAULT_ENTITY_ORDER = (
        "PERSONA",
        "EMAIL",
        "TELEFONO",
        "DNI_NIE",
        "IBAN",
        "DIRECCION",
        "ROL",
        "IMPORTE",
        "URL",
        "OTRO",
    )

    def __init__(self, excel_result, entity_order=None):
        if not isinstance(excel_result, dict):
            raise TypeError(
                "excel_result debe ser el diccionario devuelto por anonymize_excel"
            )

        self.excel_result = excel_result
        self.entity_order = tuple(entity_order or self.DEFAULT_ENTITY_ORDER)

    def to_dict(self):
        sheets = self._build_sheet_rows()
        total_sheets = len(sheets)
        sheets_requiring_review = sum(
            1 for sheet in sheets if sheet["manual_review_required"]
        )
        sheets_with_changes = sum(
            1 for sheet in sheets if sheet["cells_changed"] > 0
        )
        total_columns = sum(sheet["total_columns"] for sheet in sheets)
        total_columns_processed = sum(sheet["processed_columns"] for sheet in sheets)
        total_cells_processed = sum(sheet["cells_processed"] for sheet in sheets)
        total_cells_changed = sum(sheet["cells_changed"] for sheet in sheets)
        total_rows_with_matches = sum(
            sheet["rows_with_matches"] for sheet in sheets
        )

        replacement_types = Counter()
        for sheet in sheets:
            replacement_types.update(sheet["replacement_types"])

        return {
            "input_path": self.excel_result.get("input_path"),
            "output_path": self.excel_result.get("output_path"),
            "manual_review_required": bool(
                self.excel_result.get("manual_review_required", False)
            ),
            "total_sheets": total_sheets,
            "sheets_requiring_review": sheets_requiring_review,
            "sheets_with_changes": sheets_with_changes,
            "total_columns": total_columns,
            "total_columns_processed": total_columns_processed,
            "total_cells_processed": total_cells_processed,
            "total_cells_changed": total_cells_changed,
            "total_rows_with_matches": total_rows_with_matches,
            "replacement_types": self._ordered_counter_dict(replacement_types),
            "sheets": sheets,
        }

    def save_text(self, output_path):
        return self._save_report(output_path, self.to_text())

    def save_html(self, output_path):
        return self._save_report(output_path, self.to_html())

    def to_text(self, width=110):
        data = self.to_dict()
        width = max(int(width or 110), 80)

        status = (
            "REVISION NECESARIA"
            if data["manual_review_required"]
            else "SIN REVISION"
        )

        lines = []
        lines.append("=" * width)
        lines.append("INFORME DE ANONIMIZACION EXCEL")
        lines.append("=" * width)
        lines.append(f"Excel original:    {data.get('input_path') or ''}")
        lines.append(f"Excel anonimizado: {data.get('output_path') or ''}")
        lines.append("")
        lines.append("RESUMEN")
        lines.append("-" * width)
        lines.append(f"Estado final:                  {status}")
        lines.append(f"Hojas analizadas:              {data['total_sheets']}")
        lines.append(f"Hojas con cambios:             {data['sheets_with_changes']}")
        lines.append(f"Hojas que requieren revision:  {data['sheets_requiring_review']}")
        lines.append(f"Columnas totales:              {data['total_columns']}")
        lines.append(f"Columnas procesadas:           {data['total_columns_processed']}")
        lines.append(f"Celdas procesadas:             {data['total_cells_processed']}")
        lines.append(f"Celdas modificadas:            {data['total_cells_changed']}")
        lines.append(f"Filas con matches:             {data['total_rows_with_matches']}")
        lines.append("")
        lines.append("REEMPLAZOS POR TIPO")
        lines.append("-" * width)

        if data["replacement_types"]:
            for entity_type, count in data["replacement_types"].items():
                lines.append(f"{entity_type:>12}: {count}")
        else:
            lines.append("Sin detalle por tipo (store_matches=False o sin reemplazos).")

        lines.append("")
        lines.append("DETALLE POR HOJA")
        lines.append("-" * width)

        for sheet in data["sheets"]:
            lines.append(
                f"[{sheet['sheet_index']}] {sheet['sheet_name']} | "
                f"cols: {sheet['processed_columns']}/{sheet['total_columns']} | "
                f"celdas: {sheet['cells_changed']}/{sheet['cells_processed']} | "
                f"review: {sheet['manual_review_required']}"
            )

        return "\n".join(lines).rstrip() + "\n"

    def to_html(self):
        data = self.to_dict()

        rows = []
        for sheet in data["sheets"]:
            rows.append(
                "<tr>"
                f"<td>{sheet['sheet_index']}</td>"
                f"<td>{escape(str(sheet['sheet_name']))}</td>"
                f"<td>{sheet['processed_columns']}/{sheet['total_columns']}</td>"
                f"<td>{sheet['cells_changed']}/{sheet['cells_processed']}</td>"
                f"<td>{'Si' if sheet['manual_review_required'] else 'No'}</td>"
                "</tr>"
            )

        replacement_items = []
        for entity_type, count in data["replacement_types"].items():
            replacement_items.append(
                f"<li><strong>{escape(entity_type)}</strong>: {int(count)}</li>"
            )

        replacement_html = (
            "<ul>" + "".join(replacement_items) + "</ul>"
            if replacement_items
            else "<p>Sin detalle por tipo (store_matches=False o sin reemplazos).</p>"
        )

        status = "REVISION NECESARIA" if data["manual_review_required"] else "SIN REVISION"

        return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Informe de anonimización Excel</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
main {{ max-width: 1100px; margin: 0 auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
th {{ background: #f3f4f6; }}
.kpi {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 12px 0 18px; }}
.card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 10px; background: #fff; }}
</style>
</head>
<body>
<main>
  <h1>Informe de anonimización Excel</h1>
  <p>Excel original: <code>{escape(str(data.get("input_path") or ""))}</code></p>
  <p>Excel anonimizado: <code>{escape(str(data.get("output_path") or ""))}</code></p>
  <h2>Resumen</h2>
  <div class="kpi">
    <div class="card"><strong>Estado</strong><br>{escape(status)}</div>
    <div class="card"><strong>Hojas</strong><br>{data['total_sheets']}</div>
    <div class="card"><strong>Hojas con revisión</strong><br>{data['sheets_requiring_review']}</div>
    <div class="card"><strong>Columnas procesadas</strong><br>{data['total_columns_processed']}/{data['total_columns']}</div>
    <div class="card"><strong>Celdas procesadas</strong><br>{data['total_cells_processed']}</div>
    <div class="card"><strong>Celdas modificadas</strong><br>{data['total_cells_changed']}</div>
  </div>
  <h2>Reemplazos por tipo</h2>
  {replacement_html}
  <h2>Detalle por hoja</h2>
  <table>
    <thead>
      <tr>
        <th>Índice</th>
        <th>Hoja</th>
        <th>Columnas procesadas</th>
        <th>Celdas cambiadas</th>
        <th>Revisión</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</main>
</body>
</html>"""

    def _save_report(self, output_path, content):
        output_path = str(output_path)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(str(content))
        return output_path

    def _build_sheet_rows(self):
        sheets = []
        for fallback_index, sheet in enumerate(self.excel_result.get("sheets", []) or []):
            sheet_index = int(sheet.get("sheet_index", fallback_index))
            columns = sheet.get("columns", []) or []

            processed_columns = [column for column in columns if column.get("processed")]
            cells_processed = sum(
                int(column.get("cells_processed", 0) or 0)
                for column in processed_columns
            )
            cells_changed = sum(
                int(column.get("cells_changed", 0) or 0)
                for column in processed_columns
            )
            rows_with_matches = 0
            by_entity = Counter()

            for column in processed_columns:
                if "rows_with_matches_count" in column:
                    rows_with_matches += int(column.get("rows_with_matches_count", 0) or 0)

                for row in column.get("rows_with_matches", []) or []:
                    rows_with_matches += 1
                    for match in row.get("matches", []) or []:
                        if not match.get("auto_redact", True):
                            continue
                        entity_type = str(match.get("entity_type", "OTRO")).upper() or "OTRO"
                        by_entity[entity_type] += 1

            sheets.append({
                "sheet_index": sheet_index,
                "sheet_name": sheet.get("sheet_name", f"Sheet {sheet_index + 1}"),
                "total_columns": len(columns),
                "processed_columns": len(processed_columns),
                "cells_processed": cells_processed,
                "cells_changed": cells_changed,
                "rows_with_matches": rows_with_matches,
                "manual_review_required": bool(sheet.get("manual_review_required", False)),
                "replacement_types": self._ordered_counter_dict(by_entity),
            })

        return sheets

    def _ordered_counter_dict(self, counter):
        ordered = {}
        for entity_type in self.entity_order:
            value = int(counter.get(entity_type, 0))
            if value > 0:
                ordered[entity_type] = value

        for entity_type, value in sorted(counter.items()):
            if entity_type in ordered:
                continue
            value = int(value)
            if value > 0:
                ordered[str(entity_type)] = value

        return ordered
