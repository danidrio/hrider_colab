import pandas as pd

from hrider.anonymizer.anonymizer import Anonymizer


class ExcelAnonymizer:
    """
    Anonimiza celdas de un Excel usando el pipeline de Anonymizer.

    Permite activar/desactivar LLM por columna.
    """

    def __init__(self, anonymizer=None):
        self.anonymizer = anonymizer or Anonymizer()

    def anonymize_excel(
        self,
        input_path,
        output_path,
        people=None,
        column_config=None,
        full_name_threshold=None,
        email_user_threshold=None,
        initial_lastname_threshold=None,
        first_name_threshold=None,
        last_name_threshold=None,
        fuzzy_review_threshold=None,
        store_matches=True,
    ):
        """
        Anonimiza un fichero Excel y guarda una copia anonimizada.

        Parámetros:
        - column_config:
          Array de configuraciones por hoja en orden ordinal.
          La posición 0 aplica a la primera hoja, la 1 a la segunda, etc.
          Ejemplo:
          [
            {
              "comments": {"anonymize": True, "llm_detection": True},
              "name": {"anonymize": False}
            },
            {
              "notes": {"llm_detection": True}
            }
          ]
          - anonymize:
            Si es False, la columna se omite.
            Si no se especifica, se anonimiza por defecto.
          - llm_detection:
            Si no se especifica, False por defecto.
        - Los thresholds son globales para todas las columnas procesadas.
        """
        people = people or []
        column_config = list(column_config or [])

        excel_file = pd.ExcelFile(str(input_path))
        output_sheets = {}
        sheet_results = []

        for sheet_index, sheet_name in enumerate(excel_file.sheet_names):
            dataframe = excel_file.parse(sheet_name=sheet_name, dtype=object)
            sheet_config = (
                column_config[sheet_index]
                if sheet_index < len(column_config)
                and isinstance(column_config[sheet_index], dict)
                else {}
            )

            sheet_result = self._anonymize_sheet(
                dataframe=dataframe,
                sheet_index=sheet_index,
                sheet_name=sheet_name,
                sheet_config=sheet_config,
                people=people,
                thresholds={
                    "full_name_threshold": full_name_threshold,
                    "email_user_threshold": email_user_threshold,
                    "initial_lastname_threshold": initial_lastname_threshold,
                    "first_name_threshold": first_name_threshold,
                    "last_name_threshold": last_name_threshold,
                    "fuzzy_review_threshold": fuzzy_review_threshold,
                },
                store_matches=store_matches,
            )

            output_sheets[sheet_name] = sheet_result["dataframe"]
            sheet_results.append(sheet_result["sheet_result"])

        with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
            for sheet_name, dataframe in output_sheets.items():
                dataframe.to_excel(writer, index=False, sheet_name=sheet_name)

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "sheets": sheet_results,
            "manual_review_required": any(
                sheet_result["manual_review_required"]
                for sheet_result in sheet_results
            ),
        }

    def _anonymize_sheet(
        self,
        dataframe,
        sheet_index,
        sheet_name,
        sheet_config,
        people,
        thresholds,
        store_matches,
    ):
        dataframe = dataframe.copy()
        column_results = []

        for column_name in dataframe.columns:
            config = sheet_config.get(column_name, {})
            
            anonymize_column = bool(config.get("anonymize", True))

            if not anonymize_column:
                column_results.append({
                    "column_name": str(column_name),
                    "configured": column_name in sheet_config,
                    "processed": False,
                    "reason": "anonymization_disabled",
                    "cells_processed": 0,
                    "cells_changed": 0,
                    "manual_review_required": False,
                    "anonymize": False,
                    "llm_detection": bool(config.get("llm_detection", False)),
                })
                continue

            llm_detection = bool(config.get("llm_detection", False))
            column_anonymizer = self._build_column_anonymizer(
                llm_detection=llm_detection,
                thresholds=thresholds,
            )

            processed = 0
            changed = 0
            manual_review_required = False
            matches = []
            rows_with_matches_count = 0

            for row_index, value in dataframe[column_name].items():
                if not isinstance(value, str) or not value.strip:
                    continue

                processed += 1
                result = column_anonymizer.anonymize(value, people=people)
                dataframe.at[row_index, column_name] = result["anonymized_text"]

                if result["anonymized_text"] != value:
                    changed += 1

                if result.get("manual_review_required", False):
                    manual_review_required = True

                if result.get("matches"):
                    rows_with_matches_count += 1

                    if store_matches:
                        matches.append({
                            "row_index": int(row_index),
                            "matches": result["matches"],
                        })

            column_result = {
                "column_name": str(column_name),
                "configured": column_name in sheet_config,
                "processed": True,
                "cells_processed": processed,
                "cells_changed": changed,
                "manual_review_required": manual_review_required,
                "anonymize": True,
                "llm_detection": llm_detection,
            }

            if store_matches:
                column_result["rows_with_matches"] = matches
            else:
                column_result["rows_with_matches_count"] = rows_with_matches_count

            column_results.append(column_result)

        return {
            "dataframe": dataframe,
            "sheet_result": {
                "sheet_index": int(sheet_index),
                "sheet_name": sheet_name,
                "columns": column_results,
                "manual_review_required": any(
                    column_result.get("manual_review_required", False)
                    for column_result in column_results
                    if column_result.get("processed")
                ),
            },
        }

    def _build_column_anonymizer(self, llm_detection, thresholds):
        column_anonymizer = self.anonymizer
        column_anonymizer.enable_llm_step(llm_detection)

        for key, value in thresholds.items():
            if value is None:
                continue
            setattr(column_anonymizer, key, value)

        return column_anonymizer
