"""CSV downloads with spreadsheet formula escaping for untrusted text cells."""

import csv
import io


def csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    """Write UTF-8 with BOM for Excel; filenames cannot become spreadsheet formulas."""
    def safe(value: object) -> object:
        if isinstance(value, str) and (value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n"))):
            return "'" + value
        return value

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: safe(row.get(key, "")) for key in fields})
    return output.getvalue().encode("utf-8-sig")
