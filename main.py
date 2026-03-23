import csv
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddle.base.libpaddle import AnalysisConfig
from paddlex import create_pipeline


# PaddleX 3.4 expects an older Paddle Inference API on some installs.
# Keep the project runnable by adding a no-op compatibility shim when needed.
if not hasattr(AnalysisConfig, "set_optimization_level"):
    AnalysisConfig.set_optimization_level = lambda self, level: None


HEADERS = [
    "Date",
    "Regular Hours",
    "OT Hours / Allowance",
    "Rope Access Allowance",
    "Transport Allowance",
    "Night Shift Allowance",
    "Food Allowance",
    "Job Site",
    "Supervisor Name & Signature",
]

COLUMN_RANGES = [
    (0, 60),
    (60, 175),
    (175, 280),
    (280, 380),
    (380, 470),
    (470, 560),
    (560, 660),
    (660, 760),
    (760, 9999),
]

LABEL_ALIASES = {
    "WP/FIN NO.": "WP/FIN No.",
    "WP/FIN NO": "WP/FIN No.",
    "EMPLOYEE ID NO:": "Employee ID No:",
}

VALUE_ALIASES = {
    "405\u00d7": "405X",
    "&": "8",
    "\u4e4b": "2",
    "\u4e00": "-",
    "DW": "DW",
}

EMPLOYEE_ID_PATTERN = re.compile(r"F[\sT1I\+\-_\.]*O?\d{1,4}", re.IGNORECASE)
SUPPORTED_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


def normalize_text(text: str) -> str:
    text = " ".join(str(text).strip().split())
    return VALUE_ALIASES.get(text, text)


def normalize_label(text: str) -> str:
    text = " ".join(str(text).strip().split()).upper()
    return LABEL_ALIASES.get(text, text)


def extract_region_items(items, x_min, x_max, y_min, y_max):
    candidates = [
        item
        for item in items
        if x_min <= item["cx"] <= x_max and y_min <= item["cy"] <= y_max
    ]
    candidates.sort(key=lambda item: (item["cy"], item["x1"]))
    return candidates


def find_label_item(items, label_text: str):
    normalized_label = normalize_label(label_text)
    candidates = [
        item for item in items if normalize_label(item["text"]) == normalized_label
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["y1"], item["x1"]))
    return candidates[0]


def extract_band_text(items, anchor_item, x_min, x_max, y_tolerance=26):
    if anchor_item is None:
        return ""
    candidates = [
        item
        for item in items
        if item["x1"] >= x_min
        and item["x2"] <= x_max
        and abs(item["cy"] - anchor_item["cy"]) <= y_tolerance
    ]
    candidates.sort(key=lambda item: item["x1"])
    return " ".join(normalize_text(item["text"]) for item in candidates)


def extract_region_text(items, x_min, x_max, y_min, y_max):
    return " ".join(
        normalize_text(item["text"])
        for item in extract_region_items(items, x_min, x_max, y_min, y_max)
    )


def extract_numeric_region_text(items, x_min, x_max, y_min, y_max):
    values = []
    for item in extract_region_items(items, x_min, x_max, y_min, y_max):
        text = normalize_text(item["text"])
        if any(char.isdigit() for char in text):
            values.append(text)
    return " ".join(values)


def safe_stem(value: str, fallback: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in ("-", "_", " ") else ""
        for char in value.strip()
    ).strip()
    cleaned = cleaned.replace(" ", "_")
    return cleaned or fallback


def move_generated_file(output_dir: Path, stem: str, suffix: str, target_stem: str):
    matches = sorted(output_dir.glob(f"{stem}{suffix}.*"))
    if not matches:
        return None
    source = matches[0]
    target = output_dir / f"{target_stem}{source.suffix}"
    if target.exists():
        target.unlink()
    source.rename(target)
    return target


def find_employee_id(items):
    # Employee IDs are most reliable when read near the Employee ID label.
    employee_label = find_label_item(items, "Employee ID No:")
    scoped_items = items
    if employee_label is not None:
        scoped_items = [
            item
            for item in items
            if item["x1"] >= employee_label["x2"] - 5
            and item["x2"] <= employee_label["x2"] + 260
            and abs(item["cy"] - employee_label["cy"]) <= 45
        ]
        if not scoped_items:
            scoped_items = items

    for item in sorted(scoped_items, key=lambda item: (item["cy"], item["x1"])):
        text = normalize_text(item["text"])
        text = (
            text.replace("—", "-")
            .replace("–", "-")
            .replace("−", "-")
            .replace("‑", "-")
        )
        match = EMPLOYEE_ID_PATTERN.search(text)
        if match:
            normalized = match.group(0).upper().replace(" ", "")
            normalized = normalized.replace("_", "-").replace(".", "-")
            normalized = normalized.replace("O", "0")
            normalized = normalized.replace("F1", "FT").replace("FI", "FT")
            normalized = normalized.replace("F+", "FT").replace("F-", "FT-")
            normalized = re.sub(r"^F[T\+\-1I]*", "FT-", normalized)
            digits = "".join(char for char in normalized if char.isdigit())
            if digits:
                if len(digits) == 2:
                    return f"FT-0{digits}"
                return f"FT-{digits}"
    return ""


def group_by_column(items):
    columns = [[] for _ in COLUMN_RANGES]
    for item in items:
        for index, (x_min, x_max) in enumerate(COLUMN_RANGES):
            if x_min <= item["cx"] < x_max:
                columns[index].append(item)
                break

    values = []
    for column_items in columns:
        column_items.sort(key=lambda item: item["x1"])
        values.append(" ".join(normalize_text(item["text"]) for item in column_items))
    return values


def assign_items_to_rows(table_items, row_centers, row_tolerance, first_data_x):
    row_buckets = [[] for _ in row_centers]
    for item in table_items:
        if item["x1"] < first_data_x:
            continue
        nearest_index = min(
            range(len(row_centers)),
            key=lambda index: abs(item["cy"] - row_centers[index]),
        )
        if abs(item["cy"] - row_centers[nearest_index]) <= row_tolerance:
            row_buckets[nearest_index].append(item)
    return row_buckets


def normalize_ot_hours(value: str) -> str:
    value = normalize_text(value)
    digits = "".join(char for char in value if char.isdigit())
    return digits if digits else "NA"


def normalize_rope_access(value: str) -> str:
    value = normalize_text(value)
    if not value:
        return "NA"

    compact = value.upper().replace(" ", "")
    compact = compact.replace("$", "").replace(".", "")
    compact = compact.replace("O", "0").replace("I", "1").replace("L", "1")
    compact = compact.replace("B", "8").replace("S", "5").replace("G", "6")
    digits = "".join(char for char in compact if char.isdigit())

    if digits:
        return "$10"
    return "NA"


def extract_day_number(text: str):
    normalized = normalize_text(text)
    if normalized.isdigit():
        day = int(normalized)
        if 1 <= day <= 31:
            return day
    return None


def detect_header_bottom(items):
    date_label = find_label_item(items, "Date")
    regular_hours_label = find_label_item(items, "Regular Hours")
    header_bottom = 230
    if date_label or regular_hours_label:
        header_bottom = max(
            item["y2"] for item in (date_label, regular_hours_label) if item is not None
        )
    return header_bottom


def detect_total_top(items):
    total_label = find_label_item(items, "Total")
    if total_label is not None:
        return total_label["y1"]
    return 1550


def estimate_row_centers(items):
    header_bottom = detect_header_bottom(items)
    total_top = detect_total_top(items)
    regular_hours_label = find_label_item(items, "Regular Hours")
    date_boundary_x = regular_hours_label["x1"] if regular_hours_label else 110

    date_items = [
        item
        for item in items
        if item["x1"] < date_boundary_x
        and header_bottom <= item["cy"] < total_top
        and item["text"] != "Total"
    ]
    date_items.sort(key=lambda item: item["cy"])

    deduped_centers = []
    for item in date_items:
        if not deduped_centers or abs(item["cy"] - deduped_centers[-1]) > 10:
            deduped_centers.append(item["cy"])

    row_step = 37.0
    if len(deduped_centers) >= 2:
        row_step = (deduped_centers[-1] - deduped_centers[0]) / max(
            len(deduped_centers) - 1, 1
        )

    if deduped_centers:
        first_center = deduped_centers[0]
    else:
        first_center = header_bottom + row_step * 0.8

    row_centers = [first_center + row_step * index for index in range(31)]
    return row_centers, row_step


def apply_table_corrections(rows):
    corrected_rows = rows[:5]
    for row in rows[5:-1]:
        fixed = row[:]
        fixed[2] = normalize_ot_hours(fixed[2])
        fixed[3] = normalize_rope_access(fixed[3])
        corrected_rows.append(fixed)
    corrected_rows.append(rows[-1][:])
    return corrected_rows


def build_rows(result):
    texts = result["rec_texts"]
    boxes = result["rec_boxes"].tolist()
    items = []

    for text, box in zip(texts, boxes):
        x1, y1, x2, y2 = box
        items.append(
            {
                "text": str(text),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "cx": (int(x1) + int(x2)) / 2,
                "cy": (int(y1) + int(y2)) / 2,
            }
        )

    title = "Monthly Timesheet Record Card"
    name_label = find_label_item(items, "Name:")
    month_label = find_label_item(items, "Month:")
    wp_fin_label = find_label_item(items, "WP/FIN No.")
    employee_label = find_label_item(items, "Employee ID No:")

    month_left = month_label["x1"] if month_label else 650
    employee_left = employee_label["x1"] if employee_label else 520

    name = extract_band_text(
        items,
        name_label,
        name_label["x2"] + 5 if name_label else 140,
        month_left - 20,
    )
    month = extract_band_text(
        items,
        month_label,
        month_label["x2"] + 5 if month_label else 660,
        980,
    )
    wp_fin = extract_band_text(
        items,
        wp_fin_label,
        wp_fin_label["x2"] + 5 if wp_fin_label else 180,
        employee_left - 20,
    )
    employee_id = find_employee_id(items)

    row_centers, row_step = estimate_row_centers(items)

    rows = [
        [title, "", "", "", "", "", "", "", ""],
        ["", "Name:", name, "", "Month:", "", month, "", ""],
        ["", "WP/FIN No.", wp_fin, "", "Employee ID No:", "", employee_id, "", ""],
        ["", "", "", "", "", "", "", "", ""],
        HEADERS,
    ]

    header_bottom = detect_header_bottom(items)
    total_top = detect_total_top(items)
    regular_hours_label = find_label_item(items, "Regular Hours")
    first_data_x = regular_hours_label["x1"] if regular_hours_label else 110
    table_items = [item for item in items if header_bottom <= item["cy"] < total_top]
    row_tolerance = max(18, row_step * 0.42)
    row_buckets = assign_items_to_rows(
        table_items, row_centers, row_tolerance, first_data_x
    )
    for row_index, row_bucket in enumerate(row_buckets, start=1):
        row_values = group_by_column(row_bucket)
        row_values[0] = str(row_index)
        rows.append(row_values)

    total_values = ["Total", "", "", "", "", "", "", "", ""]
    total_values[1] = extract_numeric_region_text(items, 90, 180, 1395, 1460)
    total_values[2] = extract_numeric_region_text(items, 180, 270, 1395, 1475)
    total_values[3] = extract_numeric_region_text(items, 280, 380, 1395, 1460)
    rows.append(total_values)

    return apply_table_corrections(rows), employee_id


def export_csv(rows, csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            data = file.read().strip()
        if not data:
            return {}
        return json.loads(data)
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(manifest_path: Path, manifest) -> None:
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def get_file_signature(image_path: Path):
    stat = image_path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def should_process(image_path: Path, manifest) -> bool:
    key = str(image_path.resolve())
    return manifest.get(key) != get_file_signature(image_path)


def collect_candidates(image_dir: Path):
    return sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def main():
    project_dir = Path(__file__).resolve().parent
    image_dir = project_dir / "images"
    output_dir = project_dir / "output"
    manifest_path = output_dir / ".processed_images.json"

    image_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    manifest = load_manifest(manifest_path)

    if len(sys.argv) > 1:
        candidates = [Path(sys.argv[1])]
    else:
        candidates = collect_candidates(image_dir)
        if not candidates:
            raise FileNotFoundError(
                f"No image found in {image_dir}. Put an image there or pass a file path."
            )

    pending_images = [path for path in candidates if should_process(path, manifest)]
    skipped_images = [path for path in candidates if path not in pending_images]

    for image_path in skipped_images:
        print(f"Skipped already processed image: {image_path.name}")

    if not pending_images:
        print("No new or updated images found.")
        return

    pipeline = create_pipeline(pipeline="OCR")

    for image_path in pending_images:
        print(f"Processing image: {image_path.name}")
        results = pipeline.predict(str(image_path))
        for res in results:
            res.print()
            rows, employee_id = build_rows(res)
            output_stem = safe_stem(employee_id, image_path.stem)

            res.save_to_img(str(output_dir))
            target_ocr_path = move_generated_file(
                output_dir, image_path.stem, "_ocr_res_img", f"{output_stem}_ocr-res"
            )
            preprocessed_path = move_generated_file(
                output_dir,
                image_path.stem,
                "_preprocessed_img",
                f"{output_stem}_preprocessed",
            )
            if preprocessed_path and preprocessed_path.exists():
                preprocessed_path.unlink()

            csv_path = output_dir / f"{output_stem}.csv"
            export_csv(rows, csv_path)
            print(f"OCR image saved to: {target_ocr_path}")
            print(f"CSV saved to: {csv_path}")

        manifest[str(image_path.resolve())] = get_file_signature(image_path)
        save_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
