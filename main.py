import csv
import json
import os
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
    "405×": "405X",
    "&": "8",
    "之": "2",
    "一": "-",
    "DW": "DW",
}


def normalize_text(text: str) -> str:
    text = " ".join(str(text).strip().split())
    return VALUE_ALIASES.get(text, text)


def normalize_label(text: str) -> str:
    text = " ".join(str(text).strip().split()).upper()
    return LABEL_ALIASES.get(text, text)


def extract_region_text(items, x_min, x_max, y_min, y_max):
    candidates = [
        item
        for item in items
        if x_min <= item["cx"] <= x_max and y_min <= item["cy"] <= y_max
    ]
    candidates.sort(key=lambda item: (item["cy"], item["x1"]))
    return " ".join(normalize_text(item["text"]) for item in candidates)


def extract_region_items(items, x_min, x_max, y_min, y_max):
    candidates = [
        item
        for item in items
        if x_min <= item["cx"] <= x_max and y_min <= item["cy"] <= y_max
    ]
    candidates.sort(key=lambda item: (item["cy"], item["x1"]))
    return candidates


def extract_numeric_region_text(items, x_min, x_max, y_min, y_max):
    candidates = [
        item
        for item in extract_region_items(items, x_min, x_max, y_min, y_max)
        if any(char.isdigit() for char in str(item["text"]))
    ]
    candidates.sort(key=lambda item: item["x1"])
    return " ".join(normalize_text(item["text"]) for item in candidates)


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
    name = extract_region_text(items, 150, 420, 85, 150)
    month_items = extract_region_items(items, 650, 860, 100, 145)
    month_items.sort(key=lambda item: item["x1"])
    month = " ".join(normalize_text(item["text"]) for item in month_items)
    wp_fin = extract_region_text(items, 140, 340, 120, 190)
    employee_items = extract_region_items(items, 650, 860, 135, 180)
    employee_tokens = [
        normalize_text(item["text"])
        for item in employee_items
        if "-" in str(item["text"]) or str(item["text"]).upper().startswith("FT")
    ]
    employee_id = " ".join(employee_tokens)

    date_items = [
        item
        for item in items
        if item["cx"] < 60 and 285 <= item["cy"] <= 1550 and item["text"] != "Total"
    ]
    date_items.sort(key=lambda item: item["cy"])
    row_centers = [item["cy"] for item in date_items[:31]]

    rows = [
        [title, "", "", "", "", "", "", "", ""],
        ["", "Name:", name, "", "Month:", "", month, "", ""],
        ["", "WP/FIN No.", wp_fin, "", "Employee ID No:", "", employee_id, "", ""],
        ["", "", "", "", "", "", "", "", ""],
        HEADERS,
    ]

    table_items = [item for item in items if 285 <= item["cy"] <= 1550]
    for row_index, center in enumerate(row_centers, start=1):
        row_bucket = [
            item
            for item in table_items
            if abs(item["cy"] - center) <= 24 and item["cx"] >= 60
        ]
        row_values = group_by_column(row_bucket)
        row_values[0] = str(row_index)
        rows.append(row_values)

    total_values = ["Total", "", "", "", "", "", "", "", ""]
    total_values[1] = extract_numeric_region_text(items, 90, 180, 1395, 1460)
    total_values[2] = extract_numeric_region_text(items, 180, 270, 1395, 1475)
    total_values[3] = extract_numeric_region_text(items, 280, 380, 1395, 1460)
    rows.append(total_values)

    return rows, employee_id


def export_csv(rows, csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def main():
    project_dir = Path(__file__).resolve().parent
    image_dir = project_dir / "images"
    output_dir = project_dir / "output"
    manifest_path = output_dir / ".processed_images.json"
    image_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    manifest = load_manifest(manifest_path)

    supported = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
    if len(sys.argv) > 1:
        candidates = [Path(sys.argv[1])]
    else:
        candidates = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in supported
        )
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
                output_dir, image_path.stem, "_preprocessed_img", f"{output_stem}_preprocessed"
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
