import csv
import importlib
import importlib.util
import json
import os
import platform
import re
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_DISABLE_DEVICE_FALLBACK", "True")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "True")


class _NullWriter:
    def write(self, _text):
        return 0

    def flush(self):
        return None


if getattr(sys, "frozen", False):
    if sys.stdout is None:
        sys.stdout = _NullWriter()
    if sys.stderr is None:
        sys.stderr = _NullWriter()

import cv2  # noqa: F401
import pyclipper  # noqa: F401
from paddle.base.libpaddle import AnalysisConfig
from paddlex import create_pipeline
from paddlex.inference import PaddlePredictorOption
from paddlex.utils import deps as paddlex_deps


if not hasattr(AnalysisConfig, "set_optimization_level"):
    AnalysisConfig.set_optimization_level = lambda self, level: None


if getattr(sys, "frozen", False):
    _original_is_dep_available = paddlex_deps.is_dep_available
    _original_is_extra_available = paddlex_deps.is_extra_available

    def _module_exists(module_name: str) -> bool:
        if module_name in sys.modules:
            return True
        try:
            importlib.import_module(module_name)
            return True
        except Exception:
            return importlib.util.find_spec(module_name) is not None

    def _frozen_is_dep_available(dep, /, check_version=False):
        module_name_map = {
            "opencv-contrib-python": "cv2",
            "pyclipper": "pyclipper",
            "imagesize": "imagesize",
            "pypdfium2": "pypdfium2",
            "python-bidi": "bidi",
            "shapely": "shapely",
        }
        if dep in module_name_map and not check_version:
            return _module_exists(module_name_map[dep])
        return _original_is_dep_available(dep, check_version=check_version)

    def _frozen_is_extra_available(extra):
        if extra not in {"ocr", "ocr-core"}:
            return _original_is_extra_available(extra)
        required_specs = (
            "cv2",
            "imagesize",
            "pyclipper",
            "pypdfium2",
            "bidi",
            "shapely",
        )
        return all(_module_exists(dep) for dep in required_specs)

    paddlex_deps.is_dep_available = _frozen_is_dep_available
    paddlex_deps.is_extra_available = _frozen_is_extra_available

    def _patch_paddlex_cv2_modules():
        module_names = (
            "paddlex.inference.common.reader.image_reader",
            "paddlex.inference.utils.io.readers",
            "paddlex.inference.utils.io.writers",
            "paddlex.inference.models.common.vision.funcs",
            "paddlex.inference.models.common.vision.processors",
            "paddlex.inference.models.text_detection.processors",
            "paddlex.inference.models.text_detection.result",
            "paddlex.inference.models.text_recognition.processors",
            "paddlex.inference.pipelines.components.common.crop_image_regions",
            "paddlex.inference.pipelines.components.common.seal_det_warp",
            "paddlex.inference.pipelines.components.common.warp_image",
            "paddlex.inference.pipelines.ocr.result",
        )
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            if getattr(module, "cv2", None) is None:
                module.cv2 = cv2

    def _patch_paddlex_dependency_modules():
        _patch_paddlex_cv2_modules()

        module_bindings = {
            "paddlex.inference.models.text_detection.processors": {
                "pyclipper": pyclipper,
            },
        }
        for module_name, bindings in module_bindings.items():
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for attr_name, attr_value in bindings.items():
                if getattr(module, attr_name, None) is None:
                    setattr(module, attr_name, attr_value)

    _patch_paddlex_dependency_modules()


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

DEFAULT_COLUMN_RANGES = [
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

HEADER_KEYWORDS = [
    ("Date", ("DATE",)),
    ("Regular Hours", ("REGULAR", "HOURS")),
    ("OT Hours / Allowance", ("OT", "HOUR")),
    ("Rope Access Allowance", ("ROPE", "ACCESS")),
    ("Transport Allowance", ("TRANSPORT",)),
    ("Night Shift Allowance", ("NIGHT", "SHIFT")),
    ("Food Allowance", ("FOOD",)),
    ("Job Site", ("JOB", "SITE")),
    ("Supervisor Name & Signature", ("SUPERVISOR", "SIGNATURE")),
]

EMPLOYEE_ID_PATTERN = re.compile(r"F[\sT1I\+\-_\.]*O?\d{1,4}", re.IGNORECASE)
SUPPORTED_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
_PIPELINE = None
APP_BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
BUNDLED_MODEL_ROOT = APP_BASE_DIR / "official_models"
USER_MODEL_ROOT = Path(
    os.environ.get("PADDLE_PDX_CACHE_HOME", str(Path.home() / ".paddlex"))
) / "official_models"
CPU_THREADS = max(1, min(int(os.environ.get("TIMESHEET_OCR_CPU_THREADS", os.cpu_count() or 1)), 4))
ENABLE_MKLDNN = os.environ.get("TIMESHEET_OCR_ENABLE_MKLDNN", "1") == "1"
OCR_MODEL_NAMES = (
    "PP-LCNet_x1_0_doc_ori",
    "UVDoc",
    "PP-OCRv5_server_det",
    "PP-LCNet_x1_0_textline_ori",
    "PP-OCRv5_server_rec",
)
USE_DOC_PREPROCESSOR = os.environ.get("TIMESHEET_OCR_USE_DOC_PREPROCESSOR", "0") == "1"
USE_DOC_UNWARPING = os.environ.get("TIMESHEET_OCR_USE_DOC_UNWARPING", "0") == "1"
USE_TEXTLINE_ORIENTATION = os.environ.get("TIMESHEET_OCR_USE_TEXTLINE_ORIENTATION", "0") == "1"
WRITE_DIAGNOSTICS_ON_SUCCESS = os.environ.get("TIMESHEET_OCR_WRITE_DIAGNOSTICS", "0") == "1"
PRINT_OCR_RESULT = os.environ.get("TIMESHEET_OCR_PRINT_RESULT", "0") == "1"

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)


def _model_paths(model_root: Path):
    return {model_name: model_root / model_name for model_name in OCR_MODEL_NAMES}


def _build_local_ocr_config(model_root: Path):
    model_paths = _model_paths(model_root)
    config = {
        "pipeline_name": "OCR",
        "text_type": "general",
        "use_doc_preprocessor": USE_DOC_PREPROCESSOR,
        "use_textline_orientation": USE_TEXTLINE_ORIENTATION,
        "SubModules": {
            "TextDetection": {
                "module_name": "text_detection",
                "model_name": "PP-OCRv5_server_det",
                "model_dir": str(model_paths["PP-OCRv5_server_det"]),
                "limit_side_len": 64,
                "limit_type": "min",
                "max_side_limit": 4000,
                "thresh": 0.3,
                "box_thresh": 0.6,
                "unclip_ratio": 1.5,
            },
            "TextLineOrientation": {
                "module_name": "textline_orientation",
                "model_name": "PP-LCNet_x1_0_textline_ori",
                "model_dir": str(model_paths["PP-LCNet_x1_0_textline_ori"]),
                "batch_size": 6,
            },
            "TextRecognition": {
                "module_name": "text_recognition",
                "model_name": "PP-OCRv5_server_rec",
                "model_dir": str(model_paths["PP-OCRv5_server_rec"]),
                "batch_size": 6,
                "score_thresh": 0.0,
            },
        },
    }
    if USE_DOC_PREPROCESSOR:
        doc_preprocessor = {
            "pipeline_name": "doc_preprocessor",
            "use_doc_orientation_classify": True,
            "use_doc_unwarping": USE_DOC_UNWARPING,
            "SubModules": {
                "DocOrientationClassify": {
                    "module_name": "doc_text_orientation",
                    "model_name": "PP-LCNet_x1_0_doc_ori",
                    "model_dir": str(model_paths["PP-LCNet_x1_0_doc_ori"]),
                },
            },
        }
        if USE_DOC_UNWARPING:
            doc_preprocessor["SubModules"]["DocUnwarping"] = {
                "module_name": "image_unwarping",
                "model_name": "UVDoc",
                "model_dir": str(model_paths["UVDoc"]),
            }
        config["SubPipelines"] = {"DocPreprocessor": doc_preprocessor}
    return config


def _missing_local_models(model_root: Path):
    model_paths = _model_paths(model_root)
    return [
        f"{name}: {path}"
        for name, path in model_paths.items()
        if not path.exists()
    ]


def _build_predictor_option():
    option = PaddlePredictorOption(
        run_mode="paddle",
        device_type="cpu",
        cpu_threads=CPU_THREADS,
        enable_new_ir=False,
        enable_cinn=False,
        delete_pass=[],
    )
    try:
        option.mkldnn_cache_capacity = 10 if ENABLE_MKLDNN else 0
    except Exception:
        pass
    try:
        option.enable_mkldnn = ENABLE_MKLDNN
    except Exception:
        pass
    return option


def _resolve_model_root():
    candidate_roots = []
    if BUNDLED_MODEL_ROOT not in candidate_roots:
        candidate_roots.append(BUNDLED_MODEL_ROOT)
    if USER_MODEL_ROOT not in candidate_roots:
        candidate_roots.append(USER_MODEL_ROOT)

    for model_root in candidate_roots:
        missing_models = _missing_local_models(model_root)
        if not missing_models:
            source = "bundled" if model_root == BUNDLED_MODEL_ROOT else "user-cache"
            return model_root, source, candidate_roots

    details = "\n".join(
        f"{model_root} -> missing {', '.join(item.split(':', 1)[0] for item in _missing_local_models(model_root))}"
        for model_root in candidate_roots
    )
    raise RuntimeError(
        "OCR models are unavailable. Neither the packaged model set nor the local PaddleX cache contains a complete OCR model bundle.\n"
        f"Checked model roots:\n{details}"
    )


def _write_diagnostics(output_dir: Path, model_root: Path, model_source: str, candidate_roots):
    diagnostics = {
        "frozen": bool(getattr(sys, "frozen", False)),
        "app_base_dir": str(APP_BASE_DIR),
        "bundled_model_root": str(BUNDLED_MODEL_ROOT),
        "active_model_root": str(model_root),
        "active_model_source": model_source,
        "user_model_root": str(USER_MODEL_ROOT),
        "candidate_model_roots": [str(path) for path in candidate_roots],
        "cpu_threads": CPU_THREADS,
        "device": "cpu",
        "run_mode": "paddle",
        "mkldnn_enabled": ENABLE_MKLDNN,
        "enable_new_ir": False,
        "python_version": sys.version,
        "platform": platform.platform(),
        "model_paths": {
            model_name: str(model_root / model_name) for model_name in OCR_MODEL_NAMES
        },
        "missing_bundled_models": _missing_local_models(BUNDLED_MODEL_ROOT),
        "missing_user_models": _missing_local_models(USER_MODEL_ROOT),
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "TIMESHEET_OCR_USE_DOC_PREPROCESSOR": os.environ.get(
                "TIMESHEET_OCR_USE_DOC_PREPROCESSOR"
            ),
            "TIMESHEET_OCR_USE_DOC_UNWARPING": os.environ.get(
                "TIMESHEET_OCR_USE_DOC_UNWARPING"
            ),
            "TIMESHEET_OCR_USE_TEXTLINE_ORIENTATION": os.environ.get(
                "TIMESHEET_OCR_USE_TEXTLINE_ORIENTATION"
            ),
            "TIMESHEET_OCR_CPU_THREADS": os.environ.get(
                "TIMESHEET_OCR_CPU_THREADS"
            ),
            "TIMESHEET_OCR_ENABLE_MKLDNN": os.environ.get(
                "TIMESHEET_OCR_ENABLE_MKLDNN"
            ),
            "TIMESHEET_OCR_WRITE_DIAGNOSTICS": os.environ.get(
                "TIMESHEET_OCR_WRITE_DIAGNOSTICS"
            ),
            "TIMESHEET_OCR_PRINT_RESULT": os.environ.get(
                "TIMESHEET_OCR_PRINT_RESULT"
            ),
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": os.environ.get(
                "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"
            ),
            "PADDLE_PDX_DISABLE_DEVICE_FALLBACK": os.environ.get(
                "PADDLE_PDX_DISABLE_DEVICE_FALLBACK"
            ),
            "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT": os.environ.get(
                "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = output_dir / "ocr_runtime_diagnostics.json"
    diagnostic_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return diagnostic_path


def get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        model_root, _, _ = _resolve_model_root()
        _PIPELINE = create_pipeline(
            config=_build_local_ocr_config(model_root),
            device="cpu",
            pp_option=_build_predictor_option(),
        )
    return _PIPELINE


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


def find_keyword_items(items, keywords, y_min=None, y_max=None):
    matches = []
    for item in items:
        if y_min is not None and item["cy"] < y_min:
            continue
        if y_max is not None and item["cy"] > y_max:
            continue
        normalized = normalize_label(item["text"])
        if all(keyword in normalized for keyword in keywords):
            matches.append(item)
    matches.sort(key=lambda item: (item["y1"], item["x1"]))
    return matches


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
    employee_label = find_label_item(items, "Employee ID No:")
    scoped_items = items
    if employee_label is not None:
        scoped_items = [
            item
            for item in items
            if item["x1"] >= employee_label["x2"] - 5
            and item["x2"] <= employee_label["x2"] + 260
            and abs(item["cy"] - employee_label["cy"]) <= 45
        ] or items

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


def group_by_column(items, column_ranges):
    columns = [[] for _ in column_ranges]
    for item in items:
        for index, (x_min, x_max) in enumerate(column_ranges):
            if x_min <= item["cx"] < x_max:
                columns[index].append(item)
                break

    values = []
    for column_items in columns:
        column_items.sort(key=lambda item: item["x1"])
        values.append(" ".join(normalize_text(item["text"]) for item in column_items))
    return values


def build_column_ranges(items, header_bottom, total_top):
    header_region_top = max(0, header_bottom - 120)
    header_region_bottom = min(total_top, header_bottom + 40)
    anchors = []

    for index, (_, keywords) in enumerate(HEADER_KEYWORDS):
        matches = find_keyword_items(
            items,
            keywords,
            y_min=header_region_top,
            y_max=header_region_bottom,
        )
        if not matches:
            continue
        left = min(item["x1"] for item in matches)
        right = max(item["x2"] for item in matches)
        anchors.append(
            {
                "index": index,
                "left": left,
                "right": right,
                "center": (left + right) / 2,
            }
        )

    # We only derive custom column boundaries when all headers are found;
    # partial OCR matches can otherwise produce too few boundaries and crash.
    if len(anchors) < len(HEADERS):
        return DEFAULT_COLUMN_RANGES

    anchors.sort(key=lambda anchor: anchor["center"])
    boundaries = [0]
    for left_anchor, right_anchor in zip(anchors, anchors[1:]):
        midpoint = int((left_anchor["center"] + right_anchor["center"]) / 2)
        boundaries.append(midpoint)
    boundaries.append(9999)

    if len(boundaries) != len(HEADERS) + 1:
        return DEFAULT_COLUMN_RANGES

    column_ranges = []
    for index in range(len(HEADERS)):
        column_ranges.append((boundaries[index], boundaries[index + 1]))
    return column_ranges


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


def normalize_regular_hours(value: str, row_index: int | None = None) -> str:
    value = normalize_text(value)
    tokens = value.split()
    if len(tokens) < 2:
        return value

    prefix = tokens[0].strip(".,;:|")
    suffix = " ".join(tokens[1:]).strip()
    if not suffix:
        return value

    digits = "".join(char for char in prefix if char.isdigit())
    if digits:
        try:
            day = int(digits)
        except ValueError:
            day = None
        if day is not None and 1 <= day <= 31 and (row_index is None or day == row_index):
            return suffix

    if row_index is not None and row_index < 10 and prefix.upper() in {"L", "I", "|"}:
        return suffix

    return value


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

    first_center = deduped_centers[0] if deduped_centers else header_bottom + row_step * 0.8
    row_centers = [first_center + row_step * index for index in range(31)]
    return row_centers, row_step


def apply_table_corrections(rows):
    corrected_rows = rows[:5]
    for row_offset, row in enumerate(rows[5:-1], start=1):
        fixed = row[:]
        fixed[1] = normalize_regular_hours(fixed[1], row_offset)
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
        items, name_label, name_label["x2"] + 5 if name_label else 140, month_left - 20
    )
    month = extract_band_text(
        items, month_label, month_label["x2"] + 5 if month_label else 660, 980
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
    column_ranges = build_column_ranges(items, header_bottom, total_top)
    first_data_x = column_ranges[1][0] + 25
    table_items = [item for item in items if header_bottom <= item["cy"] < total_top]
    row_tolerance = max(18, row_step * 0.42)
    row_buckets = assign_items_to_rows(
        table_items, row_centers, row_tolerance, first_data_x
    )
    for row_index, row_bucket in enumerate(row_buckets, start=1):
        row_values = group_by_column(row_bucket, column_ranges)
        row_values[0] = str(row_index)
        rows.append(row_values)

    total_values = ["Total", "", "", "", "", "", "", "", ""]
    total_label = find_label_item(items, "Total")
    total_y_min = total_label["y1"] - 25 if total_label is not None else total_top - 20
    total_y_max = total_label["y2"] + 35 if total_label is not None else total_top + 60
    total_values[1] = extract_numeric_region_text(
        items, column_ranges[1][0], column_ranges[1][1], total_y_min, total_y_max
    )
    total_values[2] = extract_numeric_region_text(
        items, column_ranges[2][0], column_ranges[2][1], total_y_min, total_y_max
    )
    total_values[3] = extract_numeric_region_text(
        items, column_ranges[3][0], column_ranges[3][1], total_y_min, total_y_max
    )
    rows.append(total_values)
    return apply_table_corrections(rows), employee_id


def export_csv(rows, csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        csv.writer(file).writerows(rows)


def rows_to_dicts(rows):
    return [dict(zip(HEADERS, row)) for row in rows[5:-1]]


def process_image(image_path, output_dir):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_root, model_source, candidate_roots = _resolve_model_root()
    diagnostic_path = None

    try:
        if WRITE_DIAGNOSTICS_ON_SUCCESS:
            diagnostic_path = _write_diagnostics(
                output_dir,
                model_root,
                model_source,
                candidate_roots,
            )

        results = get_pipeline().predict(str(image_path))
        result = next(iter(results))
        if PRINT_OCR_RESULT:
            result.print()
        rows, employee_id = build_rows(result)
        output_stem = safe_stem(employee_id, image_path.stem)

        result.save_to_img(str(output_dir))
        ocr_image_path = move_generated_file(
            output_dir, image_path.stem, "_ocr_res_img", f"{output_stem}_ocr-res"
        )
        preprocessed_path = move_generated_file(
            output_dir, image_path.stem, "_preprocessed_img", f"{output_stem}_preprocessed"
        )
        if preprocessed_path and preprocessed_path.exists():
            preprocessed_path.unlink()

        csv_path = output_dir / f"{output_stem}.csv"
        export_csv(rows, csv_path)
        return {
            "employee_id": employee_id,
            "output_stem": output_stem,
            "ocr_image_path": str(ocr_image_path) if ocr_image_path else "",
            "csv_path": str(csv_path),
            "diagnostic_path": str(diagnostic_path) if diagnostic_path else "",
            "rows": rows,
            "table_rows": rows_to_dicts(rows),
        }
    except Exception:
        if diagnostic_path is None:
            _write_diagnostics(
                output_dir,
                model_root,
                model_source,
                candidate_roots,
            )
        raise


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            data = file.read().strip()
        return json.loads(data) if data else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(manifest_path: Path, manifest) -> None:
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def get_file_signature(image_path: Path):
    stat = image_path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def should_process(image_path: Path, manifest) -> bool:
    return manifest.get(str(image_path.resolve())) != get_file_signature(image_path)


def collect_candidates(image_dir: Path):
    return sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def process_pending_images(project_dir, explicit_paths=None):
    project_dir = Path(project_dir)
    image_dir = project_dir / "images"
    output_dir = project_dir / "output"
    manifest_path = output_dir / ".processed_images.json"
    image_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    manifest = load_manifest(manifest_path)

    candidates = [Path(path) for path in explicit_paths] if explicit_paths else collect_candidates(image_dir)
    if not candidates:
        raise FileNotFoundError(f"No image found in {image_dir}. Put an image there or pass a file path.")

    pending_images = [path for path in candidates if should_process(path, manifest)]
    skipped_images = [path for path in candidates if path not in pending_images]
    outputs = []

    for image_path in skipped_images:
        print(f"Skipped already processed image: {image_path.name}")

    if not pending_images:
        print("No new or updated images found.")
        return outputs

    for image_path in pending_images:
        print(f"Processing image: {image_path.name}")
        outputs.append(process_image(image_path, output_dir))
        manifest[str(image_path.resolve())] = get_file_signature(image_path)
        save_manifest(manifest_path, manifest)

    return outputs
