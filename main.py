import sys
from pathlib import Path

from ocr_core import process_pending_images


def main():
    project_dir = Path(__file__).resolve().parent
    explicit_paths = sys.argv[1:] if len(sys.argv) > 1 else None
    process_pending_images(project_dir, explicit_paths)


if __name__ == "__main__":
    main()
