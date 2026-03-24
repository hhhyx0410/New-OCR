import csv
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ocr_core import process_image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def default_image_directory() -> str:
    if getattr(sys, "frozen", False):
        return str(Path.home() / "Pictures")
    return str(Path(__file__).resolve().parent / "images")


class ClickableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preview_panel = None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.preview_panel is not None:
            self.preview_panel.open_zoom_dialog()
        super().mouseDoubleClickEvent(event)


class ZoomableImageLabel(QLabel):
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._original_pixmap = pixmap
        self._scale_factor = 1.0
        self.setAlignment(Qt.AlignCenter)
        self.update_pixmap()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self._scale_factor *= 1.15
        elif delta < 0:
            self._scale_factor /= 1.15
        self._scale_factor = max(0.2, min(self._scale_factor, 6.0))
        self.update_pixmap()
        event.accept()

    def update_pixmap(self):
        if self._original_pixmap.isNull():
            self.clear()
            return
        scaled = self._original_pixmap.scaled(
            self._original_pixmap.size() * self._scale_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.resize(scaled.size())


class ImageZoomDialog(QDialog):
    def __init__(self, title: str, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 900)

        image_label = ZoomableImageLabel(pixmap)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidget(image_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(scroll_area)

        self.setStyleSheet(
            """
            QDialog {
                background: #eef4ff;
            }
            QLabel {
                background: white;
                border-radius: 14px;
            }
            """
        )


class PreviewPanel(QFrame):
    def __init__(self, title: str, subtitle: str, minimum_height: int = 260):
        super().__init__()
        self.setObjectName("previewPanel")
        self.dialog = None

        self.title_label = QLabel(title)
        self.title_label.setObjectName("panelTitle")
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("panelSubtitle")
        self.subtitle_label.setWordWrap(True)

        self.image_label = ClickableImageLabel("Preview will appear here")
        self.image_label.preview_panel = self
        self.image_label.setObjectName("previewViewport")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(minimum_height)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._original_pixmap = QPixmap()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setWidget(self.image_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        layout.addWidget(self.scroll_area, 1)

    def set_placeholder(self, text: str):
        self._original_pixmap = QPixmap()
        self.image_label.clear()
        self.image_label.setText(text)

    def set_pixmap(self, pixmap: QPixmap):
        self._original_pixmap = pixmap
        if pixmap.isNull():
            self.set_placeholder("Preview unavailable")
            return
        viewport_size = self.scroll_area.viewport().size()
        scaled = pixmap.scaled(
            viewport_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def refresh_pixmap(self):
        if not self._original_pixmap.isNull():
            self.set_pixmap(self._original_pixmap)

    def open_zoom_dialog(self):
        if self._original_pixmap.isNull():
            return
        self.dialog = ImageZoomDialog(self.title_label.text(), self._original_pixmap, self)
        self.dialog.exec()


class DropUploadCard(QFrame):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setAcceptDrops(True)
        self.setObjectName("uploadCard")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in IMAGE_SUFFIXES:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    paths.append(path)
        if paths:
            self.parent_window.set_selected_images(paths)
            event.acceptProposedAction()
            return
        event.ignore()


class InfoBadge(QFrame):
    def __init__(self, label: str, value: str):
        super().__init__()
        self.setObjectName("infoBadge")

        self.value_label = QLabel(value)
        self.value_label.setObjectName("badgeValue")
        self.label_widget = QLabel(label)
        self.label_widget.setObjectName("badgeLabel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)
        layout.addWidget(self.value_label)
        layout.addWidget(self.label_widget)

    def set_value(self, value: str):
        self.value_label.setText(value)


class OcrWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Timesheet OCR Studio")
        self.resize(1680, 1000)

        self.current_image_paths = []
        self.current_result = None
        self.current_result_index = -1
        self.results = []
        self.temp_dir = Path(tempfile.mkdtemp(prefix="timesheet_ocr_"))
        self.error_log_path = self.temp_dir / "ocr_error.log"

        self.metric_image = InfoBadge("Selected Files", "0")
        self.metric_employee = InfoBadge("Employee ID", "Pending")
        self.metric_status = InfoBadge("Status", "Idle")

        self.open_button = QPushButton("Select Images")
        self.run_button = QPushButton("Run OCR")
        self.save_button = QPushButton("Save to Local")
        self.save_button.setEnabled(False)

        self.upload_title = QLabel("Click to upload or choose images to begin OCR")
        self.upload_title.setObjectName("uploadTitle")
        self.upload_hint = QLabel(
            "Supports PNG / JPG / JPEG / BMP / TIF / WEBP. You can upload multiple images and review each OCR result before saving."
        )
        self.upload_hint.setWordWrap(True)
        self.upload_hint.setObjectName("uploadHint")
        self.selected_name = QLabel("No images selected")
        self.selected_name.setObjectName("selectedName")
        self.selected_list = QListWidget()
        self.selected_list.setObjectName("resultList")
        self.selected_list.itemSelectionChanged.connect(self.on_selected_image_changed)

        self.upload_preview = PreviewPanel(
            "Selected Image",
            "Choose an item from the selected-image list to preview it before OCR starts.",
            minimum_height=260,
        )
        self.result_preview = PreviewPanel(
            "OCR Result Image",
            "Choose a processed item from the list to view the OCR result image.",
            minimum_height=300,
        )

        self.result_list = QListWidget()
        self.result_list.setObjectName("resultList")
        self.result_list.itemSelectionChanged.connect(self.on_result_selection_changed)

        self.csv_title = QLabel("CSV Preview")
        self.csv_title.setObjectName("sectionTitle")
        self.csv_hint = QLabel(
            "Preview the exported rows and key header fields for the selected OCR result."
        )
        self.csv_hint.setObjectName("panelSubtitle")

        self.meta_name = InfoBadge("Name", "Pending")
        self.meta_fin = InfoBadge("FIN", "Pending")
        self.meta_month = InfoBadge("Month", "Pending")
        self.meta_employee = InfoBadge("Employee ID", "Pending")

        self.csv_table = QTableWidget()
        self.csv_table.setColumnCount(9)
        self.csv_table.setHorizontalHeaderLabels(
            [
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
        )
        self.csv_table.setAlternatingRowColors(True)
        self.csv_table.verticalHeader().setVisible(False)

        self.build_ui()
        self.apply_styles()
        self.clear_preview_state()

    def build_ui(self):
        hero_panel = QFrame()
        hero_panel.setObjectName("heroPanel")

        hero_layout = QVBoxLayout(hero_panel)
        hero_layout.setContentsMargins(34, 30, 34, 30)
        hero_layout.setSpacing(18)

        hero_header = QLabel("Timesheet OCR Workspace")
        hero_header.setObjectName("heroHeader")
        hero_subtitle = QLabel(
            "Upload a batch of timesheet images, inspect each OCR rendering, and save the final results only when you are satisfied."
        )
        hero_subtitle.setWordWrap(True)
        hero_subtitle.setObjectName("heroSubtitle")

        badge_row = QHBoxLayout()
        badge_row.setSpacing(14)
        badge_row.addWidget(self.metric_image)
        badge_row.addWidget(self.metric_employee)
        badge_row.addWidget(self.metric_status)

        upload_card = DropUploadCard(self)
        upload_layout = QHBoxLayout(upload_card)
        upload_layout.setContentsMargins(26, 28, 26, 28)
        upload_layout.setSpacing(24)

        upload_left = QVBoxLayout()
        upload_left.setSpacing(12)

        upload_icon = QLabel("^")
        upload_icon.setAlignment(Qt.AlignCenter)
        upload_icon.setObjectName("uploadIcon")

        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        button_row.addWidget(self.open_button)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.save_button)

        upload_left.addWidget(upload_icon, alignment=Qt.AlignLeft)
        upload_left.addWidget(self.upload_title)
        upload_left.addWidget(self.upload_hint)
        upload_left.addWidget(self.selected_name)
        upload_left.addWidget(self.selected_list, 1)
        upload_left.addSpacing(6)
        upload_left.addLayout(button_row)
        upload_left.addStretch()

        upload_layout.addLayout(upload_left, 2)
        upload_layout.addWidget(self.upload_preview, 3)

        hero_layout.addWidget(hero_header)
        hero_layout.addWidget(hero_subtitle)
        hero_layout.addLayout(badge_row)
        hero_layout.addWidget(upload_card)

        result_panel = QFrame()
        result_panel.setObjectName("tablePanel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(20, 20, 20, 20)
        result_layout.setSpacing(10)

        result_title = QLabel("OCR Results")
        result_title.setObjectName("sectionTitle")
        result_hint = QLabel(
            "After OCR finishes, select an item from the list to load its OCR image and CSV preview."
        )
        result_hint.setObjectName("panelSubtitle")

        result_layout.addWidget(result_title)
        result_layout.addWidget(result_hint)
        result_layout.addWidget(self.result_list)
        result_layout.addWidget(self.result_preview, 1)

        table_panel = QFrame()
        table_panel.setObjectName("tablePanel")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(20, 20, 20, 20)
        table_layout.setSpacing(10)

        meta_grid = QGridLayout()
        meta_grid.setHorizontalSpacing(10)
        meta_grid.setVerticalSpacing(10)
        meta_grid.addWidget(self.meta_name, 0, 0)
        meta_grid.addWidget(self.meta_fin, 0, 1)
        meta_grid.addWidget(self.meta_month, 1, 0)
        meta_grid.addWidget(self.meta_employee, 1, 1)
        meta_grid.setColumnStretch(0, 1)
        meta_grid.setColumnStretch(1, 1)

        table_layout.addWidget(self.csv_title)
        table_layout.addWidget(self.csv_hint)
        table_layout.addLayout(meta_grid)
        table_layout.addWidget(self.csv_table, 1)

        lower_grid = QGridLayout()
        lower_grid.setHorizontalSpacing(18)
        lower_grid.setVerticalSpacing(18)
        lower_grid.addWidget(result_panel, 0, 0)
        lower_grid.addWidget(table_panel, 0, 1)
        lower_grid.setColumnStretch(0, 1)
        lower_grid.setColumnStretch(1, 1)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)
        layout.addWidget(hero_panel)
        layout.addLayout(lower_grid, 1)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidget(content)
        self.setCentralWidget(scroll_area)

        self.setStatusBar(QStatusBar())
        self.open_button.clicked.connect(self.choose_images)
        self.run_button.clicked.connect(self.run_ocr)
        self.save_button.clicked.connect(self.save_outputs)

    def set_selected_images(self, paths):
        deduped = []
        seen = set()
        for path in paths:
            resolved = str(Path(path))
            if resolved not in seen:
                seen.add(resolved)
                deduped.append(Path(path))

        self.current_image_paths = deduped
        self.results = []
        self.current_result = None
        self.current_result_index = -1
        self.result_list.clear()
        self.selected_list.clear()
        self.save_button.setEnabled(False)
        self.clear_preview_state()

        count = len(self.current_image_paths)
        self.metric_image.set_value(str(count))
        self.metric_employee.set_value("Pending")
        self.metric_status.set_value("Ready" if count else "Idle")

        if count:
            self.selected_name.setText(
                f"{count} image(s) selected."
            )
            for index, image_path in enumerate(self.current_image_paths, start=1):
                item = QListWidgetItem(f"{index}. {image_path.name}")
                item.setData(Qt.UserRole, index - 1)
                self.selected_list.addItem(item)
            self.selected_list.setCurrentRow(0)
            self.statusBar().showMessage(f"Selected {count} image(s).")
        else:
            self.selected_name.setText("No images selected")
            self.upload_preview.set_placeholder("Preview will appear here")
            self.statusBar().showMessage("No images selected.")

    def clear_preview_state(self):
        self.result_preview.set_placeholder("Select a processed item to preview its OCR image")
        self.meta_name.set_value("Pending")
        self.meta_fin.set_value("Pending")
        self.meta_month.set_value("Pending")
        self.meta_employee.set_value("Pending")
        self.csv_table.setRowCount(0)

    def on_selected_image_changed(self):
        item = self.selected_list.currentItem()
        if item is None:
            return
        index = item.data(Qt.UserRole)
        if index is None or index < 0 or index >= len(self.current_image_paths):
            return
        self.upload_preview.set_pixmap(QPixmap(str(self.current_image_paths[index])))

    def apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #edf5ff,
                    stop: 1 #f8fbff
                );
            }
            #heroPanel, #previewPanel, #tablePanel {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid #d7e3f2;
                border-radius: 22px;
            }
            #infoBadge {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(241, 246, 255, 0.95),
                    stop: 1 rgba(255, 255, 255, 0.98)
                );
                border: 1px solid #d7e3f2;
                border-radius: 18px;
            }
            #uploadCard {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(230, 239, 255, 0.75),
                    stop: 1 rgba(255, 255, 255, 0.95)
                );
                border: 2px dashed #b8cae6;
                border-radius: 22px;
            }
            #heroHeader {
                font-size: 30px;
                font-weight: 700;
                color: #1b2d52;
            }
            #heroSubtitle, #panelSubtitle, #uploadHint {
                font-size: 13px;
                color: #7185a6;
            }
            #uploadTitle {
                font-size: 26px;
                font-weight: 700;
                color: #1d2850;
            }
            #selectedName {
                font-size: 14px;
                font-weight: 600;
                color: #4f5f82;
            }
            #uploadIcon {
                min-width: 78px;
                min-height: 78px;
                max-width: 78px;
                max-height: 78px;
                border-radius: 18px;
                background: #4c6fff;
                color: white;
                font-size: 30px;
                font-weight: 700;
            }
            #panelTitle, #sectionTitle {
                font-size: 18px;
                font-weight: 700;
                color: #203050;
            }
            #previewViewport {
                background: #f7faff;
                border: 1px dashed #c0d2eb;
                border-radius: 16px;
                color: #8596b2;
                font-size: 14px;
                padding: 12px;
            }
            #badgeValue {
                font-size: 17px;
                font-weight: 700;
                color: #243a61;
            }
            #badgeLabel {
                font-size: 12px;
                color: #7387aa;
            }
            QPushButton {
                min-height: 44px;
                padding: 0 18px;
                border-radius: 14px;
                border: none;
                background: #4c6fff;
                color: white;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #3f60e6;
            }
            QPushButton:disabled {
                background: #a8b7d6;
                color: #eef3fb;
            }
            QListWidget {
                background: rgba(247, 250, 255, 0.88);
                border: 1px solid #dce6f3;
                border-radius: 14px;
                padding: 8px;
                color: #22365c;
            }
            QListWidget::item {
                border-radius: 10px;
                padding: 10px 12px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: #4c6fff;
                color: white;
            }
            QTableWidget {
                background: rgba(247, 250, 255, 0.88);
                border: 1px solid #dce6f3;
                border-radius: 14px;
                gridline-color: #dce6f3;
                color: #22365c;
                font-size: 13px;
                alternate-background-color: #f2f7ff;
            }
            QHeaderView::section {
                background: #edf4ff;
                color: #35507a;
                border: none;
                border-bottom: 1px solid #dce6f3;
                padding: 9px;
                font-weight: 700;
            }
            """
        )

    def choose_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            default_image_directory(),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not paths:
            return
        self.set_selected_images([Path(path) for path in paths])

    def run_ocr(self):
        if not self.current_image_paths:
            QMessageBox.warning(self, "Warning", "Please select images first.")
            return

        self.results = []
        self.result_list.clear()
        self.current_result = None
        self.current_result_index = -1
        self.clear_preview_state()

        try:
            total = len(self.current_image_paths)
            self.metric_status.set_value("Running")
            for index, image_path in enumerate(self.current_image_paths, start=1):
                self.statusBar().showMessage(f"Processing {index}/{total}: {image_path.name}")
                QApplication.processEvents()
                result = process_image(image_path, self.temp_dir)
                result["source_image_path"] = str(image_path)
                self.results.append(result)

                employee_id = result.get("employee_id") or "Unknown"
                item = QListWidgetItem(f"{index}. {employee_id}  |  {image_path.name}")
                item.setData(Qt.UserRole, len(self.results) - 1)
                self.result_list.addItem(item)

            self.save_button.setEnabled(bool(self.results))
            self.metric_status.set_value("Completed")
            self.metric_employee.set_value(
                self.results[0].get("employee_id") if self.results else "Pending"
            )

            if self.results:
                self.result_list.setCurrentRow(0)
            self.statusBar().showMessage(f"OCR completed for {len(self.results)} image(s).")
        except Exception as exc:
            self.metric_status.set_value("Failed")
            details = traceback.format_exc()
            self.error_log_path.write_text(details, encoding="utf-8")
            QMessageBox.critical(
                self,
                "OCR Failed",
                f"{exc}\n\nFull error log:\n{self.error_log_path}",
            )
            self.statusBar().showMessage("OCR failed.")

    def on_result_selection_changed(self):
        item = self.result_list.currentItem()
        if item is None:
            return
        index = item.data(Qt.UserRole)
        if index is None or index < 0 or index >= len(self.results):
            return
        self.load_result(index)

    def load_result(self, index: int):
        self.current_result_index = index
        self.current_result = self.results[index]
        self.metric_employee.set_value(self.current_result.get("employee_id") or "Unknown")

        image_path = self.current_result.get("ocr_image_path", "")
        if image_path:
            self.result_preview.set_pixmap(QPixmap(image_path))
        else:
            self.result_preview.set_placeholder("Preview unavailable")

        csv_path = Path(self.current_result["csv_path"])
        rows = list(csv.reader(csv_path.open("r", encoding="utf-8-sig", newline="")))

        self.meta_name.set_value(self.extract_value(rows, 1, 2))
        self.meta_fin.set_value(self.extract_value(rows, 2, 2))
        self.meta_month.set_value(self.extract_value(rows, 1, 6))
        self.meta_employee.set_value(self.extract_value(rows, 2, 6))

        data_rows = rows[5:]
        self.csv_table.clearContents()
        self.csv_table.setRowCount(len(data_rows))
        for row_index, row in enumerate(data_rows):
            for column_index in range(9):
                value = row[column_index] if column_index < len(row) else ""
                self.csv_table.setItem(row_index, column_index, QTableWidgetItem(value))
        self.csv_table.resizeColumnsToContents()

    @staticmethod
    def extract_value(rows, row_index: int, column_index: int) -> str:
        if row_index < len(rows) and column_index < len(rows[row_index]):
            value = rows[row_index][column_index].strip()
            return value or "NA"
        return "NA"

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image_paths:
            self.upload_preview.refresh_pixmap()
        if self.current_result and self.current_result.get("ocr_image_path"):
            self.result_preview.refresh_pixmap()

    def save_outputs(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "There are no results to save.")
            return

        target_dir = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if not target_dir:
            return

        target_dir = Path(target_dir)
        copied_diagnostics = set()
        for result in self.results:
            ocr_image_path = Path(result["ocr_image_path"])
            csv_path = Path(result["csv_path"])
            shutil.copy2(ocr_image_path, target_dir / ocr_image_path.name)
            shutil.copy2(csv_path, target_dir / csv_path.name)
            diagnostic_path = result.get("diagnostic_path")
            if diagnostic_path:
                diagnostic_path = Path(diagnostic_path)
                if diagnostic_path.exists() and diagnostic_path.name not in copied_diagnostics:
                    shutil.copy2(diagnostic_path, target_dir / diagnostic_path.name)
                    copied_diagnostics.add(diagnostic_path.name)

        QMessageBox.information(
            self,
            "Saved",
            f"Saved {len(self.results)} OCR image(s) and CSV file(s) to:\n{target_dir}",
        )
        self.statusBar().showMessage("Results saved to local folder.")


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = OcrWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
