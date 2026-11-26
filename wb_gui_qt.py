"""PyQt6 desktop application for Western blot band quantification.

The analysis itself (detection, backgrounds, profiles, lane normalisation)
lives in ``wb_analysis``; this module is the interface around it: image
loading and display, ROI editing, pairing, exports, logging, and per-image
sidecar files that restore ROIs when an image is reopened.

This is the Qt port of ``wb_gui.py`` (Tk). It uses the same ``wb_analysis``
module unchanged, writes the same CSV/ROI-file formats and keeps the same
log, so either interface can open the other's ROI files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from datetime import datetime
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import tifffile

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QImage, QKeySequence, QPainter, QPen, QPixmap,
    QShortcut, QTransform,
)
from PyQt6.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGraphicsItem, QGraphicsScene,
    QGraphicsView, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QSlider,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

# matplotlib is optional: without it the app still runs and still exports the
# profile CSV, but the live profile plots and the exported PNG are skipped.
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    HAVE_MATPLOTLIB = True
except Exception:  # pragma: no cover - depends on the user's environment
    HAVE_MATPLOTLIB = False

from wb_analysis import (
    EDGE_KEYS,
    BandMeasurement,
    BandROI,
    LaneNormalizationResult,
    align_rois_to_region_signal,
    assign_lanes,
    clamp_roi,
    detect_bands,
    detect_bands_in_region,
    expand_band_rois,
    filter_bands_below_mean_signal,
    filter_marker_lanes,
    infer_polarity,
    intensity_profiles,
    measure_all,
    normalize_target_to_reference_by_lane,
    profile_baseline,
    to_gray_array,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "Western Blot Band Analysis"

# What the two images are called in the interface. Internally (and in CSV
# column names) they remain "reference" and "target".
IMAGE_ROLE_NAMES = {"reference": "Control", "target": "Protein 1"}
DEFAULT_CATEGORY = {"reference": "GAPDH", "target": "Protein 1"}

DISPLAY_MODES = {"Linear": "linear", "Log": "log"}
LOG_STRETCH = 200.0              # log display spans ~2.3 decades between black and white
AUTO_PERCENTILES = (0.5, 99.5)   # Auto: black/white at these percentiles of the image
LEVEL_STEPS = 1000               # slider resolution (black/white as 0..1000 of the data range)

BACKGROUND_LABELS = {
    "above_below": "Above/below",
    "ring": "Ring",
    "profile": "Profile baseline (min. top/bottom)",
}
EDGE_COLOR = "#eb6834"          # hand-set profile edges and their markers
BASELINE_COLOR = "#2a78d6"      # profiles, baseline and shaded band signal
PROFILE_H_COLOR = "#1f6fb4"     # plain horizontal profile (non-baseline modes)
PROFILE_V_COLOR = "#c2410c"     # plain vertical profile
ROI_COLOR = "#00d7ff"           # band boxes drawn on the image
ROI_SELECTED_COLOR = "#ff3b30"  # the selected band's box
HIGHLIGHT_COLOR = "#ffd60a"     # grid, drag preview, measured-region boxes
CANVAS_BACKGROUND = "#25282b"

ROTATION_LIMIT = 45.0  # degrees either way
ZOOM_LIMITS = (0.05, 8.0)
HANDLE_TOLERANCE = 6   # screen px: how close to a box edge counts as grabbing it

PAIRING_STATUS_LABELS = {
    "auto_matched": "Auto Matched",
    "manual_matched": "Manual Matched",
    "confirmed": "Confirmed",
    "missing_reference": "Missing Control",
    "missing_target": "Missing Protein",
    "ambiguous_target": "Duplicate Protein Lane",
    "ambiguous_reference": "Duplicate Control Lane",
    "manual_unpaired": "Manually Unpaired",
    "invalid_manual_reference": "Invalid Pairing",
    "duplicate_reference": "Pairing Conflict",
    "zero_reference": "Control Value Is Zero",
}

TABLE_COLUMNS = (
    ("role", "Image", 80), ("id", "ID", 42), ("name", "Name", 88), ("lane", "Lane", 48),
    ("mean", "Mean Intensity", 96), ("bg", "Local Background", 110),
    ("corrected", "Corrected Mean", 100), ("integrated", "Band Integrated Density", 150),
    ("target_integrated_density", "Protein Integrated Density", 160),
    ("reference_integrated_density", "Control Integrated Density", 160),
    ("target_over_reference", "Protein/Control", 100),
    ("pairing_status", "Pairing Status", 120), ("missing_reason", "Issue / Notes", 260),
)

LOG_FILE_NAME = "wb_analysis_log.txt"
ROI_FILE_SUFFIX = ".rois.json"
ROI_FILE_FORMAT = "wb_analysis_rois"
ROI_FILE_VERSION = 1


# ---------------------------------------------------------------------------
# Display mapping (view only; measurements always use the raw data)
# ---------------------------------------------------------------------------


def normalise_for_display(data: np.ndarray) -> np.ndarray:
    """Scale a 2-D array to float32 0..1 for display, keeping full precision.

    Non-finite pixels (NaN/inf, which float TIFFs may contain) are set to the
    minimum finite value so they render as background rather than garbage.
    """

    arr = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.float32)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    arr = np.where(finite, arr, lo)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def float_to_display_image(data: np.ndarray) -> Image.Image:
    """0..1 float data -> 8-bit greyscale image (the stored/exported view)."""

    return Image.fromarray(np.clip(np.rint(data * 255.0), 0, 255).astype(np.uint8))


def display_curve(values: np.ndarray, black: float, white: float, mode: str) -> np.ndarray:
    """Map 0..1 data to 0..1 grey: black/white points, then linear or log.

    Display only: never applied to the data used for detection or measurement.
    """

    t = np.clip((values - black) / max(white - black, 1e-6), 0.0, 1.0)
    if mode == "log":
        t = np.log1p(LOG_STRETCH * t) / np.log1p(LOG_STRETCH)
    return t


def display_lut(size: int, black: float, white: float, mode: str) -> np.ndarray:
    """Lookup table from ``size`` input levels (0..1 evenly spaced) to 8-bit grey."""

    levels = np.arange(size, dtype=np.float64) / (size - 1)
    return np.rint(display_curve(levels, black, white, mode) * 255).astype(np.uint8)


def array_to_qimage(arr: np.ndarray) -> QImage:
    """uint8 grey (H,W), RGB (H,W,3) or RGBA (H,W,4) array -> an independent QImage."""

    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        fmt, bpl = QImage.Format.Format_Grayscale8, w
    elif arr.shape[2] == 3:
        fmt, bpl = QImage.Format.Format_RGB888, 3 * w
    else:
        fmt, bpl = QImage.Format.Format_RGBA8888, 4 * w
    # .copy() detaches the QImage from the numpy buffer, which may be freed.
    return QImage(arr.tobytes(), w, h, bpl, fmt).copy()


def auto_display_levels(values: np.ndarray) -> tuple[float, float]:
    """Black/white points (0..1) at the AUTO_PERCENTILES of the image.

    Percentiles rather than min/max, so a few very bright pixels (a saturated
    ladder, dust) don't set white and leave real bands in the darkest greys.
    """

    arr = np.asarray(values, dtype=np.float64)
    step = max(1, int(np.ceil(np.sqrt(arr.size / 1_000_000))))  # sample large images
    arr = arr[::step, ::step] if arr.ndim >= 2 else arr[::step]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    black, white = (float(v) for v in np.percentile(arr, AUTO_PERCENTILES))
    if white - black < 1e-3:
        return 0.0, 1.0
    return max(0.0, black), min(1.0, white)


# ---------------------------------------------------------------------------
# Image geometry
# ---------------------------------------------------------------------------


def border_median(array: np.ndarray) -> float | tuple[int, ...]:
    """Median of an image's outermost pixels: the fill for rotated corners.

    Filling with the blot's own edge background (rather than black/white)
    keeps the new corners from looking like bands or skewing background
    estimates near the edge.
    """

    arr = np.asarray(array)
    edges = np.concatenate([arr[0], arr[-1], arr[1:-1, 0], arr[1:-1, -1]])
    if arr.ndim == 3:  # colour: per-channel median
        return tuple(int(round(float(np.median(edges[:, c])))) for c in range(arr.shape[2]))
    finite = edges[np.isfinite(edges)] if np.issubdtype(edges.dtype, np.floating) else edges
    return float(np.median(finite)) if finite.size else 0.0


def rotate_float(data: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 2-D float array (bilinear, canvas expanded to fit)."""

    fill = border_median(data)
    rotated = Image.fromarray(np.asarray(data, dtype=np.float32)).rotate(
        angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=fill,
    )
    return np.asarray(rotated, dtype=np.float32)


# ---------------------------------------------------------------------------
# Logging and ROI sidecar files
# ---------------------------------------------------------------------------


log = logging.getLogger("wb_analysis")


def setup_logging() -> Path | None:
    """Append to a log next to the scripts, or in ~/WB_Analysis if that's read-only."""

    if log.handlers:
        handler = log.handlers[0]
        return Path(getattr(handler, "baseFilename", "")) or None
    candidates = [
        Path(__file__).resolve().parent / LOG_FILE_NAME,
        Path.home() / "WB_Analysis" / LOG_FILE_NAME,
    ]
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 5 MB per file, keeping 3 old files, so the log can't grow without limit.
            handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        except OSError:
            continue
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        return path
    return None


def file_sha256(path: Path) -> str:
    """Content fingerprint: shows whether an image file has changed since ROIs were saved."""

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def roi_file_path(image_path: Path) -> Path:
    """Sidecar ROI file kept next to the image: blot.tif -> blot.tif.rois.json."""

    return image_path.with_name(image_path.name + ROI_FILE_SUFFIX)


def set_aside(path: Path, reason: str) -> Path | None:
    """Rename a ROI file that can't be used, so it is kept rather than overwritten."""

    backup = path.with_name(f"{path.name}.{reason}-{datetime.now():%Y%m%d-%H%M%S}.bak")
    try:
        os.replace(path, backup)
    except OSError as exc:
        log.warning("Could not set aside ROI file %s: %s", path, exc)
        return None
    log.warning("ROI file %s set aside as %s (%s)", path, backup.name, reason)
    return backup


def ensure_suffix(filename: str, suffix: str) -> str:
    """Add ``suffix`` when a save dialog returned a name without an extension."""

    return filename if Path(filename).suffix else filename + suffix


# ---------------------------------------------------------------------------
# Per-image state
# ---------------------------------------------------------------------------


@dataclass
class ImageState:
    key: str
    label: str
    image: Image.Image | None = None
    gray: np.ndarray | None = None
    path: Path | None = None
    rois: list[BandROI] = field(default_factory=list)
    reference_uid: int | None = None
    selected_uid: int | None = None
    # Full-precision display source (float32 scaled 0..1) for high-bit-depth
    # images. When present, contrast is applied before 8-bit quantisation,
    # so brightening reveals real detail instead of stretching 8-bit steps.
    display_data: np.ndarray | None = None
    # display_data quantised to 16 bits once, so contrast changes are a fast
    # 65536-entry lookup rather than float arithmetic on every pixel.
    display_u16: np.ndarray | None = None
    # Contrast-mapped pixmap shown in the view, and the settings it was made with.
    pixmap: QPixmap | None = None
    pixmap_key: tuple | None = None
    # Display-only contrast; never touches ``gray`` / quantification.
    # Black/white are fractions (0..1) of the image's data range.
    display_black: float = 0.0
    display_white: float = 1.0
    display_mode: str = "linear"
    display_range: tuple[float, float] = (0.0, 1.0)  # data values at 0 and 1, for labels
    detection_cache_key: tuple[float, int, str] | None = None
    detection_cache: tuple[list[BandROI], str] | None = None
    auto_polarity: str | None = None
    # Unrotated originals. ``image``/``gray``/``display_data`` above are always
    # derived from these, so repeated rotations never compound interpolation.
    source_image: Image.Image | None = None
    source_gray: np.ndarray | None = None
    source_display_data: np.ndarray | None = None
    rotation: float = 0.0  # degrees, positive = anticlockwise
    source_sha256: str | None = None
    # Content of the ROI file as last loaded/saved, so it is only rewritten
    # when ROIs, rotation or the category actually change.
    roi_file_snapshot: str | None = None
    roi_file_error_shown: bool = False
    # Profile-baseline edges set by hand, per ROI uid: {"left"|"right"|"top"|"bottom": image px}.
    edge_overrides: dict[int, dict[str, int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graphics items and the blot view
# ---------------------------------------------------------------------------


def cosmetic_pen(color: str, width: float, dash: tuple[float, ...] | None = None) -> QPen:
    """A pen whose width is in screen pixels, whatever the zoom."""

    pen = QPen(QColor(color))
    pen.setCosmetic(True)
    pen.setWidthF(width)
    if dash:
        pen.setDashPattern([d / max(width, 1.0) for d in dash])
    return pen


class RoiLabelItem(QGraphicsItem):
    """Band ID tag drawn just above a box's top-left corner at a constant screen size."""

    def __init__(self, text: str, color: str) -> None:
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.text = text
        self.color = QColor(color)
        self.font = QFont()
        self.font.setPointSizeF(9)
        self.font.setBold(True)
        metrics = QFontMetricsF(self.font)
        width = metrics.horizontalAdvance(text) + 6
        height = metrics.height() + 1
        self._rect = QRectF(0, -height, width, height)

    def boundingRect(self) -> QRectF:  # noqa: N802 (Qt API)
        return self._rect

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        painter.fillRect(self._rect, self.color)
        painter.setPen(QColor("white"))
        painter.setFont(self.font)
        painter.drawText(self._rect, Qt.AlignmentFlag.AlignCenter, self.text)


class BlotView(QGraphicsView):
    """Zoomable, pannable blot view. Scene coordinates are image pixels.

    Left drag: draw a region / band box, or move/resize the selected box.
    Middle drag: pan. Wheel: zoom about the pointer. Right-click (or
    Ctrl-click): rename/delete menu.
    """

    def __init__(self, app: "WBAnalyzerApp") -> None:
        super().__init__()
        self.app = app
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(CANVAS_BACKGROUND))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewport().setMouseTracking(True)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        self._pan_last: QPointF | None = None

    def _scene_point(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        button = event.button()
        modifiers = event.modifiers()
        if button == Qt.MouseButton.MiddleButton:
            self._pan_last = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        context_click = button == Qt.MouseButton.RightButton or (
            button == Qt.MouseButton.LeftButton
            and modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)
        )
        if context_click:
            self.app._canvas_context_menu(self._scene_point(event), event.globalPosition().toPoint())
            return
        if button == Qt.MouseButton.LeftButton:
            self.app._canvas_press(self._scene_point(event), event.position(), modifiers)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._pan_last is not None:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.app._canvas_drag(self._scene_point(event))
        else:
            self.app._canvas_hover(event.position(), event.modifiers())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.button() == Qt.MouseButton.MiddleButton and self._pan_last is not None:
            self._pan_last = None
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.app._canvas_release(self._scene_point(event))

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt API)
        delta = event.angleDelta().y()
        if delta:
            self.app._mousewheel(delta)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self.app._canvas_resized()


# ---------------------------------------------------------------------------
# Application window
# ---------------------------------------------------------------------------


class WBAnalyzerApp(QMainWindow):
    """Main window: zoomable blot view on the left; profiles and results on the right.

    Two ``ImageState`` objects (reference and target) hold everything per
    image; the attributes set in ``__init__`` mirror whichever is active.
    """

    # ----------------------------------------------------------------------
    # Window construction
    # ----------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        screen_width = available.width() if available else 1600
        screen_height = available.height() if available else 1000
        minimum_width = min(1050, max(760, screen_width - 40))
        minimum_height = min(680, max(540, screen_height - 80))
        self.resize(
            max(minimum_width, min(1480, screen_width - 80)),
            max(minimum_height, min(850, screen_height - 100)),
        )
        self.setMinimumSize(minimum_width, minimum_height)

        # Working copies of the active image's state; _sync_active_state
        # writes them back into image_states before anything else reads it.
        self.original_image: Image.Image | None = None
        self.gray: np.ndarray | None = None
        self.image_path: Path | None = None
        self.rois: list[BandROI] = []
        self.reference_uid: int | None = None
        self.selected_uid: int | None = None
        self.next_uid = 1
        self.image_states = {
            "reference": ImageState("reference", DEFAULT_CATEGORY["reference"]),
            "target": ImageState("target", DEFAULT_CATEGORY["target"]),
        }
        self.active_image_key = "reference"
        self.combined_measurements: list[tuple[str, BandMeasurement]] = []
        self.normalized_results: list[LaneNormalizationResult] = []
        self.pairing_result_by_uid: dict[int, LaneNormalizationResult] = {}

        # View / mouse state.
        self.drag_start: tuple[float, float] | None = None
        self.preview_item = None
        self._edit: dict | None = None            # move/resize of the selected box in progress
        self._roi_items: dict[int, tuple] = {}     # uid -> (rect item, label item)
        self._auto_fit = True
        self._sashes_placed = False

        # Settings (edited in the Settings window). Defaults: positive bands
        # (bright on dark, e.g. LI-COR fluorescence) measured with the profile baseline.
        self.polarity = "light"
        self.background_mode = "profile"
        self.sensitivity = 55.0
        self.min_area = 30
        self.region_padding = 6
        self.roi_gap = 0
        self.filter_below_mean = True
        self.marker_mode = "auto"
        self.settings_window: QDialog | None = None
        self.log_window: QDialog | None = None

        self._syncing_display = False
        self._syncing_rotation = False
        self._table_updating = False
        self._row_by_uid: dict[int, int] = {}

        self._rotation_timer = QTimer(self)
        self._rotation_timer.setSingleShot(True)
        self._rotation_timer.setInterval(150)
        self._rotation_timer.timeout.connect(self._apply_rotation)
        self._rotation_key: str | None = None

        self.log_path = setup_logging()
        log.info("Session started")
        self._build_ui()
        self._bind_shortcuts()
        self._redraw()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(4)
        outer.addLayout(self._build_toolbar())

        self.main_split = QSplitter(Qt.Orientation.Horizontal)
        self.main_split.setChildrenCollapsible(False)
        image_panel = QWidget()
        control_panel = QWidget()
        control_panel.setMinimumWidth(360)
        self._build_image_panel(image_panel)
        self._build_control_panel(control_panel)
        self.main_split.addWidget(image_panel)
        self.main_split.addWidget(control_panel)
        self.main_split.setStretchFactor(0, 2)
        self.main_split.setStretchFactor(1, 1)
        outer.addWidget(self.main_split, 1)

        self.status_label = QLabel("Load a control image and a protein image to begin.")
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.statusBar().addWidget(self.status_label, 1)

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.image_group = QButtonGroup(self)
        self.image_radios: dict[str, QRadioButton] = {}
        self.name_edits: dict[str, QLineEdit] = {}
        self._name_timers: dict[str, QTimer] = {}
        # One group per image: select it for editing, name it, load it.
        for key in ("reference", "target"):
            role = IMAGE_ROLE_NAMES[key]
            radio = QRadioButton(role)
            radio.setChecked(key == "reference")
            radio.clicked.connect(lambda _checked=False, k=key: self.activate_image(k))
            self.image_group.addButton(radio)
            self.image_radios[key] = radio
            bar.addWidget(radio)
            edit = QLineEdit(self.image_states[key].label)
            edit.setFixedWidth(100)
            edit.setToolTip("Name used for this image in the table, the CSV and exported file names. "
                            "Updates as you type.")
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(300)
            timer.timeout.connect(lambda k=key: self.apply_category_name(k))
            self._name_timers[key] = timer
            # Names apply as you type (after a short pause). Leaving the box,
            # or pressing Enter, puts the default back if it was left empty.
            edit.textEdited.connect(lambda _text, t=timer: t.start())
            edit.editingFinished.connect(lambda k=key: self.apply_category_name(k, finished=True))
            self.name_edits[key] = edit
            bar.addWidget(edit)
            load = QPushButton("Load…")
            load.setToolTip(f"Open the {role.lower()} image.")
            load.clicked.connect(lambda _checked=False, k=key: self.open_image(k))
            bar.addWidget(load)
            bar.addWidget(self._vline())

        self.mode_group = QButtonGroup(self)
        self.region_radio = QRadioButton("Split Selected Region")
        self.region_radio.setChecked(True)
        self.region_radio.setToolTip("Drag a box around a row of bands to detect and split them.")
        self.rect_radio = QRadioButton("Single Band ROI")
        self.rect_radio.setToolTip("Drag a box to add it as one band, exactly as drawn.")
        for radio in (self.region_radio, self.rect_radio):
            self.mode_group.addButton(radio)
            bar.addWidget(radio)
        delete = QPushButton("Delete ROI")
        delete.clicked.connect(self.delete_selected)
        clear = QPushButton("Clear All")
        clear.clicked.connect(self.clear_rois)
        bar.addSpacing(6)
        bar.addWidget(delete)
        bar.addWidget(clear)
        bar.addStretch(1)
        for text, slot in (("Export Image", self.export_annotated), ("Export CSV", self.export_csv),
                           ("View Log", self.open_log_viewer)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            bar.addWidget(button)
        return bar

    @staticmethod
    def _vline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _build_image_panel(self, panel: QWidget) -> None:
        """Left side: zoomable view, then the display and rotation bars."""

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.view = BlotView(self)
        layout.addWidget(self.view, 1)

        display_bar = QHBoxLayout()
        levels_hint = ("Display only: pixel values at or below Black show as black, at or above White as "
                       "white. Detection and measurements always use the original data.")
        self.level_sliders: dict[str, QSlider] = {}
        self.level_labels: dict[str, QLabel] = {}
        for which, text in (("black", "Black"), ("white", "White")):
            label = QLabel(text)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, LEVEL_STEPS)
            slider.setValue(0 if which == "black" else LEVEL_STEPS)
            slider.valueChanged.connect(lambda _v, w=which: self._display_levels_changed(w))
            value_label = QLabel("")
            value_label.setMinimumWidth(56)
            for widget in (label, slider):
                widget.setToolTip(levels_hint)
            display_bar.addWidget(label)
            display_bar.addWidget(slider, 1)
            display_bar.addWidget(value_label)
            self.level_sliders[which] = slider
            self.level_labels[which] = value_label
        self.mode_box = QComboBox()
        self.mode_box.addItems(list(DISPLAY_MODES))
        self.mode_box.currentTextChanged.connect(lambda _t: self._display_mode_changed())
        self.mode_box.setToolTip(
            "Linear: grey level proportional to intensity (twice as bright looks twice as bright). "
            "Log: lifts faint bands and compresses bright ones (e.g. a saturated ladder); "
            "differences no longer look proportional. Display only.")
        display_bar.addWidget(self.mode_box)
        auto_button = QPushButton("Auto")
        auto_button.clicked.connect(lambda: self.auto_contrast())
        auto_button.setToolTip(
            "Set Black and White to the 0.5th and 99.5th percentiles of this image, so a few very "
            "bright pixels don't leave the bands dark. Applied automatically when an image is opened.")
        display_bar.addWidget(auto_button)
        fit_button = QPushButton("Fit")
        fit_button.clicked.connect(self._fit_image)
        fit_button.setToolTip("Fit the blot to the view. It stays fitted as you resize panes until you "
                              "zoom with the mouse wheel. Middle-drag pans.")
        display_bar.addSpacing(6)
        display_bar.addWidget(fit_button)
        layout.addLayout(display_bar)

        geometry_bar = QHBoxLayout()
        geometry_bar.addWidget(QLabel("Rotate (this image)"))
        # The slider works in tenths of a degree; the box shows degrees.
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(int(-ROTATION_LIMIT * 10), int(ROTATION_LIMIT * 10))
        self.rotation_slider.valueChanged.connect(self._rotation_slider_moved)
        self.rotation_slider.sliderReleased.connect(self._rotation_changed)
        geometry_bar.addWidget(self.rotation_slider, 1)
        self.rotation_spin = QDoubleSpinBox()
        self.rotation_spin.setRange(-ROTATION_LIMIT, ROTATION_LIMIT)
        self.rotation_spin.setDecimals(1)
        self.rotation_spin.setSingleStep(0.1)
        self.rotation_spin.setSuffix("°")
        self.rotation_spin.setKeyboardTracking(False)  # apply on Enter/arrows, not every keystroke
        self.rotation_spin.valueChanged.connect(self._rotation_spin_changed)
        geometry_bar.addWidget(self.rotation_spin)
        reset = QPushButton("Reset")
        reset.clicked.connect(self.reset_rotation)
        geometry_bar.addWidget(reset)
        geometry_bar.addWidget(self._vline())
        self.grid_check = QCheckBox("Show grid")
        self.grid_check.toggled.connect(lambda _c: self._redraw())
        geometry_bar.addWidget(self.grid_check)
        geometry_bar.addWidget(QLabel("Spacing"))
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(5, 2000)
        self.grid_spin.setSingleStep(5)
        self.grid_spin.setValue(50)
        self.grid_spin.setSuffix(" px")
        self.grid_spin.setKeyboardTracking(False)
        self.grid_spin.valueChanged.connect(lambda _v: self._redraw())
        geometry_bar.addWidget(self.grid_spin)
        layout.addLayout(geometry_bar)

    def _build_control_panel(self, panel: QWidget) -> None:
        """Right side: settings summary, profile plots, results table."""

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        settings_bar = QHBoxLayout()
        settings_button = QPushButton("Settings...")
        settings_button.clicked.connect(self.open_settings)
        settings_bar.addWidget(settings_button)
        self.settings_summary = QLabel()
        self.settings_summary.setStyleSheet("color: #555;")
        self.settings_summary.setToolTip("Current band polarity and background method. "
                                         "Change them, and the detection parameters, in Settings.")
        settings_bar.addWidget(self.settings_summary, 1)
        layout.addLayout(settings_bar)
        self._update_settings_summary()

        # Profile plots above, results table below; drag the divider to resize.
        self.control_split = QSplitter(Qt.Orientation.Vertical)
        self.control_split.setChildrenCollapsible(False)
        layout.addWidget(self.control_split, 1)

        profile_group = QGroupBox("Intensity Profiles (Selected Band)")
        profile_layout = QVBoxLayout(profile_group)
        profile_layout.setContentsMargins(5, 5, 5, 5)
        self._profile_signature: tuple | None = None
        self._profile_pb = None
        self._drag_edge: str | None = None
        if HAVE_MATPLOTLIB:
            self.profile_figure = Figure(figsize=(4.4, 2.1), dpi=90, layout="constrained")
            self.profile_axes = (
                self.profile_figure.add_subplot(1, 2, 1),
                self.profile_figure.add_subplot(1, 2, 2),
            )
            self.profile_canvas = FigureCanvasQTAgg(self.profile_figure)
            self.profile_canvas.setMinimumHeight(140)
            # Keyboard focus on click, so matplotlib key events reach the plot.
            self.profile_canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            # The same matplotlib callbacks as the Tk version: these events are
            # backend-independent, so the edge dragging code is unchanged.
            self.profile_canvas.mpl_connect("button_press_event", self._on_profile_press)
            self.profile_canvas.mpl_connect("motion_notify_event", self._on_profile_motion)
            self.profile_canvas.mpl_connect("button_release_event", self._on_profile_release)
            self.profile_canvas.setToolTip(
                "Profile baseline mode: click or drag in the left plot to move the left/right edge, "
                "in the right plot to move the top/bottom edge. Dashed = automatic, solid = set by hand. "
                "The shaded area is the band signal counted."
            )
            profile_layout.addWidget(self.profile_canvas, 1)
            reset_edges = QPushButton("Reset Edges")
            reset_edges.setToolTip("Return the selected band to automatic (profile-minimum) edges.")
            reset_edges.clicked.connect(self.reset_edges)
            profile_layout.addWidget(reset_edges, 0, Qt.AlignmentFlag.AlignRight)
        else:
            note = QLabel("Install matplotlib to see profile plots (pip install matplotlib). "
                          "Profiles are still exported with the CSV.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #555;")
            profile_layout.addWidget(note)
        self.control_split.addWidget(profile_group)

        table_group = QGroupBox("Quantification Results (Protein/Control by Lane)  ⓘ")
        table_group.setToolTip(
            "Bands are paired by lane: a band is compared with the control band in the same lane "
            "when each image has exactly one band in that lane. Duplicate or missing lanes are "
            "flagged in Issue / Notes and never silently calculated."
        )
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(5, 5, 5, 5)
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([heading for _key, heading, _w in TABLE_COLUMNS])
        for column, (_key, _heading, width) in enumerate(TABLE_COLUMNS):
            self.table.setColumnWidth(column, width)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._tree_select)
        self.table.cellDoubleClicked.connect(lambda _r, _c: self.rename_selected())
        table_layout.addWidget(self.table)
        self.control_split.addWidget(table_group)

    def _bind_shortcuts(self) -> None:
        for sequence in (QKeySequence(Qt.Key.Key_Delete), QKeySequence(Qt.Key.Key_Backspace)):
            QShortcut(sequence, self, activated=self._shortcut_delete)
        # StandardKey maps to Cmd on macOS and Ctrl elsewhere.
        QShortcut(QKeySequence.StandardKey.Open, self, activated=lambda: self.open_image(self.active_image_key))
        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.export_csv)

    def _shortcut_delete(self) -> None:
        # Never delete a band while the user is editing text in a box.
        if isinstance(QApplication.focusWidget(), (QLineEdit, QAbstractSpinBox, QComboBox)):
            return
        self.delete_selected()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        if not self._sashes_placed:
            self._sashes_placed = True
            QTimer.singleShot(0, self._place_sashes)

    def _place_sashes(self) -> None:
        """Set the starting split once the window has its real size."""

        width = self.main_split.width()
        self.main_split.setSizes([int(width * 0.66), width - int(width * 0.66)])  # image gets two thirds
        height = self.control_split.height()
        self.control_split.setSizes([int(height * 0.45), height - int(height * 0.45)])
        if self._auto_fit:
            self._fit_image()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    # ----------------------------------------------------------------------
    # Image loading and switching
    # ----------------------------------------------------------------------

    def open_image(self, key: str) -> None:
        label = self.image_states[key].label
        filename, _filter = QFileDialog.getOpenFileName(
            self, f"Select {label} Image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*)",
        )
        if filename:
            self.load_image(key, filename)

    def load_image(self, key: str, filename: str) -> bool:
        """Load an image file into the reference or target slot. Returns success."""

        label = self.image_states[key].label
        display_data: np.ndarray | None = None
        try:
            try:
                # Scientific TIFF (LI-COR, Odyssey, Typhoon, etc.). Analysis
                # keeps the raw values; the display keeps float precision.
                raw = tifffile.imread(filename)
                file_format = f"TIFF {raw.dtype}"
                loaded_gray = raw.astype(np.float32)
                if loaded_gray.ndim == 2:
                    display_data = normalise_for_display(loaded_gray)
                    loaded_image = float_to_display_image(display_data)
                else:
                    display = normalise_for_display(loaded_gray)
                    loaded_image = Image.fromarray((display * 255).astype(np.uint8))
                image = loaded_image

            except Exception:
                # Standard images
                image = Image.open(filename)
                image.load()
                file_format = f"{image.format or 'image'} {image.mode}"

                if image.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
                    # 16-bit / 32-bit greyscale (e.g. 16-bit PNG). Converting
                    # to RGB would clip every value above 255, so keep the raw
                    # values for analysis and a float source for display.
                    loaded_gray = np.asarray(image).astype(np.float32)
                    display_data = normalise_for_display(loaded_gray)
                    loaded_image = float_to_display_image(display_data)
                    image = loaded_image
                else:
                    if image.mode not in ("L", "RGB", "RGBA"):
                        image = image.convert("RGB")
                    loaded_image = image.copy()
                    loaded_gray = to_gray_array(np.asarray(image))
        except Exception as exc:
            log.error("Failed to open %s image %s: %s", key, filename, exc)
            QMessageBox.critical(self, "Unable to Open Image", str(exc))
            return False
        try:
            sha256 = file_sha256(Path(filename))
        except OSError:
            sha256 = None
        self._sync_active_state()
        state = self.image_states[key]
        state.source_sha256 = sha256
        state.roi_file_error_shown = False
        state.image = loaded_image
        state.gray = loaded_gray
        state.path = Path(filename)
        state.rois = []
        state.reference_uid = None
        state.selected_uid = None
        state.edge_overrides = {}
        state.display_data = display_data
        state.display_u16 = None
        state.pixmap = None
        state.pixmap_key = None
        if display_data is not None:
            finite = loaded_gray[np.isfinite(loaded_gray)]
            state.display_range = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        else:
            state.display_range = (0.0, 255.0)
        state.display_mode = "linear"
        state.display_black, state.display_white = auto_display_levels(self._display_source(state))
        state.detection_cache_key = None
        state.detection_cache = None
        state.auto_polarity = None
        state.source_image = loaded_image
        state.source_gray = loaded_gray
        state.source_display_data = display_data
        state.rotation = 0.0
        self._sync_display_controls(state)
        self._rotation_timer.stop()
        log.info(
            "Opened %s image (%s): %s | %d x %d px | %s | sha256 %s",
            key, label, state.path, image.width, image.height, file_format, sha256 or "unavailable",
        )
        roi_note = self._load_roi_file(key)
        self._set_rotation_control(state.rotation)
        self.active_image_key = key
        self.image_radios[key].setChecked(True)
        self.original_image = state.image
        self.gray = state.gray
        self.image_path = state.path
        self.rois = state.rois
        self.reference_uid = None
        self.selected_uid = None
        self._update_results()
        self._fit_image()
        self.set_status(
            f"Loaded {label} image: {self.image_path.name} | {image.width} × {image.height} px | {roi_note}"
        )
        return True

    def activate_image(self, key: str, fit: bool = True) -> None:
        if key not in self.image_states:
            return
        self._sync_active_state()
        self.active_image_key = key
        self.image_radios[key].setChecked(True)
        state = self.image_states[key]
        self.original_image = state.image
        self.gray = state.gray
        self.image_path = state.path
        self.rois = state.rois
        self.reference_uid = state.reference_uid
        self.selected_uid = state.selected_uid
        self._sync_display_controls(state)
        self._rotation_timer.stop()
        self._set_rotation_control(state.rotation)
        if fit and self.original_image is not None:
            self._fit_image()
        else:
            self._redraw()
        self._select_table_row(self.selected_uid)
        name = self.image_path.name if self.image_path else "none selected"
        self.set_status(f"Now editing: {state.label} image | {name}")

    def _sync_active_state(self) -> None:
        state = self.image_states[self.active_image_key]
        state.image = self.original_image
        state.gray = self.gray
        state.path = self.image_path
        state.rois = self.rois
        state.reference_uid = self.reference_uid
        state.selected_uid = self.selected_uid

    def apply_category_name(self, key: str, finished: bool = False) -> None:
        """Use the box's text as the image's name (empty box = default name).

        Only labels change, so this refreshes the table and the ROI file
        without re-measuring any bands. ``finished`` (Enter or leaving the
        box) also writes the default back into an empty box.
        """

        self._name_timers[key].stop()
        text = self.name_edits[key].text().strip()
        name = text or DEFAULT_CATEGORY[key]
        if finished and not text:
            self.name_edits[key].setText(name)
        state = self.image_states[key]
        if state.label == name:
            return
        state.label = name
        self._refresh_results_table()
        self._save_roi_file(key)

    # ----------------------------------------------------------------------
    # Display contrast (view only)
    # ----------------------------------------------------------------------

    def _display_source(self, state: ImageState) -> np.ndarray | None:
        """The 0..1 values the display maps (before black/white/mode)."""

        if state.display_data is not None:
            return state.display_data
        if state.image is None:
            return None
        return np.asarray(state.image.convert("L"), dtype=np.float64) / 255.0

    def _display_pixmap(self, state: ImageState) -> QPixmap:
        """The contrast-mapped full-resolution pixmap, rebuilt only when settings change."""

        key = (state.display_black, state.display_white, state.display_mode)
        if state.pixmap is not None and state.pixmap_key == key:
            return state.pixmap
        if state.display_data is not None:
            if state.display_u16 is None:
                clean = np.clip(np.nan_to_num(state.display_data, nan=0.0), 0.0, 1.0)
                state.display_u16 = np.rint(clean * 65535).astype(np.uint16)
            mapped = display_lut(65536, *key)[state.display_u16]
        else:
            image = state.image
            if image.mode not in ("L", "RGB", "RGBA"):
                image = image.convert("RGB")
            pixels = np.asarray(image)
            lut = display_lut(256, *key)
            if pixels.ndim == 3 and pixels.shape[2] == 4:
                mapped = pixels.copy()
                mapped[..., :3] = lut[pixels[..., :3]]  # alpha kept
            else:
                mapped = lut[pixels]
        state.pixmap = QPixmap.fromImage(array_to_qimage(mapped))
        state.pixmap_key = key
        return state.pixmap

    @staticmethod
    def _level_value(state: ImageState, fraction: float) -> str:
        lo, hi = state.display_range
        return f"{lo + fraction * (hi - lo):.3g}"

    def _sync_display_controls(self, state: ImageState) -> None:
        """Show an image's stored black/white/mode on the controls without redrawing."""

        self._syncing_display = True
        try:
            self.level_sliders["black"].setValue(round(state.display_black * LEVEL_STEPS))
            self.level_sliders["white"].setValue(round(state.display_white * LEVEL_STEPS))
            self.mode_box.setCurrentText(next(k for k, v in DISPLAY_MODES.items() if v == state.display_mode))
            self.level_labels["black"].setText(self._level_value(state, state.display_black))
            self.level_labels["white"].setText(self._level_value(state, state.display_white))
        finally:
            self._syncing_display = False

    def _display_levels_changed(self, moved: str) -> None:
        if self._syncing_display:
            return
        state = self.image_states[self.active_image_key]
        black = self.level_sliders["black"].value() / LEVEL_STEPS
        white = self.level_sliders["white"].value() / LEVEL_STEPS
        gap = 2 / LEVEL_STEPS
        if white - black < gap:  # keep White above Black: push the other slider
            if moved == "black":
                white = min(1.0, black + gap)
                black = white - gap
            else:
                black = max(0.0, white - gap)
                white = black + gap
        if (black, white) == (state.display_black, state.display_white):
            return
        state.display_black, state.display_white = black, white
        self._sync_display_controls(state)
        if self.original_image is not None:
            self._redraw()

    def _display_mode_changed(self) -> None:
        if self._syncing_display:
            return
        state = self.image_states[self.active_image_key]
        state.display_mode = DISPLAY_MODES.get(self.mode_box.currentText(), "linear")
        if self.original_image is not None:
            self._redraw()

    def auto_contrast(self, announce: bool = True) -> None:
        state = self.image_states[self.active_image_key]
        source = self._display_source(state)
        if source is None:
            return
        state.display_black, state.display_white = auto_display_levels(source)
        self._sync_display_controls(state)
        if self.original_image is not None:
            self._redraw()
        if announce:
            self.set_status(
                f"Auto contrast ({state.label} image): black {self._level_value(state, state.display_black)}, "
                f"white {self._level_value(state, state.display_white)} "
                f"({AUTO_PERCENTILES[0]:g}th / {AUTO_PERCENTILES[1]:g}th percentiles). Display only."
            )

    # ----------------------------------------------------------------------
    # Rotation
    # ----------------------------------------------------------------------

    def _set_rotation_control(self, angle: float) -> None:
        """Show an image's stored angle on the slider and box without triggering a rotation."""

        self._syncing_rotation = True
        try:
            self.rotation_slider.setValue(int(round(angle * 10)))
            self.rotation_spin.setValue(round(angle, 1))
        finally:
            self._syncing_rotation = False

    def _rotation_slider_moved(self, value: int) -> None:
        if self._syncing_rotation:
            return
        self._syncing_rotation = True
        try:
            self.rotation_spin.setValue(value / 10)
        finally:
            self._syncing_rotation = False
        # While the slider is held, a confirmation dialog would fight the
        # mouse grab: if the image has ROIs, wait for the release.
        if self.rotation_slider.isSliderDown() and self.rois:
            return
        self._rotation_changed()

    def _rotation_spin_changed(self, value: float) -> None:
        if self._syncing_rotation:
            return
        self._syncing_rotation = True
        try:
            self.rotation_slider.setValue(int(round(value * 10)))
        finally:
            self._syncing_rotation = False
        self._rotation_changed()

    def _read_rotation(self) -> float:
        return round(max(-ROTATION_LIMIT, min(ROTATION_LIMIT, float(self.rotation_spin.value()))), 1)

    def _rotation_changed(self) -> None:
        if self._syncing_rotation:
            return
        state = self.image_states[self.active_image_key]
        angle = self._read_rotation()
        if state.source_gray is None or angle == state.rotation:
            if state.source_gray is None:
                self._set_rotation_control(0.0)
            return
        if state.source_gray.ndim != 2:
            self._set_rotation_control(0.0)
            self.set_status("Rotation is only available for single-channel (greyscale) images.")
            return
        if self.rois:
            # Axis-aligned ROIs can't follow a rotated band, so they must go.
            count = len(self.rois)
            answer = QMessageBox.question(
                self, APP_TITLE,
                f"Rotating the {state.label} image will clear its {count} ROI(s). Rotate and clear them?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                self._set_rotation_control(state.rotation)
                return
            self.rois.clear()
            self.reference_uid = None
            self.selected_uid = None
            self._sync_active_state()
            self._update_results()
        # Debounce: the slider fires continuously while dragging; only do the
        # full-resolution rotation once it pauses.
        self._rotation_key = self.active_image_key
        self._rotation_timer.start()

    def _apply_rotation(self) -> None:
        key = self._rotation_key
        if key != self.active_image_key:
            return
        angle = self._read_rotation()
        state = self.image_states[key]
        if state.source_gray is None or angle == state.rotation:
            return
        self._rotate_state(state, angle)
        self.original_image = state.image
        self.gray = state.gray
        self._redraw()
        if self._auto_fit:
            self._fit_image()
        log.info("Rotated %s image %s to %+.1f deg", key, state.path, angle)
        self._save_roi_file(key)
        self.set_status(
            f"{state.label} image rotated to {angle:+.1f}° "
            f"({state.image.width} × {state.image.height} px)."
        )

    @staticmethod
    def _rotate_state(state: ImageState, angle: float) -> None:
        """Rebuild an image's data and display from its unrotated source."""

        if angle == 0.0:
            # Exact originals: no interpolation at all at 0°.
            state.image = state.source_image
            state.gray = state.source_gray
            state.display_data = state.source_display_data
        else:
            state.gray = rotate_float(state.source_gray, angle)
            if state.source_display_data is not None:
                state.display_data = rotate_float(state.source_display_data, angle)
                state.image = float_to_display_image(state.display_data)
            else:
                source = state.source_image
                fill = border_median(np.asarray(source))
                if source.mode == "RGBA" and isinstance(fill, tuple):
                    fill = fill[:3] + (255,)
                elif source.mode == "L" and isinstance(fill, tuple):
                    fill = fill[0]
                elif isinstance(fill, float):
                    fill = int(round(fill))
                state.image = source.rotate(
                    angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=fill,
                )
        state.rotation = angle
        state.display_u16 = None
        state.pixmap = None
        state.pixmap_key = None
        state.detection_cache_key = None
        state.detection_cache = None
        state.auto_polarity = None

    def reset_rotation(self) -> None:
        self.rotation_spin.setValue(0.0)  # its valueChanged handler applies it

    # ----------------------------------------------------------------------
    # ROI sidecar files
    # ----------------------------------------------------------------------

    @staticmethod
    def _roi_file_core(state: ImageState) -> dict:
        """Everything in a ROI file except the save time (used for change checks)."""

        width, height = state.source_gray.shape[1], state.source_gray.shape[0]
        return {
            "format": ROI_FILE_FORMAT,
            "version": ROI_FILE_VERSION,
            "image_file": state.path.name if state.path else "",
            "image_sha256": state.source_sha256,
            "image_size_unrotated": [int(width), int(height)],
            "rotation_deg": round(float(state.rotation), 1),
            "category": state.label,
            "coordinates": "pixels in the image after rotation_deg is applied",
            "rois": [
                {
                    "name": roi.name, "lane": int(roi.lane), "source": roi.source,
                    "x": int(roi.x), "y": int(roi.y), "width": int(roi.width), "height": int(roi.height),
                    **(
                        {"profile_edges": {k: int(v) for k, v in sorted(state.edge_overrides[roi.uid].items())}}
                        if state.edge_overrides.get(roi.uid) else {}
                    ),
                }
                for roi in state.rois
            ],
        }

    def _save_roi_file(self, key: str) -> None:
        """Write the image's ROI file if its ROIs, rotation or category changed."""

        state = self.image_states[key]
        if state.path is None or state.source_gray is None:
            return
        core = self._roi_file_core(state)
        snapshot = json.dumps(core, sort_keys=True)
        if snapshot == state.roi_file_snapshot:
            return
        path = roi_file_path(state.path)
        if not state.rois and state.rotation == 0.0 and not path.exists():
            state.roi_file_snapshot = snapshot  # nothing worth saving yet
            return
        payload = dict(core, saved=datetime.now().isoformat(timespec="seconds"))
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, path)  # never leaves a half-written ROI file
        except OSError as exc:
            log.warning("Could not save ROI file %s: %s", path, exc)
            if not state.roi_file_error_shown:
                state.roi_file_error_shown = True
                QMessageBox.warning(
                    self, APP_TITLE,
                    f"Could not save the ROI file next to the {state.label} image:\n{exc}\n\n"
                    "ROIs will not be restored next time this image is opened.",
                )
            return
        state.roi_file_snapshot = snapshot
        state.roi_file_error_shown = False
        log.info("Saved ROI file %s (%d ROIs, rotation %+.1f deg)", path, len(state.rois), state.rotation)

    def _load_roi_file(self, key: str) -> str:
        """Restore rotation and ROIs from the image's ROI file. Returns a status note."""

        state = self.image_states[key]
        path = roi_file_path(state.path)
        state.roi_file_snapshot = None
        if not path.exists():
            log.info("No ROI file for %s", state.path)
            return "no saved ROIs"
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("format") != ROI_FILE_FORMAT:
                raise ValueError("not a WB analysis ROI file")
            saved_size = [int(v) for v in data["image_size_unrotated"]]
            rotation = round(float(data.get("rotation_deg", 0.0)), 1)
            entries = list(data.get("rois", []))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("Unreadable ROI file %s: %s", path, exc)
            backup = set_aside(path, "unreadable")
            QMessageBox.warning(
                self, APP_TITLE,
                f"The saved ROI file for this image could not be read ({exc}).\n"
                + (f"It has been kept as {backup.name}." if backup else ""),
            )
            return "saved ROI file unreadable"

        actual_size = [int(state.source_gray.shape[1]), int(state.source_gray.shape[0])]
        if saved_size != actual_size:
            log.warning("ROI file %s is for a %s image, but %s is %s", path, saved_size, state.path, actual_size)
            backup = set_aside(path, "size-mismatch")
            QMessageBox.warning(
                self, APP_TITLE,
                f"The saved ROIs were for a {saved_size[0]} × {saved_size[1]} px image, but this image is "
                f"{actual_size[0]} × {actual_size[1]} px, so they were not loaded."
                + (f"\nThe old ROI file has been kept as {backup.name}." if backup else ""),
            )
            return "saved ROIs were for a different-sized image"
        saved_hash = data.get("image_sha256")
        if saved_hash and state.source_sha256 and saved_hash != state.source_sha256:
            log.warning("ROI file %s was saved for different image contents (same size)", path)
            answer = QMessageBox.question(
                self, APP_TITLE,
                "This image file has changed since its ROIs were saved (same size, different contents).\n\n"
                "Load the saved ROIs anyway?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                set_aside(path, "image-changed")
                return "saved ROIs not loaded (image changed)"

        if rotation and state.source_gray.ndim == 2:
            self._rotate_state(state, max(-ROTATION_LIMIT, min(ROTATION_LIMIT, rotation)))
        height, width = state.gray.shape[:2]
        rois: list[BandROI] = []
        state.edge_overrides = {}
        for entry in entries:
            try:
                roi = clamp_roi(
                    BandROI(
                        uid=self.next_uid, x=int(entry["x"]), y=int(entry["y"]),
                        width=int(entry["width"]), height=int(entry["height"]),
                        name=str(entry.get("name") or f"Band {self.next_uid}"),
                        lane=int(entry.get("lane", 0)),
                        source=str(entry.get("source", "Automatic")),
                    ),
                    (height, width),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if roi.width < 1 or roi.height < 1:
                continue
            saved_edges = entry.get("profile_edges")
            if isinstance(saved_edges, dict):
                edges = {}
                for edge_key in EDGE_KEYS:
                    try:
                        if edge_key in saved_edges:
                            edges[edge_key] = int(saved_edges[edge_key])
                    except (TypeError, ValueError):
                        pass
                if edges:
                    state.edge_overrides[roi.uid] = edges
            rois.append(roi)
            self.next_uid += 1
        state.rois = rois
        if len(rois) == len(entries) and saved_hash == state.source_sha256:
            state.roi_file_snapshot = json.dumps(self._roi_file_core(state), sort_keys=True)
        log.info("Loaded ROI file %s (%d ROIs, rotation %+.1f deg)", path, len(rois), state.rotation)
        rotated = f", rotated {state.rotation:+.1f}°" if state.rotation else ""
        return f"restored {len(rois)} ROI(s){rotated} from {path.name}"

    # ----------------------------------------------------------------------
    # Band detection
    # ----------------------------------------------------------------------

    def _get_full_detection(self) -> tuple[list[BandROI], str]:
        """Return full-image candidates cached for the active parameters."""

        if self.gray is None:
            return [], self.polarity
        state = self.image_states[self.active_image_key]
        cache_key = (round(float(self.sensitivity), 4), max(4, int(self.min_area)), self.polarity)
        if state.detection_cache_key == cache_key and state.detection_cache is not None:
            return state.detection_cache
        detected = detect_bands(
            self.gray,
            sensitivity=cache_key[0],
            min_area=cache_key[1],
            polarity=cache_key[2],
            first_uid=1,
        )
        state.detection_cache_key = cache_key
        state.detection_cache = detected
        if cache_key[2] == "auto":
            state.auto_polarity = detected[1]
        return detected

    def auto_detect(self) -> None:
        if self.gray is None:
            QMessageBox.information(self, APP_TITLE, "Please open a Western blot image first.")
            return
        try:
            cached_rois, actual = self._get_full_detection()
            rois = [
                replace(roi, uid=self.next_uid + offset, name=f"Band {self.next_uid + offset}")
                for offset, roi in enumerate(cached_rois)
            ]
        except Exception as exc:
            QMessageBox.critical(self, "Auto Detection Failed", str(exc))
            return
        candidate_count = len(rois)
        rois, marker_removed = filter_marker_lanes(rois, self.marker_mode)
        rois = expand_band_rois(
            rois,
            (0, 0, self.gray.shape[1], self.gray.shape[0]),
            padding=max(0, self.region_padding),
            gap=max(0, self.roi_gap),
        )
        self.rois = rois
        self.next_uid = max((r.uid for r in rois), default=0) + 1
        self.reference_uid = None
        self.selected_uid = rois[0].uid if rois else None
        self._update_results()
        self._redraw()
        self.set_status(
            f"Full-image detection complete: kept {len(rois)}/{candidate_count} bands; "
            f"{len(marker_removed)} removed by marker filter; ROIs expanded by "
            f"{max(0, self.region_padding)} px; polarity used: {actual}."
        )

    def _detect_region(
        self,
        region: tuple[float, float, float, float],
        replace_uid: int | None = None,
    ) -> None:
        """Detect bands in a user region and merge them into the ROI list."""

        if self.gray is None:
            return
        x1, y1, x2, y2 = (int(round(v)) for v in region)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            found, actual = detect_bands_in_region(
                self.gray,
                (x1, y1, x2, y2),
                sensitivity=self.sensitivity,
                min_area=max(4, self.min_area),
                polarity=self.polarity,
                first_uid=self.next_uid,
                full_detection=self._get_full_detection(),
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Region Split Failed", str(exc))
            return
        QApplication.restoreOverrideCursor()
        if not found:
            self.set_status("No bands found in this region. Try raising Sensitivity, lowering Minimum Area, "
                            "or extending the selection further above and below the bands.")
            self._redraw()
            return

        candidate_count = len(found)
        found, marker_removed = filter_marker_lanes(found, self.marker_mode)
        found = align_rois_to_region_signal(
            self.gray,
            found,
            (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
            polarity=actual,
        )
        found = expand_band_rois(
            found,
            (0, 0, self.gray.shape[1], self.gray.shape[0]),
            padding=max(0, self.region_padding),
            gap=max(0, self.roi_gap),
        )
        signal_threshold: float | None = None
        if self.filter_below_mean:
            found, signal_threshold = filter_bands_below_mean_signal(
                self.gray,
                found,
                polarity=actual,
                region=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
            )
        weak_excluded_count = candidate_count - len(marker_removed) - len(found)
        if not found:
            self.set_status(
                f"All {candidate_count} candidates in the region were filtered out: "
                f"{len(marker_removed)} as marker, {weak_excluded_count} as weak. "
                "Try changing the Marker Filter or turning off the weak-candidate filter."
            )
            self._redraw()
            return

        # Re-running a region should replace bands inside it instead of adding
        # duplicates. A deliberately selected broad ROI is always removed.
        rx1, rx2 = sorted((x1, x2))
        ry1, ry2 = sorted((y1, y2))
        kept: list[BandROI] = []
        removed_uids: set[int] = set()
        for old in self.rois:
            cx, cy = old.center
            inside = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
            if old.uid == replace_uid or inside:
                removed_uids.add(old.uid)
            else:
                kept.append(old)
        self.rois = assign_lanes(kept + found)
        self.next_uid = max((r.uid for r in self.rois), default=0) + 1
        self.selected_uid = found[0].uid
        if self.reference_uid in removed_uids:
            self.reference_uid = None
        self._update_results()
        self._redraw()
        filter_text = ""
        if signal_threshold is not None:
            filter_text = (
                f"; mean corrected signal threshold {signal_threshold:.2f}, "
                f"{weak_excluded_count} weak candidates excluded"
            )
        self.set_status(
            f"Region split complete: kept {len(found)}/{candidate_count} bands"
            f"; {len(marker_removed)} removed by marker filter{filter_text}; "
            f"ROIs expanded by {max(0, self.region_padding)} px, "
            f"gap {max(0, self.roi_gap)} px; "
            f"polarity used: {actual}."
        )

    # ----------------------------------------------------------------------
    # View, zoom and mouse
    # ----------------------------------------------------------------------

    @property
    def zoom(self) -> float:
        return self.view.transform().m11()

    def _fit_image(self) -> None:
        """Scale the image to fill the view (enlarging small blots too)."""

        self._auto_fit = True
        if self.original_image is None:
            self._redraw()
            return
        viewport = self.view.viewport().size()
        cw, ch = max(1, viewport.width() - 10), max(1, viewport.height() - 10)
        zoom = min(cw / self.original_image.width, ch / self.original_image.height, ZOOM_LIMITS[1])
        zoom = max(zoom, ZOOM_LIMITS[0])
        self.view.setTransform(QTransform.fromScale(zoom, zoom))
        self._redraw()
        self.view.centerOn(self.original_image.width / 2, self.original_image.height / 2)

    def _canvas_resized(self) -> None:
        # Keep the blot fitted to the view as panes/window are resized, until
        # the user zooms by hand (Fit restores this).
        if self._auto_fit and self.original_image is not None:
            self._fit_image()

    def _mousewheel(self, delta: int) -> None:
        if self.original_image is None:
            return
        factor = 1.12 if delta > 0 else 1 / 1.12
        old = self.zoom
        new = float(np.clip(old * factor, *ZOOM_LIMITS))
        if new == old:
            return
        self._auto_fit = False
        self.view.scale(new / old, new / old)  # anchored under the pointer
        self._redraw()
        self.set_status(f"Zoom: {self.zoom * 100:.0f}%")

    def _clamp_point(self, point: tuple[float, float], upper: bool = False) -> tuple[float, float]:
        assert self.original_image is not None
        max_x = self.original_image.width if upper else self.original_image.width - 1
        max_y = self.original_image.height if upper else self.original_image.height - 1
        return max(0.0, min(point[0], max_x)), max(0.0, min(point[1], max_y))

    def _point_in_image(self, point: tuple[float, float]) -> bool:
        if self.original_image is None:
            return False
        return 0 <= point[0] < self.original_image.width and 0 <= point[1] < self.original_image.height

    def _roi_at_point(self, point: tuple[float, float]) -> BandROI | None:
        """Return the smallest ROI containing an image-coordinate point."""

        hits = [
            roi for roi in self.rois
            if roi.x <= point[0] <= roi.x2 and roi.y <= point[1] <= roi.y2
        ]
        return min(hits, key=lambda roi: roi.width * roi.height, default=None)

    def _redraw(self) -> None:
        scene = self.view.scene()
        scene.clear()
        self.preview_item = None
        self._roi_items = {}
        if self.original_image is None:
            text = scene.addSimpleText("Open a Western blot image to begin analysis.")
            font = QFont()
            font.setPointSize(16)
            text.setFont(font)
            text.setBrush(QBrush(QColor("#c8c8c8")))
            scene.setSceneRect(text.boundingRect().adjusted(-20, -20, 20, 20))
            self.view.resetTransform()
            self._update_profile_plot()
            return
        state = self.image_states[self.active_image_key]
        width, height = self.original_image.width, self.original_image.height
        pixmap_item = scene.addPixmap(self._display_pixmap(state))
        # Smooth when shrunk; nearest-neighbour when enlarged, so pixels stay honest.
        pixmap_item.setTransformationMode(
            Qt.TransformationMode.SmoothTransformation if self.zoom < 1
            else Qt.TransformationMode.FastTransformation
        )
        margin = 4 / max(self.zoom, 1e-6)
        scene.setSceneRect(QRectF(-margin, -margin, width + 2 * margin, height + 2 * margin))
        if self.grid_check.isChecked():
            self._draw_grid(width, height)
        if self.background_mode == "profile":
            # Show the region actually measured (profile minima) as a dashed box.
            for key, measurement in self.combined_measurements:
                if key != self.active_image_key or measurement.measure_x1 is None:
                    continue
                manual = measurement.edges_source.startswith("manual")
                scene.addRect(
                    QRectF(measurement.measure_x1, measurement.measure_y1,
                           measurement.measure_x2 - measurement.measure_x1,
                           measurement.measure_y2 - measurement.measure_y1),
                    cosmetic_pen(EDGE_COLOR if manual else HIGHLIGHT_COLOR, 1, (4, 3)),
                )
        for roi in self.rois:
            selected = roi.uid == self.selected_uid
            color = ROI_SELECTED_COLOR if selected else ROI_COLOR
            rect_item = scene.addRect(
                QRectF(roi.x, roi.y, roi.width, roi.height), cosmetic_pen(color, 3 if selected else 2),
            )
            label = RoiLabelItem(str(roi.uid), color)
            label.setPos(roi.x, roi.y)
            scene.addItem(label)
            if selected:
                rect_item.setZValue(2)
                label.setZValue(2)
            self._roi_items[roi.uid] = (rect_item, label)
        self._update_profile_plot()

    def _draw_grid(self, width: int, height: int) -> None:
        """Overlay a grid every N image pixels (display only)."""

        spacing = int(self.grid_spin.value())
        if spacing < 1 or self.zoom <= 0:
            return
        # Keep lines at least ~8 screen px apart at low zoom by skipping some.
        step = spacing * max(1, int(np.ceil(8 / (spacing * self.zoom))))
        pen = cosmetic_pen(HIGHLIGHT_COLOR, 1, (3, 3))
        scene = self.view.scene()
        for x in range(step, width, step):
            scene.addLine(x, 0, x, height, pen)
        for y in range(step, height, step):
            scene.addLine(0, y, width, y, pen)

    # -- moving / resizing the selected box --------------------------------

    def _handle_at(self, view_pos: QPointF) -> str | None:
        """Which part of the selected box is under the pointer (view px).

        Returns an edge/corner code ("l", "rt", ...), "move" for the inside,
        or None. Codes use l/r for left/right and t/b for top/bottom.
        """

        roi = next((r for r in self.rois if r.uid == self.selected_uid), None)
        if roi is None:
            return None
        top_left = self.view.mapFromScene(QPointF(roi.x, roi.y))
        bottom_right = self.view.mapFromScene(QPointF(roi.x2, roi.y2))
        x, y = view_pos.x(), view_pos.y()
        x1, y1, x2, y2 = top_left.x(), top_left.y(), bottom_right.x(), bottom_right.y()
        tol = HANDLE_TOLERANCE
        if not (x1 - tol <= x <= x2 + tol and y1 - tol <= y <= y2 + tol):
            return None
        horizontal = ""
        if abs(x - x1) <= tol or abs(x - x2) <= tol:
            horizontal = "l" if abs(x - x1) <= abs(x - x2) else "r"
        vertical = ""
        if abs(y - y1) <= tol or abs(y - y2) <= tol:
            vertical = "t" if abs(y - y1) <= abs(y - y2) else "b"
        code = horizontal + vertical
        if code:
            return code
        return "move" if (x1 < x < x2 and y1 < y < y2) else None

    _HANDLE_CURSORS = {
        "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
        "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
        "lt": Qt.CursorShape.SizeFDiagCursor, "rb": Qt.CursorShape.SizeFDiagCursor,
        "rt": Qt.CursorShape.SizeBDiagCursor, "lb": Qt.CursorShape.SizeBDiagCursor,
        "move": Qt.CursorShape.SizeAllCursor,
    }

    def _canvas_hover(self, view_pos: QPointF, modifiers) -> None:
        handle = None
        if self.original_image is not None and not modifiers & Qt.KeyboardModifier.ShiftModifier:
            handle = self._handle_at(view_pos)
        self.view.viewport().setCursor(self._HANDLE_CURSORS.get(handle, Qt.CursorShape.CrossCursor))

    def _edit_rect(self, point: tuple[float, float]) -> tuple[float, float, float, float]:
        """New box (x1, y1, x2, y2) for the move/resize in progress, kept inside the image."""

        edit = self._edit
        w, h = self.original_image.width, self.original_image.height
        x1, y1, x2, y2 = edit["rect"]
        dx, dy = point[0] - edit["start"][0], point[1] - edit["start"][1]
        handle = edit["handle"]
        if handle == "move":
            dx = max(-x1, min(dx, w - x2))
            dy = max(-y1, min(dy, h - y2))
            return x1 + dx, y1 + dy, x2 + dx, y2 + dy
        if "l" in handle:
            x1 = max(0.0, min(x1 + dx, x2 - 2))
        if "r" in handle:
            x2 = min(float(w), max(x2 + dx, x1 + 2))
        if "t" in handle:
            y1 = max(0.0, min(y1 + dy, y2 - 2))
        if "b" in handle:
            y2 = min(float(h), max(y2 + dy, y1 + 2))
        return x1, y1, x2, y2

    def _finish_edit(self, point: tuple[float, float]) -> None:
        edit = self._edit
        x1, y1, x2, y2 = (int(round(v)) for v in self._edit_rect(point))
        self._edit = None
        roi = next((r for r in self.rois if r.uid == edit["uid"]), None)
        if roi is None or (x1, y1, x2, y2) == (roi.x, roi.y, roi.x2, roi.y2):
            self._redraw()
            return
        source = roi.source if roi.source.endswith("(edited)") else f"{roi.source} (edited)"
        edited = clamp_roi(replace(roi, x=x1, y=y1, width=x2 - x1, height=y2 - y1, source=source),
                           self.gray.shape[:2])
        self.rois = assign_lanes([edited if r.uid == roi.uid else r for r in self.rois])
        # Hand-set profile edges were placed for the old box.
        self.image_states[self.active_image_key].edge_overrides.pop(roi.uid, None)
        verb = "moved" if edit["handle"] == "move" else "resized"
        log.info("ROI %s on %s image %s to x=%d y=%d w=%d h=%d", roi.uid, self.active_image_key, verb,
                 edited.x, edited.y, edited.width, edited.height)
        self._update_results()
        self._redraw()
        self.set_status(f"ID {roi.uid} {verb}: x {edited.x}, y {edited.y}, {edited.width} × {edited.height} px.")

    # -- press / drag / release ---------------------------------------------

    def _canvas_press(self, scene_point: QPointF, view_pos: QPointF, modifiers) -> None:
        if self.original_image is None:
            return
        p = (scene_point.x(), scene_point.y())
        # Grabbing the selected box's edge or inside moves/resizes it;
        # Shift draws a new box even from inside the selected one.
        if not modifiers & Qt.KeyboardModifier.ShiftModifier:
            handle = self._handle_at(view_pos)
            if handle is not None:
                roi = next(r for r in self.rois if r.uid == self.selected_uid)
                self._edit = {"uid": roi.uid, "handle": handle, "start": p,
                              "rect": (float(roi.x), float(roi.y), float(roi.x2), float(roi.y2))}
                return
        if not self._point_in_image(p):
            return
        hit = self._roi_at_point(p)
        if hit is not None:
            self._select_uid(hit.uid)
            self.set_status(f"Selected ID {hit.uid}: {hit.name}")
        self.drag_start = p
        self.preview_item = self.view.scene().addRect(
            QRectF(scene_point, scene_point), cosmetic_pen(HIGHLIGHT_COLOR, 2, (5, 3)),
        )
        self.preview_item.setZValue(5)

    def _canvas_drag(self, scene_point: QPointF) -> None:
        p = (scene_point.x(), scene_point.y())
        if self._edit is not None:
            x1, y1, x2, y2 = self._edit_rect(p)
            items = self._roi_items.get(self._edit["uid"])
            if items is not None:
                items[0].setRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                items[1].setPos(x1, y1)
            return
        if self.drag_start is None or self.preview_item is None:
            return
        p = self._clamp_point(p)
        x1, x2 = sorted((self.drag_start[0], p[0]))
        y1, y2 = sorted((self.drag_start[1], p[1]))
        self.preview_item.setRect(QRectF(x1, y1, x2 - x1, y2 - y1))

    def _canvas_release(self, scene_point: QPointF) -> None:
        if self.original_image is None:
            return
        if self._edit is not None:
            self._finish_edit((scene_point.x(), scene_point.y()))
            return
        if self.drag_start is None:
            return
        end = self._clamp_point((scene_point.x(), scene_point.y()))
        start = self.drag_start
        self.drag_start = None
        if self.preview_item is not None:
            self.view.scene().removeItem(self.preview_item)
            self.preview_item = None
        if self.region_radio.isChecked():
            x1, x2 = sorted((start[0], end[0]))
            y1, y2 = sorted((start[1], end[1]))
            x1, y1 = self._clamp_point((x1, y1))
            x2, y2 = self._clamp_point((x2, y2), upper=True)
            if x2 - x1 < 3 or y2 - y1 < 3:
                if self._roi_at_point(start) is None:
                    self.set_status("Selected region is too small; please drag a larger box.")
                self._redraw()
                return
            self._detect_region((x1, y1, x2, y2))
            return
        x1, x2 = sorted((start[0], end[0]))
        y1, y2 = sorted((start[1], end[1]))
        x1, y1 = self._clamp_point((x1, y1))
        x2, y2 = self._clamp_point((x2, y2), upper=True)
        if x2 - x1 < 2 or y2 - y1 < 2:
            hits = [r for r in self.rois if r.x <= start[0] <= r.x2 and r.y <= start[1] <= r.y2]
            if hits:
                self._select_uid(min(hits, key=lambda r: r.width * r.height).uid)
                self.set_status("Band selected; drag its edges to resize it, or inside it to move it.")
            else:
                self.set_status("ROI too small; not added.")
            self._redraw()
            return
        roi = BandROI(
            uid=self.next_uid, x=int(round(x1)), y=int(round(y1)),
            width=max(1, int(round(x2 - x1))), height=max(1, int(round(y2 - y1))),
            name=f"Band {self.next_uid}", source="Manual",
        )
        self.rois.append(roi)
        self.rois = assign_lanes(self.rois)
        self.selected_uid = self.next_uid
        self.next_uid += 1
        self._update_results()
        self._redraw()
        self.set_status(f"Added manual ROI ID {self.selected_uid}")

    def _canvas_context_menu(self, scene_point: QPointF, global_pos) -> None:
        if self.original_image is None:
            return
        roi = self._roi_at_point((scene_point.x(), scene_point.y()))
        if roi is None:
            self.set_status("Right-click an ROI to rename or delete it.")
            return
        self._select_uid(roi.uid)
        self.set_status(f"Selected ID {roi.uid}: {roi.name}")
        menu = QMenu(self)
        menu.addAction("Rename ROI", self.rename_selected)
        menu.addAction("Delete ROI", self.delete_selected)
        menu.exec(global_pos)

    # ----------------------------------------------------------------------
    # Profile plot and draggable band edges
    # ----------------------------------------------------------------------

    def _update_profile_plot(self) -> None:
        """Plot the selected ROI's horizontal and vertical mean-intensity profiles."""

        if not HAVE_MATPLOTLIB:
            return
        roi = None
        if self.gray is not None:
            roi = next((r for r in self.rois if r.uid == self.selected_uid), None)
        # _redraw runs on every zoom/contrast change; only replot when the
        # selected band, its geometry or the underlying image actually changes.
        profile_mode = self.background_mode == "profile"
        state = self.image_states[self.active_image_key]
        overrides = state.edge_overrides.get(roi.uid, {}) if roi is not None else {}
        polarity = self._measurement_polarity(state) if (profile_mode and self.gray is not None) else ""
        signature = (
            self.active_image_key,
            id(self.gray),
            None if roi is None else (roi.uid, roi.name, roi.x, roi.y, roi.width, roi.height),
            profile_mode,
            polarity,
            tuple(sorted(overrides.items())),
        )
        if signature == self._profile_signature:
            return
        self._profile_signature = signature
        self._profile_pb = None

        ax_h, ax_v = self.profile_axes
        for ax in (ax_h, ax_v):
            ax.clear()
            ax.tick_params(labelsize=7)
        ax_h.set_xlabel("x (px)", fontsize=7)
        ax_v.set_xlabel("y (px)", fontsize=7)
        ax_h.set_ylabel("Mean intensity", fontsize=7)
        self.profile_figure.suptitle("")
        if roi is None:
            ax_h.set_title("Horizontal (column means)", fontsize=8)
            ax_v.set_title("Vertical (row means)", fontsize=8)
            for ax in (ax_h, ax_v):
                ax.text(0.5, 0.5, "Select a band", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="#888")
        elif profile_mode and polarity in ("dark", "light"):
            pb = profile_baseline(self.gray, roi, polarity, overrides)
            self._profile_pb = pb
            # Horizontal: lane profile with the measured columns shaded.
            ax_h.plot(pb.x_positions, pb.horizontal, color=BASELINE_COLOR, linewidth=1.2)
            ax_h.axvspan(pb.left - 0.5, pb.right + 0.5, color=BASELINE_COLOR, alpha=0.10, linewidth=0)
            # Vertical: profile, straight baseline between the minima, and the
            # integrated area (signal above the baseline) shaded.
            ax_v.plot(pb.y_positions, pb.vertical, color=BASELINE_COLOR, linewidth=1.2)
            rows = np.arange(pb.top, pb.bottom + 1)
            baseline = np.interp(rows, [pb.top, pb.bottom], [pb.baseline_top, pb.baseline_bottom])
            segment = pb.vertical[pb.top - pb.y_positions[0] : pb.bottom - pb.y_positions[0] + 1]
            ax_v.fill_between(rows, baseline, segment, color=BASELINE_COLOR, alpha=0.25, linewidth=0)
            ax_v.plot([pb.top, pb.bottom], [pb.baseline_top, pb.baseline_bottom], color=EDGE_COLOR, linewidth=1.2)
            ax_v.plot([pb.top, pb.bottom], [pb.baseline_top, pb.baseline_bottom], "o", color=EDGE_COLOR, markersize=6)
            for ax, keys in ((ax_h, ("left", "right")), (ax_v, ("top", "bottom"))):
                for key in keys:
                    manual = key in pb.manual
                    ax.axvline(getattr(pb, key), color=EDGE_COLOR, linewidth=1.6 if manual else 1.0,
                               linestyle="-" if manual else "--")
            ax_h.set_title("x edges (drag)", fontsize=8)
            ax_v.set_title("y edges + baseline (drag)", fontsize=8)
            edges_text = "manual: " + ", ".join(pb.manual) if pb.manual else "auto edges"
            self.profile_figure.suptitle(
                f"ID {roi.uid} | lane {roi.lane} | IntDen {pb.integrated_density:,.0f} | {edges_text}", fontsize=8,
            )
        else:
            ax_h.set_title("Horizontal (column means)", fontsize=8)
            ax_v.set_title("Vertical (row means)", fontsize=8)
            profiles = intensity_profiles(self.gray, roi)
            ax_h.plot(profiles.x_positions, profiles.horizontal, color=PROFILE_H_COLOR, linewidth=1.2)
            ax_v.plot(profiles.y_positions, profiles.vertical, color=PROFILE_V_COLOR, linewidth=1.2)
            self.profile_figure.suptitle(f"ID {roi.uid} | lane {roi.lane} | {roi.name}", fontsize=8)
        self.profile_canvas.draw_idle()

    def _edge_for_event(self, event) -> str | None:
        """The edge nearest a click: x edges on the left plot, y edges on the right."""

        pb = self._profile_pb
        if pb is None or event.xdata is None:
            return None
        ax_h, ax_v = self.profile_axes
        if event.inaxes is ax_h:
            keys = ("left", "right")
        elif event.inaxes is ax_v:
            keys = ("top", "bottom")
        else:
            return None
        return min(keys, key=lambda key: abs(getattr(pb, key) - event.xdata))

    def _move_edge(self, key: str, position: float) -> None:
        pb = self._profile_pb
        roi = next((r for r in self.rois if r.uid == self.selected_uid), None)
        if pb is None or roi is None or self.gray is None:
            return
        h, w = self.gray.shape[:2]
        value = int(round(position))
        # Keep at least 2 px between opposite edges and stay inside the image.
        if key == "left":
            value = max(0, min(value, pb.right - 2))
        elif key == "right":
            value = min(w - 1, max(value, pb.left + 2))
        elif key == "top":
            value = max(0, min(value, pb.bottom - 2))
        else:
            value = min(h - 1, max(value, pb.top + 2))
        state = self.image_states[self.active_image_key]
        edges = state.edge_overrides.setdefault(roi.uid, {})
        if edges.get(key) == value:
            return
        edges[key] = value
        self._update_profile_plot()

    def _on_profile_press(self, event) -> None:
        if self.background_mode != "profile" or getattr(event, "button", 1) != 1:
            return
        key = self._edge_for_event(event)
        if key is None:
            return
        self._drag_edge = key
        self._move_edge(key, event.xdata)

    def _on_profile_motion(self, event) -> None:
        if self._drag_edge is None or event.xdata is None:
            return
        ax_h, ax_v = self.profile_axes
        expected = ax_h if self._drag_edge in ("left", "right") else ax_v
        if event.inaxes is expected:
            self._move_edge(self._drag_edge, event.xdata)

    def _on_profile_release(self, _event) -> None:
        if self._drag_edge is None:
            return
        key = self._drag_edge
        self._drag_edge = None
        roi_uid = self.selected_uid
        edges = self.image_states[self.active_image_key].edge_overrides.get(roi_uid, {})
        # Re-measure everything (ratios, table, ROI file) once the drag ends.
        self._update_results()
        self._redraw()
        if key in edges:
            log.info("Set %s edge of ROI %s on %s image to %d px", key, roi_uid, self.active_image_key, edges[key])
            self.set_status(f"ID {roi_uid}: {key} edge set to {edges[key]} px. Reset Edges restores automatic edges.")

    def reset_edges(self) -> None:
        """Return the selected band to automatic (profile-minimum) edges."""

        state = self.image_states[self.active_image_key]
        uid = self.selected_uid
        if uid is None or uid not in state.edge_overrides:
            self.set_status("The selected band already uses automatic edges.")
            return
        state.edge_overrides.pop(uid, None)
        log.info("Reset edges of ROI %s on %s image to automatic", uid, self.active_image_key)
        self._update_results()
        self._redraw()
        self.set_status(f"ID {uid}: edges reset to the automatic profile minima.")

    # ----------------------------------------------------------------------
    # Measurement results and pairing
    # ----------------------------------------------------------------------

    def _measurement_polarity(self, state: ImageState) -> str:
        """The polarity actually used for an image (resolves 'auto')."""

        if self.polarity != "auto":
            return self.polarity
        if state.auto_polarity is None and state.gray is not None:
            state.auto_polarity = infer_polarity(state.gray)
        return state.auto_polarity or ""

    def _update_results(self) -> None:
        self._sync_active_state()
        self.combined_measurements = []
        measurements_by_role: dict[str, list[BandMeasurement]] = {"target": [], "reference": []}
        for key, state in self.image_states.items():
            if state.gray is None:
                continue
            # Drop hand-set edges for ROIs that no longer exist.
            live_uids = {roi.uid for roi in state.rois}
            state.edge_overrides = {
                uid: edges for uid, edges in state.edge_overrides.items() if uid in live_uids and edges
            }
            measurements = measure_all(
                state.gray,
                state.rois,
                self._measurement_polarity(state),
                None,
                background_mode=self.background_mode,
                edge_overrides=state.edge_overrides,
            )
            measurements_by_role[key] = measurements
            self.combined_measurements.extend((key, measurement) for measurement in measurements)
            # Every ROI edit (add, split, detect, move, rename, delete, clear)
            # and category rename passes through here, so autosave from here.
            self._save_roi_file(key)
        self.normalized_results = normalize_target_to_reference_by_lane(
            measurements_by_role["target"],
            measurements_by_role["reference"],
        )
        self.pairing_result_by_uid = {}
        for result in self.normalized_results:
            if result.target_uid is not None:
                self.pairing_result_by_uid[result.target_uid] = result
            if result.reference_uid is not None:
                self.pairing_result_by_uid[result.reference_uid] = result
        # Ambiguous lane results intentionally have no selected reference UID.
        # Still attach their diagnostic to the raw reference rows in the main
        # table so every affected band visibly explains why no ratio exists.
        for key, measurement in self.combined_measurements:
            if measurement.uid in self.pairing_result_by_uid:
                continue
            if key == "reference":
                diagnostic = next(
                    (
                        result
                        for result in self.normalized_results
                        if (
                            result.lane == measurement.lane
                            and result.pairing_status in {"ambiguous_target", "ambiguous_reference"}
                        )
                    ),
                    None,
                )
                if diagnostic is not None:
                    self.pairing_result_by_uid[measurement.uid] = diagnostic
        self._refresh_results_table()

    def _refresh_results_table(self) -> None:
        """Rebuild the results table from the current measurements, keeping the selection."""

        self._table_updating = True
        try:
            self.table.clearContents()
            self.table.setRowCount(len(self.combined_measurements))
            self._row_by_uid = {}
            for row, (key, m) in enumerate(self.combined_measurements):
                state = self.image_states[key]
                result = self.pairing_result_by_uid.get(m.uid)

                def density_text(role: str) -> str:
                    value = None if result is None else (
                        result.target_integrated_density if role == "target"
                        else result.reference_integrated_density
                    )
                    if value is not None:
                        return f"{value:.2f}"
                    # Unpaired: show the band's own density in its own column only.
                    return f"{m.integrated_density:.2f}" if key == role else ""

                ratio = (
                    "" if result is None or result.target_over_reference is None
                    else f"{result.target_over_reference:.6f}"
                )
                pairing_status = (
                    "" if result is None
                    else PAIRING_STATUS_LABELS.get(result.pairing_status, result.pairing_status)
                )
                values = (
                    state.label, m.uid, m.name, m.lane, f"{m.mean_gray:.2f}", f"{m.background:.2f}",
                    f"{m.corrected_mean:.2f}", f"{m.integrated_density:.2f}",
                    density_text("target"), density_text("reference"), ratio, pairing_status,
                    "" if result is None else result.missing_reason,
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setData(Qt.ItemDataRole.UserRole, m.uid)
                    self.table.setItem(row, column, item)
                self._row_by_uid[m.uid] = row
        finally:
            self._table_updating = False
        self._select_table_row(self.selected_uid)

    def _select_table_row(self, uid: int | None) -> None:
        self._table_updating = True
        try:
            row = self._row_by_uid.get(uid) if uid is not None else None
            if row is None:
                self.table.clearSelection()
            else:
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, 0))
        finally:
            self._table_updating = False

    def _tree_select(self) -> None:
        if self._table_updating:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.table.item(rows[0].row(), 0)
        if item is None:
            return
        uid = int(item.data(Qt.ItemDataRole.UserRole))
        selected_key = next(
            (key for key, state in self.image_states.items() if any(roi.uid == uid for roi in state.rois)),
            None,
        )
        if selected_key is not None and selected_key != self.active_image_key:
            self._sync_active_state()
            self.image_states[selected_key].selected_uid = uid
            self.activate_image(selected_key)
            return
        self.selected_uid = uid
        self._sync_active_state()
        self._redraw()

    def _select_uid(self, uid: int) -> None:
        self.selected_uid = uid
        self._select_table_row(uid)
        self._redraw()

    def rename_selected(self) -> None:
        roi = next((r for r in self.rois if r.uid == self.selected_uid), None)
        if roi is None:
            return
        name, ok = QInputDialog.getText(self, "Rename Band", "Enter Name:", text=roi.name)
        if not ok or not name.strip():
            return
        roi.name = name.strip()
        self._update_results()
        self._redraw()

    def delete_selected(self) -> None:
        if self.selected_uid is None:
            return
        uid = self.selected_uid
        self.rois = [r for r in self.rois if r.uid != uid]
        if self.reference_uid == uid:
            self.reference_uid = self.rois[0].uid if self.rois else None
        self.selected_uid = self.rois[0].uid if self.rois else None
        self.rois = assign_lanes(self.rois)
        self._update_results()
        self._redraw()
        self.set_status(f"Deleted ROI ID {uid}")

    def clear_rois(self) -> None:
        if not self.rois:
            return
        label = self.image_states[self.active_image_key].label
        answer = QMessageBox.question(self, APP_TITLE, f"Clear all ROIs on the {label} image?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.rois.clear()
        self.reference_uid = None
        self.selected_uid = None
        self._update_results()
        self._redraw()
        self.set_status(f"Cleared all ROIs on the {label} image; the other image's results are unchanged.")

    # ----------------------------------------------------------------------
    # Settings window
    # ----------------------------------------------------------------------

    def open_settings(self) -> None:
        """Measurement and detection settings in their own window (non-modal)."""

        if self.settings_window is not None:
            self.settings_window.show()
            self.settings_window.raise_()
            self.settings_window.activateWindow()
            return
        window = QDialog(self)
        window.setWindowTitle("Measurement & Detection Settings")
        window.setModal(False)
        body = QVBoxLayout(window)

        measurement = QGroupBox("Measurement")
        measurement_form = QFormLayout(measurement)
        background_box = QComboBox()
        background_box.addItems(list(BACKGROUND_LABELS.values()))
        background_box.setCurrentText(BACKGROUND_LABELS[self.background_mode])
        background_box.currentTextChanged.connect(self._background_changed)
        background_box.setMinimumWidth(260)
        background_box.setToolTip(
            "How each band's local background is estimated:\n\n"
            "• Above/below: median of thin strips just above and below the band, in the same lane.\n"
            "• Ring: median of a ring of pixels all around the band.\n"
            "• Profile baseline (min. top/bottom, default): band edges at the minima of the band's x and y "
            "profiles; background is a straight line between the top and bottom minima. "
            "Edges can be dragged in the profile plots."
        )
        measurement_form.addRow("Background", background_box)
        body.addWidget(measurement)

        detection = QGroupBox("Detection Parameters")
        form = QFormLayout(detection)
        polarity = QComboBox()
        polarity.addItems(["light", "dark", "auto"])
        polarity.setCurrentText(self.polarity)
        polarity.currentTextChanged.connect(self._polarity_changed)
        polarity.setToolTip("Light (default): positive bands, bright on a dark background "
                            "(fluorescence, e.g. LI-COR). Dark: negative bands, dark on a light background "
                            "(chemiluminescence film, colorimetric). Auto: decide from each image.")
        form.addRow("Band Polarity", polarity)

        sensitivity_row = QHBoxLayout()
        sensitivity = QSlider(Qt.Orientation.Horizontal)
        sensitivity.setRange(1, 100)
        sensitivity.setValue(int(round(self.sensitivity)))
        sensitivity_label = QLabel(f"{self.sensitivity:.0f}")
        sensitivity_label.setMinimumWidth(28)

        def sensitivity_changed(value: int) -> None:
            self.sensitivity = float(value)
            sensitivity_label.setText(str(value))

        sensitivity.valueChanged.connect(sensitivity_changed)
        sensitivity_row.addWidget(sensitivity, 1)
        sensitivity_row.addWidget(sensitivity_label)
        form.addRow("Sensitivity", sensitivity_row)

        def spin(low: int, high: int, value: int, setter) -> QSpinBox:
            box = QSpinBox()
            box.setRange(low, high)
            box.setValue(value)
            box.valueChanged.connect(setter)
            return box

        form.addRow("Minimum Area (px)", spin(4, 100000, self.min_area, lambda v: setattr(self, "min_area", v)))
        expansion = spin(0, 100, self.region_padding, lambda v: setattr(self, "region_padding", v))
        expansion.setToolTip("Pixels added around each box after detection or region splitting.")
        form.addRow("ROI Expansion (px)", expansion)
        form.addRow("ROI Gap (px)", spin(0, 30, self.roi_gap, lambda v: setattr(self, "roi_gap", v)))
        weak = QCheckBox("Exclude candidates below average corrected ROI signal")
        weak.setChecked(self.filter_below_mean)
        weak.toggled.connect(lambda checked: setattr(self, "filter_below_mean", checked))
        form.addRow(weak)
        marker = QComboBox()
        marker.addItems(["auto", "left", "right", "off"])
        marker.setCurrentText(self.marker_mode)
        marker.currentTextChanged.connect(lambda text: setattr(self, "marker_mode", text))
        marker.setToolTip("Removes the marker/ladder lane from detection. Auto: find it; "
                          "Left / Right: it's the outermost lane on that side; Off: keep every lane.")
        form.addRow("Marker Filter", marker)
        body.addWidget(detection)

        close = QPushButton("Close")
        close.clicked.connect(window.hide)
        body.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)
        self.settings_window = window
        window.adjustSize()
        top_right = self.mapToGlobal(self.rect().topRight())
        window.move(max(0, top_right.x() - window.width() - 40), top_right.y() + 60)
        window.show()

    def _update_settings_summary(self) -> None:
        polarity = {
            "light": "Positive bands", "dark": "Negative bands", "auto": "Auto polarity",
        }.get(self.polarity, self.polarity)
        background = {
            "above_below": "Above/below background", "ring": "Ring background", "profile": "Profile baseline",
        }.get(self.background_mode, self.background_mode)
        self.settings_summary.setText(f"{polarity} · {background}")

    def _polarity_changed(self, value: str) -> None:
        if value == self.polarity:
            return
        self.polarity = value
        self._update_settings_summary()
        log.info("Band polarity set to %s", value)
        self._update_results()
        self._redraw()

    def _background_changed(self, label: str) -> None:
        mode = next((key for key, text in BACKGROUND_LABELS.items() if text == label), "above_below")
        if mode == self.background_mode:
            return
        self.background_mode = mode
        self._update_settings_summary()
        log.info("Background method set to %s", mode)
        self._update_results()
        self._redraw()
        self.set_status(f"Background method: {label}. All bands re-measured.")

    # ----------------------------------------------------------------------
    # Exports
    # ----------------------------------------------------------------------

    def export_csv(self, filename: str | None = None) -> None:
        self._update_results()
        if not self.normalized_results:
            QMessageBox.information(self, APP_TITLE, "There are no quantification results to export.")
            return
        if not filename:
            filename, _filter = QFileDialog.getSaveFileName(
                self, "Export Quantification Results", "WB_target_reference_normalized_by_lane.csv", "CSV (*.csv)",
            )
            if not filename:
                return
            filename = ensure_suffix(filename, ".csv")
        measurement_lookup = {
            (key, measurement.uid): measurement
            for key, measurement in self.combined_measurements
        }
        target_state = self.image_states["target"]
        reference_state = self.image_states["reference"]
        headers = [
            "lane", "target_lane", "reference_lane",
            "target_category", "reference_category",
            "target_image", "reference_image", "polarity",
            "target_uid", "reference_uid", "target_name", "reference_name",
            "target_x", "target_y", "target_width", "target_height",
            "reference_x", "reference_y", "reference_width", "reference_height",
            "target_mean_gray", "target_local_background", "target_background_corrected_mean",
            "reference_mean_gray", "reference_local_background", "reference_background_corrected_mean",
            "target_integrated_density", "reference_integrated_density", "target_over_reference",
            "pairing_status", "missing_reason", "target_source", "reference_source",
            "target_rotation_deg", "reference_rotation_deg",
            "target_polarity_used", "reference_polarity_used",
            "target_background_method", "reference_background_method",
            "target_measured_x1", "target_measured_x2", "target_measured_y1", "target_measured_y2",
            "target_baseline_top", "target_baseline_bottom", "target_edges",
            "reference_measured_x1", "reference_measured_x2", "reference_measured_y1", "reference_measured_y2",
            "reference_baseline_top", "reference_baseline_bottom", "reference_edges",
        ]
        target_polarity = self._measurement_polarity(target_state) if target_state.gray is not None else ""
        reference_polarity = self._measurement_polarity(reference_state) if reference_state.gray is not None else ""

        def number(value: float | int | None) -> str:
            if value is None:
                return ""
            return f"{value:.6f}" if isinstance(value, float) else str(value)

        def profile_columns(measurement) -> list[str]:
            """Measured region (x2/y2 exclusive), baseline ends and edge source; blank outside profile mode."""

            if measurement is None or measurement.measure_x1 is None:
                return [""] * 7
            return [
                number(measurement.measure_x1), number(measurement.measure_x2),
                number(measurement.measure_y1), number(measurement.measure_y2),
                number(measurement.baseline_top), number(measurement.baseline_bottom),
                measurement.edges_source,
            ]

        with open(filename, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for result in self.normalized_results:
                target = measurement_lookup.get(("target", result.target_uid))
                reference = measurement_lookup.get(("reference", result.reference_uid))
                writer.writerow([
                    result.lane,
                    number(result.target_lane),
                    number(result.reference_lane),
                    target_state.label,
                    reference_state.label,
                    target_state.path.name if target_state.path else "",
                    reference_state.path.name if reference_state.path else "",
                    self.polarity,
                    number(result.target_uid),
                    number(result.reference_uid),
                    target.name if target is not None else "",
                    reference.name if reference is not None else "",
                    number(target.x if target is not None else None),
                    number(target.y if target is not None else None),
                    number(target.width if target is not None else None),
                    number(target.height if target is not None else None),
                    number(reference.x if reference is not None else None),
                    number(reference.y if reference is not None else None),
                    number(reference.width if reference is not None else None),
                    number(reference.height if reference is not None else None),
                    number(target.mean_gray if target is not None else None),
                    number(target.background if target is not None else None),
                    number(target.corrected_mean if target is not None else None),
                    number(reference.mean_gray if reference is not None else None),
                    number(reference.background if reference is not None else None),
                    number(reference.corrected_mean if reference is not None else None),
                    number(result.target_integrated_density),
                    number(result.reference_integrated_density),
                    number(result.target_over_reference),
                    result.pairing_status,
                    result.missing_reason,
                    target.source if target is not None else "",
                    reference.source if reference is not None else "",
                    f"{target_state.rotation:.1f}",
                    f"{reference_state.rotation:.1f}",
                    target_polarity if target is not None else "",
                    reference_polarity if reference is not None else "",
                    target.background_method if target is not None else "",
                    reference.background_method if reference is not None else "",
                    *profile_columns(target),
                    *profile_columns(reference),
                ])
        try:
            profile_csv, profile_png = self._export_profiles(Path(filename))
        except Exception as exc:
            log.error("Exported results CSV %s but profile export failed: %s", filename, exc)
            QMessageBox.critical(self, "Profile Export Failed",
                                 f"The results CSV was saved, but the profiles were not:\n{exc}")
            self.set_status(f"CSV exported: {filename} (profile export failed)")
            return
        log.info("Exported results CSV %s (%d rows)", filename, len(self.normalized_results))
        log.info("Exported profiles %s%s", profile_csv, f" and {profile_png}" if profile_png else "")
        exported = [profile_csv.name] + ([profile_png.name] if profile_png else [])
        self.set_status(f"CSV exported: {filename} | profiles: {', '.join(exported)}")

    def _collect_profiles(self):
        """Return (image key, state, profiles) for every ROI in both images."""

        collected = []
        for key in ("reference", "target"):
            state = self.image_states[key]
            if state.gray is None:
                continue
            for roi in sorted(state.rois, key=lambda r: (r.lane, r.y, r.uid)):
                collected.append((key, state, intensity_profiles(state.gray, roi)))
        return collected

    def _export_profiles(self, results_path: Path) -> tuple[Path, Path | None]:
        """Write every ROI's profiles next to the results CSV.

        The CSV is in long format (one row per profile point), which loads
        directly into pandas, R or Prism-style reshaping. The PNG has one row
        of plots per image: horizontal profiles on the left, vertical on the right.
        """

        self._sync_active_state()
        collected = self._collect_profiles()
        csv_path = results_path.with_name(f"{results_path.stem}_profiles.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "image_role", "category", "image", "rotation_deg", "uid", "name", "lane",
                "profile", "averaged_over", "position_px", "offset_in_roi_px", "mean_intensity",
            ])
            for key, state, profiles in collected:
                image_name = state.path.name if state.path else ""
                for axis, averaged_over, positions, values in (
                    ("horizontal", "columns (mean of each column; position is x)",
                     profiles.x_positions, profiles.horizontal),
                    ("vertical", "rows (mean of each row; position is y)",
                     profiles.y_positions, profiles.vertical),
                ):
                    for offset, (position, value) in enumerate(zip(positions, values)):
                        writer.writerow([
                            key, state.label, image_name, f"{state.rotation:.1f}", profiles.uid, profiles.name, profiles.lane,
                            axis, averaged_over, int(position), offset,
                            "" if not np.isfinite(value) else f"{value:.6f}",
                        ])

        if not HAVE_MATPLOTLIB or not collected:
            return csv_path, None
        keys = [key for key in ("reference", "target") if any(item[0] == key for item in collected)]
        figure = Figure(figsize=(12, 4.2 * len(keys)), dpi=150, layout="constrained")
        for row, key in enumerate(keys):
            items = [item for item in collected if item[0] == key]
            state = self.image_states[key]
            ax_h = figure.add_subplot(len(keys), 2, row * 2 + 1)
            ax_v = figure.add_subplot(len(keys), 2, row * 2 + 2)
            for _key, _state, profiles in items:
                label = f"ID {profiles.uid} (lane {profiles.lane})"
                line, = ax_h.plot(profiles.x_positions, profiles.horizontal, linewidth=1.1, label=label)
                ax_v.plot(profiles.y_positions, profiles.vertical, linewidth=1.1, color=line.get_color(), label=label)
            image_name = state.path.name if state.path else ""
            ax_h.set_title(f"{state.label} ({image_name}): horizontal profiles (column means)", fontsize=10)
            ax_v.set_title(f"{state.label} ({image_name}): vertical profiles (row means)", fontsize=10)
            ax_h.set_xlabel("x (px)")
            ax_v.set_xlabel("y (px)")
            ax_h.set_ylabel("Mean intensity")
            ax_v.set_ylabel("Mean intensity")
            ax_v.legend(fontsize=7, ncol=max(1, int(np.ceil(len(items) / 12))), loc="best")
        png_path = results_path.with_name(f"{results_path.stem}_profiles.png")
        figure.savefig(png_path)
        return csv_path, png_path

    def export_annotated(self, filename: str | None = None) -> None:
        if self.original_image is None or not self.rois:
            QMessageBox.information(self, APP_TITLE, "Please open an image and create ROIs first.")
            return
        if not filename:
            role = self.image_states[self.active_image_key].label
            safe_role = role.translate(str.maketrans({"/": "_", "\\": "_", ":": "_"}))
            initial = f"{self.image_path.stem if self.image_path else 'WB'}_{safe_role}_annotated.png"
            filename, chosen = QFileDialog.getSaveFileName(
                self, "Export Annotated Image", initial, "PNG (*.png);;TIFF (*.tif)",
            )
            if not filename:
                return
            filename = ensure_suffix(filename, ".tif" if chosen.startswith("TIFF") else ".png")
        annotated = self.original_image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        line_width = max(2, round(min(annotated.size) / 350))
        for roi in self.rois:
            color = (0, 180, 230)
            draw.rectangle((roi.x, roi.y, roi.x2, roi.y2), outline=color, width=line_width)
            draw.text((roi.x + 2, max(0, roi.y - 13)), f"{roi.uid} {roi.name}", fill=color,
                      stroke_width=1, stroke_fill=(255, 255, 255))
        annotated.save(filename)
        log.info("Exported annotated %s image %s", self.active_image_key, filename)
        self.set_status(f"Annotated image exported: {filename}")

    # ----------------------------------------------------------------------
    # Log viewer and shutdown
    # ----------------------------------------------------------------------

    def open_log_viewer(self) -> None:
        """Show the log (opened files, ROI files, rotations, exports) in a window."""

        if self.log_path is None:
            QMessageBox.information(self, APP_TITLE,
                                    "Logging is unavailable: no writable folder was found for the log file.")
            return
        if self.log_window is None:
            window = QDialog(self)
            window.setWindowTitle("Activity Log")
            window.resize(980, 480)
            layout = QVBoxLayout(window)
            top = QHBoxLayout()
            path_label = QLabel(str(self.log_path))
            path_label.setStyleSheet("color: #555;")
            top.addWidget(path_label, 1)
            refresh = QPushButton("Refresh")
            refresh.clicked.connect(self._refresh_log_viewer)
            top.addWidget(refresh)
            layout.addLayout(top)
            self.log_text = QPlainTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            mono = QFont("Courier")
            mono.setStyleHint(QFont.StyleHint.Monospace)
            self.log_text.setFont(mono)
            layout.addWidget(self.log_text, 1)
            self.log_window = window
        self._refresh_log_viewer()
        self.log_window.show()
        self.log_window.raise_()

    def _refresh_log_viewer(self) -> None:
        for handler in log.handlers:
            handler.flush()
        try:
            content = self.log_path.read_text(encoding="utf-8")
        except OSError as exc:
            content = f"Could not read the log file: {exc}"
        self.log_text.setPlainText(content)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        log.info("Session ended")
        super().closeEvent(event)


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    window = WBAnalyzerApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
