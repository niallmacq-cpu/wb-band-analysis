"""Tkinter desktop application for Western blot band quantification.

The analysis itself (detection, backgrounds, profiles, lane normalisation)
lives in ``wb_analysis``; this module is the interface around it: image
loading and display, ROI editing, pairing, exports, logging, and per-image
sidecar files that restore ROIs when an image is reopened.
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
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tifffile

# matplotlib is optional: without it the app still runs and still exports the
# profile CSV, but the live profile plots and the exported PNG are skipped.
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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


def map_float_for_display(data: np.ndarray, black: float, white: float, mode: str) -> Image.Image:
    """Full-precision 0..1 data -> 8-bit greyscale, quantising only at the end."""

    grey = display_curve(np.nan_to_num(data, nan=0.0), black, white, mode)
    return Image.fromarray(np.rint(grey * 255).astype(np.uint8))


def map_image_for_display(image: Image.Image, black: float, white: float, mode: str) -> Image.Image:
    """8-bit greyscale/colour image -> display, via a lookup table (alpha kept)."""

    lut = [int(v) for v in np.rint(display_curve(np.arange(256) / 255.0, black, white, mode) * 255)]
    if image.mode == "RGBA":
        r, g, b, a = image.split()
        return Image.merge("RGBA", (r.point(lut), g.point(lut), b.point(lut), a))
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")
    return image.point(lut * len(image.getbands()))


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
    preview_image: ImageTk.PhotoImage | None = None
    preview_size: tuple[int, int] | None = None
    # Resized (but not contrast-adjusted) preview, cached so the black/white
    # sliders only need a cheap mapping pass instead of a full LANCZOS resize.
    preview_base: Image.Image | None = None
    preview_display_key: tuple | None = None
    # Full-precision display source (float32 scaled 0..1) for high-bit-depth
    # images. When present, contrast is applied before 8-bit quantisation,
    # so brightening reveals real detail instead of stretching 8-bit steps.
    display_data: np.ndarray | None = None
    preview_base_float: np.ndarray | None = None
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
# Widgets
# ---------------------------------------------------------------------------


class ToolTip:
    """Hover hint: shows ``text`` after a short delay while the pointer is over ``widget``."""

    def __init__(self, widget: tk.Misc, text: str, delay_ms: int = 450, wraplength: int = 380) -> None:
        self.widget, self.text, self.delay_ms, self.wraplength = widget, text, delay_ms, wraplength
        self._job: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._job = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._job is not None:
            self.widget.after_cancel(self._job)
            self._job = None

    def _show(self) -> None:
        self._job = None
        if self._tip is not None or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tip, text=self.text, justify="left", wraplength=self.wraplength,
            background="#fffbe6", foreground="#1a1a19", relief="solid", borderwidth=1,
            padx=8, pady=5, font=("TkDefaultFont", 10),
        ).pack()
        self._tip = tip

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


# ---------------------------------------------------------------------------
# Application window
# ---------------------------------------------------------------------------


class WBAnalyzerApp(tk.Tk):
    """Main window: zoomable blot canvas on the left; profiles and results on the right.

    Two ``ImageState`` objects (reference and target) hold everything per
    image; the attributes set in ``__init__`` mirror whichever is active.
    """

    # ----------------------------------------------------------------------
    # Window construction
    # ----------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        minimum_width = min(1050, max(760, screen_width - 40))
        minimum_height = min(680, max(540, screen_height - 80))
        initial_width = max(minimum_width, min(1480, screen_width - 80))
        initial_height = max(minimum_height, min(850, screen_height - 100))
        self.geometry(f"{initial_width}x{initial_height}")
        self.minsize(minimum_width, minimum_height)

        # Working copies of the active image's state; _sync_active_state
        # writes them back into image_states before anything else reads it.
        self.original_image: Image.Image | None = None
        self.gray: np.ndarray | None = None
        self.image_path: Path | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
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

        # Canvas view state.
        self.zoom = 1.0
        self.image_origin = (0.0, 0.0)
        self.drag_start: tuple[float, float] | None = None
        self.preview_item: int | None = None

        # Tk control variables (toolbar, settings window, display bars).
        self.mode_var = tk.StringVar(value="region")
        self.active_image_var = tk.StringVar(value="reference")
        self._name_jobs: dict[str, str] = {}
        self.category_name_vars = {
            "reference": tk.StringVar(value=self.image_states["reference"].label),
            "target": tk.StringVar(value=self.image_states["target"].label),
        }
        # Defaults: positive bands (bright on dark, e.g. LI-COR fluorescence)
        # measured with the profile baseline.
        self.polarity_var = tk.StringVar(value="light")
        self.background_mode = "profile"
        self.background_label_var = tk.StringVar(value=BACKGROUND_LABELS["profile"])
        self.settings_window: tk.Toplevel | None = None
        self.sensitivity_var = tk.DoubleVar(value=55)
        self.min_area_var = tk.IntVar(value=30)
        self.region_padding_var = tk.IntVar(value=6)
        self.roi_gap_var = tk.IntVar(value=0)
        self.filter_below_mean_var = tk.BooleanVar(value=True)
        self.marker_mode_var = tk.StringVar(value="auto")
        self.black_var = tk.DoubleVar(value=0.0)
        self.white_var = tk.DoubleVar(value=LEVEL_STEPS)
        self.display_mode_var = tk.StringVar(value="Linear")
        self._syncing_display = False
        self.rotation_var = tk.DoubleVar(value=0.0)
        self._syncing_rotation = False
        self._rotation_job: str | None = None
        self.show_grid_var = tk.BooleanVar(value=False)
        self.grid_spacing_var = tk.IntVar(value=50)
        self.status_var = tk.StringVar(value="Load a control image and a protein image to begin.")
        self.log_path = setup_logging()
        self.log_window: tk.Toplevel | None = None
        log.info("Session started")
        self._auto_fit = True
        self._build_ui()
        self._bind_events()
        self.after(50, self._place_sashes)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.option_add("*Font", ("Arial", 12))
        self._build_toolbar()

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self.main_pane = main
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 6))

        image_frame = ttk.Frame(main)
        control_frame = ttk.Frame(main, width=410)
        main.add(image_frame, weight=4)
        main.add(control_frame, weight=2)
        self._build_image_panel(image_frame)
        self._build_control_panel(control_frame)

        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=(8, 4)).pack(fill=tk.X, side=tk.BOTTOM)

        self.roi_context_menu = tk.Menu(self, tearoff=False)
        self.roi_context_menu.add_command(label="Rename ROI", command=self.rename_selected)
        self.roi_context_menu.add_command(label="Delete ROI", command=self.delete_selected)

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self, padding=(8, 7))
        toolbar.pack(fill=tk.X)
        # One group per image: select it for editing, name it, load it.
        for value in ("reference", "target"):
            role = IMAGE_ROLE_NAMES[value]
            ttk.Radiobutton(
                toolbar,
                text=role,
                value=value,
                variable=self.active_image_var,
                command=lambda key=value: self.activate_image(key),
            ).pack(side=tk.LEFT, padx=(0, 3))
            name_entry = ttk.Entry(toolbar, width=10, textvariable=self.category_name_vars[value])
            name_entry.pack(side=tk.LEFT, padx=(0, 3))
            # Names apply as you type (after a short pause). Leaving the box,
            # or pressing Enter, puts the default back if it was left empty.
            self.category_name_vars[value].trace_add(
                "write", lambda *_args, key=value: self._category_name_typed(key),
            )
            name_entry.bind("<Return>", lambda _event, key=value: self.apply_category_name(key, finished=True))
            name_entry.bind("<FocusOut>", lambda _event, key=value: self.apply_category_name(key, finished=True))
            ToolTip(name_entry, "Name used for this image in the table, the CSV and exported file names. "
                    "Updates as you type.")
            load = ttk.Button(toolbar, text="Load…", width=6, command=lambda key=value: self.open_image(key))
            load.pack(side=tk.LEFT, padx=(0, 2))
            ToolTip(load, f"Open the {role.lower()} image.")
            ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=7)
        ttk.Radiobutton(
            toolbar,
            text="Split Selected Region",
            value="region",
            variable=self.mode_var,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(
            toolbar,
            text="Single Band ROI",
            value="rect",
            variable=self.mode_var,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Delete ROI", command=self.delete_selected).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(toolbar, text="Clear All", command=self.clear_rois).pack(side=tk.LEFT, padx=2)

        ttk.Button(toolbar, text="View Log", command=self.open_log_viewer).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="Export CSV", command=self.export_csv).pack(side=tk.RIGHT, padx=2)
        ttk.Button(toolbar, text="Export Image", command=self.export_annotated).pack(side=tk.RIGHT, padx=2)

    def _build_image_panel(self, image_frame: ttk.Frame) -> None:
        """Left side: zoomable canvas, then the display and rotation bars."""

        self.canvas = tk.Canvas(image_frame, background=CANVAS_BACKGROUND, highlightthickness=0, cursor="crosshair")
        xscroll = ttk.Scrollbar(image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        yscroll = ttk.Scrollbar(image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        display_bar = ttk.Frame(image_frame, padding=(0, 5, 0, 0))
        display_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        black_label = ttk.Label(display_bar, text="Black")
        black_label.pack(side=tk.LEFT)
        black_scale = ttk.Scale(
            display_bar, from_=0, to=LEVEL_STEPS, variable=self.black_var,
            command=lambda _value: self._display_levels_changed("black"),
        )
        black_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 4))
        self.black_value_label = ttk.Label(display_bar, text="", width=7, anchor="w")
        self.black_value_label.pack(side=tk.LEFT)
        white_label = ttk.Label(display_bar, text="White")
        white_label.pack(side=tk.LEFT, padx=(6, 0))
        white_scale = ttk.Scale(
            display_bar, from_=0, to=LEVEL_STEPS, variable=self.white_var,
            command=lambda _value: self._display_levels_changed("white"),
        )
        white_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 4))
        self.white_value_label = ttk.Label(display_bar, text="", width=7, anchor="w")
        self.white_value_label.pack(side=tk.LEFT)
        mode_box = ttk.Combobox(
            display_bar, state="readonly", width=7, textvariable=self.display_mode_var,
            values=list(DISPLAY_MODES),
        )
        mode_box.pack(side=tk.LEFT, padx=(6, 0))
        mode_box.bind("<<ComboboxSelected>>", lambda _event: self._display_mode_changed())
        auto_button = ttk.Button(display_bar, text="Auto", command=self.auto_contrast)
        auto_button.pack(side=tk.LEFT, padx=(6, 0))
        levels_hint = ("Display only: pixel values at or below Black show as black, at or above White as "
                       "white. Detection and measurements always use the original data.")
        for widget in (black_label, black_scale, white_label, white_scale):
            ToolTip(widget, levels_hint)
        ToolTip(mode_box, "Linear: grey level proportional to intensity (twice as bright looks twice as "
                "bright). Log: lifts faint bands and compresses bright ones (e.g. a saturated ladder); "
                "differences no longer look proportional. Display only.")
        ToolTip(auto_button, "Set Black and White to the 0.5th and 99.5th percentiles of this image, so a "
                "few very bright pixels don't leave the bands dark. Applied automatically when an image is opened.")
        fit_button = ttk.Button(display_bar, text="Fit", command=self._fit_image)
        fit_button.pack(side=tk.LEFT, padx=(10, 0))
        ToolTip(fit_button, "Fit the blot to the view. It stays fitted as you resize panes until you "
                "zoom with the mouse wheel.")

        geometry_bar = ttk.Frame(image_frame, padding=(0, 4, 0, 0))
        geometry_bar.grid(row=3, column=0, columnspan=2, sticky="ew")
        ttk.Label(geometry_bar, text="Rotate (this image)").pack(side=tk.LEFT)
        ttk.Scale(
            geometry_bar,
            from_=-ROTATION_LIMIT,
            to=ROTATION_LIMIT,
            variable=self.rotation_var,
            command=lambda _value: self._rotation_changed(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 5))
        rotation_entry = ttk.Spinbox(
            geometry_bar,
            from_=-ROTATION_LIMIT,
            to=ROTATION_LIMIT,
            increment=0.1,
            width=6,
            format="%.1f",
            textvariable=self.rotation_var,
            command=self._rotation_changed,
        )
        rotation_entry.pack(side=tk.LEFT)
        rotation_entry.bind("<Return>", lambda _e: self._rotation_changed())
        rotation_entry.bind("<FocusOut>", lambda _e: self._rotation_changed())
        ttk.Label(geometry_bar, text="°").pack(side=tk.LEFT)
        ttk.Button(geometry_bar, text="Reset", command=self.reset_rotation).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Separator(geometry_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Checkbutton(
            geometry_bar, text="Show grid", variable=self.show_grid_var, command=self._redraw,
        ).pack(side=tk.LEFT)
        ttk.Label(geometry_bar, text="Spacing").pack(side=tk.LEFT, padx=(8, 2))
        grid_spacing = ttk.Spinbox(
            geometry_bar, from_=5, to=2000, increment=5, width=5,
            textvariable=self.grid_spacing_var, command=self._redraw,
        )
        grid_spacing.pack(side=tk.LEFT)
        grid_spacing.bind("<Return>", lambda _e: self._redraw())
        grid_spacing.bind("<FocusOut>", lambda _e: self._redraw())
        ttk.Label(geometry_bar, text="px").pack(side=tk.LEFT, padx=(2, 0))
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)

    def _build_control_panel(self, control_frame: ttk.Frame) -> None:
        """Right side: action buttons, settings summary, profile plots, results table."""

        # Right-hand panel: settings and profile plots on top (scrolling if
        # they don't fit), results table below; drag the divider to resize.
        control_pane = ttk.Panedwindow(control_frame, orient=tk.VERTICAL)
        self.control_pane = control_pane
        control_pane.pack(fill=tk.BOTH, expand=True)
        profile_pane = ttk.Frame(control_pane)
        lower = ttk.Frame(control_pane)
        control_pane.add(profile_pane, weight=1)
        control_pane.add(lower, weight=1)

        settings_bar = ttk.Frame(control_frame)
        settings_bar.pack(fill=tk.X, pady=(0, 2), before=control_pane)
        ttk.Button(settings_bar, text="Settings...", command=self.open_settings).pack(side=tk.LEFT)
        self.settings_summary = ttk.Label(settings_bar, foreground="#555")
        self.settings_summary.pack(side=tk.LEFT, padx=(8, 0), fill=tk.X, expand=True)
        ToolTip(self.settings_summary, "Current band polarity and background method. "
                "Change them, and the detection parameters, in Settings.")
        self.polarity_var.trace_add("write", lambda *_: self._update_settings_summary())
        self._update_settings_summary()

        profile_frame = ttk.LabelFrame(profile_pane, text="Intensity Profiles (Selected Band)", padding=5)
        profile_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 4))
        self._profile_signature: tuple | None = None
        if HAVE_MATPLOTLIB:
            self.profile_figure = Figure(figsize=(4.4, 2.1), dpi=90, layout="constrained")
            self.profile_axes = (
                self.profile_figure.add_subplot(1, 2, 1),
                self.profile_figure.add_subplot(1, 2, 2),
            )
            # Reset Edges is packed first (at the bottom) so the plot, which
            # grows with its pane, can never squeeze it out.
            reset_edges = ttk.Button(profile_frame, text="Reset Edges", command=self.reset_edges)
            reset_edges.pack(side=tk.BOTTOM, anchor="e", pady=(3, 0))
            ToolTip(reset_edges, "Return the selected band to automatic (profile-minimum) edges.")
            self.profile_canvas = FigureCanvasTkAgg(self.profile_figure, master=profile_frame)
            self.profile_canvas.get_tk_widget().configure(height=160)
            self.profile_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._profile_pb = None
            self._drag_edge: str | None = None
            self.profile_canvas.mpl_connect("button_press_event", self._on_profile_press)
            self.profile_canvas.mpl_connect("motion_notify_event", self._on_profile_motion)
            self.profile_canvas.mpl_connect("button_release_event", self._on_profile_release)
            edge_hint = (
                "Profile baseline mode: click or drag in the left plot to move the left/right edge, "
                "in the right plot to move the top/bottom edge. Dashed = automatic, solid = set by hand. "
                "The shaded area is the band signal counted."
            )
            ToolTip(self.profile_canvas.get_tk_widget(), edge_hint)
        else:
            ttk.Label(
                profile_frame,
                text="Install matplotlib to see profile plots (pip install matplotlib). "
                "Profiles are still exported with the CSV.",
                wraplength=380,
                foreground="#555",
            ).pack(fill=tk.X)

        table_heading = ttk.Label(lower, text="Quantification Results (Protein/Control by Lane)  ⓘ")
        table_frame = ttk.LabelFrame(lower, labelwidget=table_heading, padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True)
        columns = (
            "role", "id", "name", "lane", "mean", "bg", "corrected", "integrated",
            "target_integrated_density", "reference_integrated_density",
            "target_over_reference", "pairing_status", "missing_reason",
        )
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        headings = {
            "role": "Image", "id": "ID", "name": "Name", "lane": "Lane", "mean": "Mean Intensity",
            "bg": "Local Background", "corrected": "Corrected Mean", "integrated": "Band Integrated Density",
            "target_integrated_density": "Protein Integrated Density",
            "reference_integrated_density": "Control Integrated Density",
            "target_over_reference": "Protein/Control",
            "pairing_status": "Pairing Status",
            "missing_reason": "Issue / Notes",
        }
        widths = {
            "role": 80, "id": 42, "name": 88, "lane": 48, "mean": 76, "bg": 76,
            "corrected": 76, "integrated": 100, "target_integrated_density": 110,
            "reference_integrated_density": 110, "target_over_reference": 90,
            "pairing_status": 90, "missing_reason": 240,
        }
        for col in columns:
            self.tree.heading(col, text=headings[col])
            self.tree.column(col, width=widths[col], minwidth=38, anchor="center")
        table_scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        table_scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=table_scroll_y.set, xscrollcommand=table_scroll_x.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        table_scroll_y.grid(row=0, column=1, sticky="ns")
        table_scroll_x.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        help_text = (
            "Bands are paired by lane: a band is compared with the control band in the same lane "
            "when each image has exactly one band in that lane. Duplicate or missing lanes are "
            "flagged in Issue / Notes and never silently calculated."
        )
        ToolTip(table_heading, help_text)

    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>", lambda _e: self._canvas_resized())
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<Button-2>", self._canvas_context_menu)
        self.canvas.bind("<Button-3>", self._canvas_context_menu)
        self.canvas.bind("<Control-Button-1>", self._canvas_context_menu)
        self.canvas.bind("<MouseWheel>", self._mousewheel)
        self.tree.bind("<<TreeviewSelect>>", self._tree_select)
        self.tree.bind("<Double-1>", lambda _e: self.rename_selected())
        self.bind("<Delete>", lambda _e: self.delete_selected())
        self.bind("<BackSpace>", lambda _e: self.delete_selected())
        self.bind("<Command-o>", lambda _e: self.open_image(self.active_image_key))
        self.bind("<Command-s>", lambda _e: self.export_csv())
        self.bind("<Control-o>", lambda _e: self.open_image(self.active_image_key))
        self.bind("<Control-s>", lambda _e: self.export_csv())

    def _place_sashes(self, attempt: int = 0) -> None:
        """Set the starting split once the window has its real size.

        Panes otherwise start at their requested sizes, and the results table
        asks for its full column width, which squeezes the image.
        """

        self.update_idletasks()
        width = self.main_pane.winfo_width()
        height = self.control_pane.winfo_height()
        if (width < 300 or height < 200) and attempt < 30:
            self.after(100, lambda: self._place_sashes(attempt + 1))
            return
        self.main_pane.sashpos(0, int(width * 0.66))       # image gets two thirds of the width
        self.update_idletasks()
        height = self.control_pane.winfo_height()
        self.control_pane.sashpos(0, int(height * 0.45))   # profile plots above, table below
        if self._auto_fit:
            self._fit_image()


    # ----------------------------------------------------------------------
    # Image loading and switching
    # ----------------------------------------------------------------------

    def open_image(self, key: str) -> None:
        label = self.image_states[key].label
        filename = filedialog.askopenfilename(
            title=f"Select {label} Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All Files", "*.*")],
        )
        if not filename:
            return
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
            messagebox.showerror("Unable to Open Image", str(exc))
            return
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
        state.preview_image = None
        state.preview_size = None
        state.preview_base = None
        state.preview_display_key = None
        state.display_data = display_data
        state.preview_base_float = None
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
        if self._rotation_job is not None:
            self.after_cancel(self._rotation_job)
            self._rotation_job = None
        log.info(
            "Opened %s image (%s): %s | %d x %d px | %s | sha256 %s",
            key, label, state.path, image.width, image.height, file_format, sha256 or "unavailable",
        )
        roi_note = self._load_roi_file(key)
        self._set_rotation_control(state.rotation)
        self.active_image_key = key
        self.active_image_var.set(key)
        self.original_image = state.image
        self.gray = state.gray
        self.image_path = state.path
        self.rois = state.rois
        self.reference_uid = None
        self.selected_uid = None
        self.after(50, self._fit_image)
        self._update_results()
        self.status_var.set(
            f"Loaded {label} image: {self.image_path.name} | {image.width} × {image.height} px | {roi_note}"
        )

    def activate_image(self, key: str, fit: bool = True) -> None:
        if key not in self.image_states:
            return
        self._sync_active_state()
        self.active_image_key = key
        self.active_image_var.set(key)
        state = self.image_states[key]
        self.original_image = state.image
        self.gray = state.gray
        self.image_path = state.path
        self.rois = state.rois
        self.reference_uid = state.reference_uid
        self.selected_uid = state.selected_uid
        self._sync_display_controls(state)
        if self._rotation_job is not None:
            self.after_cancel(self._rotation_job)
            self._rotation_job = None
        self._set_rotation_control(state.rotation)
        if fit and self.original_image is not None:
            self.after(10, self._fit_image)
        else:
            self._redraw()
        name = self.image_path.name if self.image_path else "none selected"
        self.status_var.set(f"Now editing: {state.label} image | {name}")

    def _sync_active_state(self) -> None:
        state = self.image_states[self.active_image_key]
        state.image = self.original_image
        state.gray = self.gray
        state.path = self.image_path
        state.rois = self.rois
        state.reference_uid = self.reference_uid
        state.selected_uid = self.selected_uid

    def _category_name_typed(self, key: str) -> None:
        """Debounce typing: rename once the user pauses for 300 ms."""

        job = self._name_jobs.pop(key, None)
        if job is not None:
            self.after_cancel(job)
        self._name_jobs[key] = self.after(300, lambda: self.apply_category_name(key))

    def apply_category_name(self, key: str, finished: bool = False) -> None:
        """Use the box's text as the image's name (empty box = default name).

        Only labels change, so this refreshes the table and the ROI file
        without re-measuring any bands. ``finished`` (Enter or leaving the
        box) also writes the default back into an empty box.
        """

        job = self._name_jobs.pop(key, None)
        if job is not None:
            self.after_cancel(job)
        name = self.category_name_vars[key].get().strip() or DEFAULT_CATEGORY[key]
        if finished and not self.category_name_vars[key].get().strip():
            self.category_name_vars[key].set(name)
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

    @staticmethod
    def _level_value(state: ImageState, fraction: float) -> str:
        lo, hi = state.display_range
        return f"{lo + fraction * (hi - lo):.3g}"

    def _sync_display_controls(self, state: ImageState) -> None:
        """Show an image's stored black/white/mode on the controls without redrawing."""

        self._syncing_display = True
        try:
            self.black_var.set(state.display_black * LEVEL_STEPS)
            self.white_var.set(state.display_white * LEVEL_STEPS)
            self.display_mode_var.set(next(k for k, v in DISPLAY_MODES.items() if v == state.display_mode))
            self.black_value_label.configure(text=self._level_value(state, state.display_black))
            self.white_value_label.configure(text=self._level_value(state, state.display_white))
        finally:
            self._syncing_display = False

    def _display_levels_changed(self, moved: str) -> None:
        if self._syncing_display:
            return
        state = self.image_states[self.active_image_key]
        black = round(float(self.black_var.get())) / LEVEL_STEPS
        white = round(float(self.white_var.get())) / LEVEL_STEPS
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
        state = self.image_states[self.active_image_key]
        state.display_mode = DISPLAY_MODES.get(self.display_mode_var.get(), "linear")
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
            self.status_var.set(
                f"Auto contrast ({state.label} image): black {self._level_value(state, state.display_black)}, "
                f"white {self._level_value(state, state.display_white)} "
                f"({AUTO_PERCENTILES[0]:g}th / {AUTO_PERCENTILES[1]:g}th percentiles). Display only."
            )


    # ----------------------------------------------------------------------
    # Rotation
    # ----------------------------------------------------------------------

    def _set_rotation_control(self, angle: float) -> None:
        """Show an image's stored angle on the slider without triggering a rotation."""

        self._syncing_rotation = True
        try:
            self.rotation_var.set(round(angle, 1))
        finally:
            self._syncing_rotation = False

    def _read_rotation(self) -> float | None:
        try:
            angle = float(self.rotation_var.get())
        except (tk.TclError, ValueError):
            return None  # half-typed entry such as "-" or "1."
        return round(max(-ROTATION_LIMIT, min(ROTATION_LIMIT, angle)), 1)

    def _rotation_changed(self) -> None:
        if self._syncing_rotation:
            return
        state = self.image_states[self.active_image_key]
        angle = self._read_rotation()
        if angle is None:
            return
        if state.source_gray is None or angle == state.rotation:
            if state.source_gray is None:
                self._set_rotation_control(0.0)
            return
        if state.source_gray.ndim != 2:
            self._set_rotation_control(0.0)
            self.status_var.set("Rotation is only available for single-channel (greyscale) images.")
            return
        if self.rois:
            # Axis-aligned ROIs can't follow a rotated band, so they must go.
            count = len(self.rois)
            if not messagebox.askyesno(
                APP_TITLE,
                f"Rotating the {state.label} image will clear its {count} ROI(s). "
                "Rotate and clear them?",
            ):
                self._set_rotation_control(state.rotation)
                return
            self.rois.clear()
            self.reference_uid = None
            self.selected_uid = None
            self._sync_active_state()
            self._update_results()
        # Debounce: the slider fires continuously while dragging; only do the
        # full-resolution rotation once it pauses.
        if self._rotation_job is not None:
            self.after_cancel(self._rotation_job)
        self._rotation_job = self.after(150, lambda key=self.active_image_key: self._apply_rotation(key))

    def _apply_rotation(self, key: str) -> None:
        self._rotation_job = None
        if key != self.active_image_key:
            return
        angle = self._read_rotation()
        state = self.image_states[key]
        if angle is None or state.source_gray is None or angle == state.rotation:
            return
        self._rotate_state(state, angle)
        self.original_image = state.image
        self.gray = state.gray
        self._redraw()
        log.info("Rotated %s image %s to %+.1f deg", key, state.path, angle)
        self._save_roi_file(key)
        self.status_var.set(
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
        state.preview_image = None
        state.preview_size = None
        state.preview_base = None
        state.preview_base_float = None
        state.preview_display_key = None
        state.detection_cache_key = None
        state.detection_cache = None
        state.auto_polarity = None

    def reset_rotation(self) -> None:
        self.rotation_var.set(0.0)
        self._rotation_changed()


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
                messagebox.showwarning(
                    APP_TITLE,
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
            messagebox.showwarning(
                APP_TITLE,
                f"The saved ROI file for this image could not be read ({exc}).\n"
                + (f"It has been kept as {backup.name}." if backup else ""),
            )
            return "saved ROI file unreadable"

        actual_size = [int(state.source_gray.shape[1]), int(state.source_gray.shape[0])]
        if saved_size != actual_size:
            log.warning("ROI file %s is for a %s image, but %s is %s", path, saved_size, state.path, actual_size)
            backup = set_aside(path, "size-mismatch")
            messagebox.showwarning(
                APP_TITLE,
                f"The saved ROIs were for a {saved_size[0]} × {saved_size[1]} px image, but this image is "
                f"{actual_size[0]} × {actual_size[1]} px, so they were not loaded."
                + (f"\nThe old ROI file has been kept as {backup.name}." if backup else ""),
            )
            return "saved ROIs were for a different-sized image"
        saved_hash = data.get("image_sha256")
        if saved_hash and state.source_sha256 and saved_hash != state.source_sha256:
            log.warning("ROI file %s was saved for different image contents (same size)", path)
            if not messagebox.askyesno(
                APP_TITLE,
                "This image file has changed since its ROIs were saved (same size, different contents).\n\n"
                "Load the saved ROIs anyway?",
            ):
                backup = set_aside(path, "image-changed")
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
            return [], self.polarity_var.get()
        state = self.image_states[self.active_image_key]
        cache_key = (
            round(float(self.sensitivity_var.get()), 4),
            max(4, int(self.min_area_var.get())),
            self.polarity_var.get(),
        )
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
            messagebox.showinfo(APP_TITLE, "Please open a Western blot image first.")
            return
        try:
            cached_rois, actual = self._get_full_detection()
            rois = [
                replace(
                    roi,
                    uid=self.next_uid + offset,
                    name=f"Band {self.next_uid + offset}",
                )
                for offset, roi in enumerate(cached_rois)
            ]
        except Exception as exc:
            messagebox.showerror("Auto Detection Failed", str(exc))
            return
        candidate_count = len(rois)
        rois, marker_removed = filter_marker_lanes(rois, self.marker_mode_var.get())
        rois = expand_band_rois(
            rois,
            (0, 0, self.gray.shape[1], self.gray.shape[0]),
            padding=max(0, self.region_padding_var.get()),
            gap=max(0, self.roi_gap_var.get()),
        )
        self.rois = rois
        self.next_uid = max((r.uid for r in rois), default=0) + 1
        self.reference_uid = None
        self.selected_uid = rois[0].uid if rois else None
        self._update_results()
        self._redraw()
        self.status_var.set(
            f"Full-image detection complete: kept {len(rois)}/{candidate_count} bands; "
            f"{len(marker_removed)} removed by marker filter; ROIs expanded by "
            f"{max(0, self.region_padding_var.get())} px; polarity used: {actual}."
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
        try:
            found, actual = detect_bands_in_region(
                self.gray,
                (x1, y1, x2, y2),
                sensitivity=self.sensitivity_var.get(),
                min_area=max(4, self.min_area_var.get()),
                polarity=self.polarity_var.get(),
                first_uid=self.next_uid,
                full_detection=self._get_full_detection(),
            )
        except Exception as exc:
            messagebox.showerror("Region Split Failed", str(exc))
            return
        if not found:
            self.status_var.set("No bands found in this region. Try raising Sensitivity, lowering Minimum Area, "
                "or extending the selection further above and below the bands.")
            return

        candidate_count = len(found)
        found, marker_removed = filter_marker_lanes(found, self.marker_mode_var.get())
        found = align_rois_to_region_signal(
            self.gray,
            found,
            (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
            polarity=actual,
        )
        found = expand_band_rois(
            found,
            (0, 0, self.gray.shape[1], self.gray.shape[0]),
            padding=max(0, self.region_padding_var.get()),
            gap=max(0, self.roi_gap_var.get()),
        )
        signal_threshold: float | None = None
        if self.filter_below_mean_var.get():
            found, signal_threshold = filter_bands_below_mean_signal(
                self.gray,
                found,
                polarity=actual,
                region=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
            )
        weak_excluded_count = candidate_count - len(marker_removed) - len(found)
        if not found:
            self.status_var.set(
                f"All {candidate_count} candidates in the region were filtered out: "
                f"{len(marker_removed)} as marker, {weak_excluded_count} as weak. "
                "Try changing the Marker Filter or turning off the weak-candidate filter."
            )
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
        self.status_var.set(
            f"Region split complete: kept {len(found)}/{candidate_count} bands"
            f"; {len(marker_removed)} removed by marker filter{filter_text}; "
            f"ROIs expanded by {max(0, self.region_padding_var.get())} px, "
            f"gap {max(0, self.roi_gap_var.get())} px; "
            f"polarity used: {actual}."
        )


    # ----------------------------------------------------------------------
    # Canvas view and mouse
    # ----------------------------------------------------------------------

    def _fit_image(self) -> None:
        """Scale the image to fill the view (enlarging small blots too)."""

        self._auto_fit = True
        if self.original_image is None:
            return
        cw, ch = max(1, self.canvas.winfo_width() - 10), max(1, self.canvas.winfo_height() - 10)
        self.zoom = min(cw / self.original_image.width, ch / self.original_image.height, 8.0)
        self.zoom = max(self.zoom, 0.05)
        self._redraw()

    def _canvas_resized(self) -> None:
        # Keep the blot fitted to the view as panes/window are resized, until
        # the user zooms by hand (Fit restores this).
        if self._auto_fit and self.original_image is not None:
            self._fit_image()
        else:
            self._redraw()

    def _display_geometry(self) -> tuple[int, int, float, float]:
        assert self.original_image is not None
        dw = max(1, round(self.original_image.width * self.zoom))
        dh = max(1, round(self.original_image.height * self.zoom))
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        ox, oy = max(4.0, (cw - dw) / 2), max(4.0, (ch - dh) / 2)
        return dw, dh, ox, oy

    def _image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self.image_origin
        return ox + x * self.zoom, oy + y * self.zoom

    def _canvas_to_image(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self.image_origin
        return (self.canvas.canvasx(x) - ox) / self.zoom, (self.canvas.canvasy(y) - oy) / self.zoom

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
        self.canvas.delete("all")
        if self.original_image is None:
            self.canvas.create_text(
                max(1, self.canvas.winfo_width()) / 2,
                max(1, self.canvas.winfo_height()) / 2,
                text="Open a Western blot image to begin analysis.", fill="#c8c8c8", font=("Arial", 18),
            )
            self._update_profile_plot()
            return
        dw, dh, ox, oy = self._display_geometry()
        self.image_origin = (ox, oy)
        state = self.image_states[self.active_image_key]
        high_bit_depth = state.display_data is not None
        base_missing = (
            state.preview_base_float is None if high_bit_depth else state.preview_base is None
        )
        if base_missing or state.preview_size != (dw, dh):
            if high_bit_depth:
                # Resample the float data itself (PIL mode "F") so no
                # precision is lost before contrast is applied.
                resized = Image.fromarray(state.display_data).resize(
                    (dw, dh), Image.Resampling.LANCZOS,
                )
                state.preview_base_float = np.asarray(resized, dtype=np.float32)
            else:
                state.preview_base = self.original_image.resize((dw, dh), Image.Resampling.LANCZOS)
            state.preview_size = (dw, dh)
            state.preview_image = None
        display_key = (state.display_black, state.display_white, state.display_mode)
        if state.preview_image is None or state.preview_display_key != display_key:
            if high_bit_depth:
                display = map_float_for_display(state.preview_base_float, *display_key)
            else:
                display = map_image_for_display(state.preview_base, *display_key)
            state.preview_image = ImageTk.PhotoImage(display)
            state.preview_display_key = display_key
        self.tk_image = state.preview_image
        self.canvas.create_image(ox, oy, image=self.tk_image, anchor="nw", tags="image")
        self.canvas.configure(scrollregion=(0, 0, max(self.canvas.winfo_width(), ox + dw + 4), max(self.canvas.winfo_height(), oy + dh + 4)))
        if self.show_grid_var.get():
            self._draw_grid(dw, dh, ox, oy)
        if self.background_mode == "profile":
            # Show the region actually measured (profile minima) as a dashed box.
            for key, measurement in self.combined_measurements:
                if key != self.active_image_key or measurement.measure_x1 is None:
                    continue
                mx1, my1 = self._image_to_canvas(measurement.measure_x1, measurement.measure_y1)
                mx2, my2 = self._image_to_canvas(measurement.measure_x2, measurement.measure_y2)
                manual = measurement.edges_source.startswith("manual")
                self.canvas.create_rectangle(
                    mx1, my1, mx2, my2, outline=EDGE_COLOR if manual else HIGHLIGHT_COLOR,
                    width=1, dash=(4, 3), tags=("measured",),
                )
        for roi in self.rois:
            x1, y1 = self._image_to_canvas(roi.x, roi.y)
            x2, y2 = self._image_to_canvas(roi.x2, roi.y2)
            selected = roi.uid == self.selected_uid
            color = ROI_SELECTED_COLOR if selected else ROI_COLOR
            width = 3 if selected else 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags=("roi", f"roi_{roi.uid}"))
            label = f"{roi.uid}"
            text_id = self.canvas.create_text(x1 + 2, y1 - 2, text=label, fill="white", anchor="sw", font=("Arial", 11, "bold"), tags=("roi", f"roi_{roi.uid}"))
            bbox = self.canvas.bbox(text_id)
            if bbox:
                bg = self.canvas.create_rectangle(*bbox, fill=color, outline=color, tags=("roi", f"roi_{roi.uid}"))
                self.canvas.tag_lower(bg, text_id)
        self._update_profile_plot()

    def _draw_grid(self, dw: int, dh: int, ox: float, oy: float) -> None:
        """Overlay a grid every N image pixels (display only)."""

        try:
            spacing = int(self.grid_spacing_var.get())
        except (tk.TclError, ValueError):
            return
        if spacing < 1 or self.zoom <= 0:
            return
        # Keep lines at least ~8 screen px apart at low zoom by skipping some.
        step = spacing * max(1, int(np.ceil(8 / (spacing * self.zoom))))
        width, height = self.original_image.width, self.original_image.height
        for x in range(step, width, step):
            cx = ox + x * self.zoom
            self.canvas.create_line(cx, oy, cx, oy + dh, fill=HIGHLIGHT_COLOR, dash=(3, 3), tags="grid")
        for y in range(step, height, step):
            cy = oy + y * self.zoom
            self.canvas.create_line(ox, cy, ox + dw, cy, fill=HIGHLIGHT_COLOR, dash=(3, 3), tags="grid")

    def _canvas_press(self, event: tk.Event) -> None:
        if self.original_image is None:
            return
        p = self._canvas_to_image(event.x, event.y)
        if not self._point_in_image(p):
            return
        hit = self._roi_at_point(p)
        if hit is not None:
            self._select_uid(hit.uid)
            self.status_var.set(f"Selected ID {hit.uid}: {hit.name}")
        self.drag_start = p
        cx, cy = self._image_to_canvas(*p)
        self.preview_item = self.canvas.create_rectangle(cx, cy, cx, cy, outline=HIGHLIGHT_COLOR, width=2, dash=(5, 3))

    def _canvas_drag(self, event: tk.Event) -> None:
        if self.drag_start is None or self.preview_item is None:
            return
        p = self._canvas_to_image(event.x, event.y)
        p = self._clamp_point(p)
        x1, y1 = self._image_to_canvas(*self.drag_start)
        x2, y2 = self._image_to_canvas(*p)
        self.canvas.coords(self.preview_item, x1, y1, x2, y2)

    def _canvas_release(self, event: tk.Event) -> None:
        if self.drag_start is None or self.original_image is None:
            return
        end = self._clamp_point(self._canvas_to_image(event.x, event.y))
        start = self.drag_start
        self.drag_start = None
        if self.preview_item is not None:
            self.canvas.delete(self.preview_item)
            self.preview_item = None
        if self.mode_var.get() == "region":
            x1, x2 = sorted((start[0], end[0]))
            y1, y2 = sorted((start[1], end[1]))
            x1, y1 = self._clamp_point((x1, y1))
            x2, y2 = self._clamp_point((x2, y2), upper=True)
            if x2 - x1 < 3 or y2 - y1 < 3:
                self.status_var.set("Selected region is too small; please drag a larger box.")
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
                self.status_var.set("Band selected; drag to create a new ROI.")
            else:
                self.status_var.set("ROI too small; not added.")
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
        self.status_var.set(f"Added manual ROI ID {self.selected_uid}")

    def _canvas_context_menu(self, event: tk.Event) -> str | None:
        if self.original_image is None:
            return None
        point = self._canvas_to_image(event.x, event.y)
        roi = self._roi_at_point(point)
        if roi is None:
            self.status_var.set("Right-click an ROI to rename or delete it.")
            return None
        self._select_uid(roi.uid)
        self.status_var.set(f"Selected ID {roi.uid}: {roi.name}")
        try:
            self.roi_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.roi_context_menu.grab_release()
        return "break"

    def _mousewheel(self, event: tk.Event) -> None:
        if self.original_image is None:
            return
        factor = 1.12 if event.delta > 0 else 1 / 1.12
        self._auto_fit = False
        self.zoom = float(np.clip(self.zoom * factor, 0.05, 8.0))
        self._redraw()
        self.status_var.set(f"Zoom: {self.zoom * 100:.0f}%")


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
            self.status_var.set(f"ID {roi_uid}: {key} edge set to {edges[key]} px. Reset Edges restores automatic edges.")

    def reset_edges(self) -> None:
        """Return the selected band to automatic (profile-minimum) edges."""

        state = self.image_states[self.active_image_key]
        uid = self.selected_uid
        if uid is None or uid not in state.edge_overrides:
            self.status_var.set("The selected band already uses automatic edges.")
            return
        state.edge_overrides.pop(uid, None)
        log.info("Reset edges of ROI %s on %s image to automatic", uid, self.active_image_key)
        self._update_results()
        self._redraw()
        self.status_var.set(f"ID {uid}: edges reset to the automatic profile minima.")


    # ----------------------------------------------------------------------
    # Measurement results and pairing
    # ----------------------------------------------------------------------

    def _measurement_polarity(self, state: ImageState) -> str:
        """The polarity actually used for an image (resolves 'auto')."""

        requested = self.polarity_var.get()
        if requested != "auto":
            return requested
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
            # Every ROI edit (add, split, detect, rename, delete, clear) and
            # category rename passes through here, so autosave from here.
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

        selected = self.selected_uid
        self.tree.delete(*self.tree.get_children())
        for key, m in self.combined_measurements:
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
            self.tree.insert("", tk.END, iid=str(m.uid), values=(
                state.label, m.uid, m.name, m.lane, f"{m.mean_gray:.2f}", f"{m.background:.2f}",
                f"{m.corrected_mean:.2f}", f"{m.integrated_density:.2f}",
                density_text("target"), density_text("reference"), ratio, pairing_status,
                "" if result is None else result.missing_reason,
            ))
        if selected is not None and self.tree.exists(str(selected)):
            self.tree.selection_set(str(selected))
            self.tree.see(str(selected))

    def _tree_select(self, _event: tk.Event | None = None) -> None:
        selection = self.tree.selection()
        if selection:
            uid = int(selection[0])
            selected_key = next(
                (key for key, state in self.image_states.items() if any(roi.uid == uid for roi in state.rois)),
                None,
            )
            if selected_key is not None and selected_key != self.active_image_key:
                self.image_states[selected_key].selected_uid = uid
                self.activate_image(selected_key)
                return
            self.selected_uid = uid
            self._sync_active_state()
            self._redraw()

    def _select_uid(self, uid: int) -> None:
        self.selected_uid = uid
        if self.tree.exists(str(uid)):
            self.tree.selection_set(str(uid))
            self.tree.see(str(uid))
        self._redraw()

    def rename_selected(self) -> None:
        roi = next((r for r in self.rois if r.uid == self.selected_uid), None)
        if roi is None:
            return
        name = simpledialog.askstring("Rename Band", "Enter Name:", initialvalue=roi.name, parent=self)
        if name is None or not name.strip():
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
        self.status_var.set(f"Deleted ROI ID {uid}")

    def clear_rois(self) -> None:
        if not self.rois:
            return
        label = self.image_states[self.active_image_key].label
        if not messagebox.askyesno(APP_TITLE, f"Clear all ROIs on the {label} image?"):
            return
        self.rois.clear()
        self.reference_uid = None
        self.selected_uid = None
        self._update_results()
        self._redraw()
        self.status_var.set(f"Cleared all ROIs on the {label} image; the other image's results are unchanged.")

    # ----------------------------------------------------------------------
    # Settings window
    # ----------------------------------------------------------------------

    def open_settings(self) -> None:
        """Measurement and detection settings in their own window (non-modal)."""

        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.deiconify()
            self.settings_window.lift()
            return
        window = tk.Toplevel(self)
        window.title("Measurement & Detection Settings")
        window.transient(self)
        window.resizable(True, False)
        self.settings_window = window
        body = ttk.Frame(window, padding=10)
        body.pack(fill=tk.BOTH, expand=True)
        measurement = ttk.LabelFrame(body, text="Measurement", padding=9)
        measurement.pack(fill=tk.X, pady=(0, 7))
        background_label = ttk.Label(measurement, text="Background")
        background_label.grid(row=0, column=0, sticky="w")
        background_box = ttk.Combobox(
            measurement, state="readonly", width=32, textvariable=self.background_label_var,
            values=list(BACKGROUND_LABELS.values()),
        )
        background_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        background_box.bind("<<ComboboxSelected>>", lambda _event: self._background_changed())
        background_hint = (
            "How each band's local background is estimated:\n\n"
            "• Above/below: median of thin strips just above and below the band, "
            "in the same lane.\n"
            "• Ring: median of a ring of pixels all around the band.\n"
            "• Profile baseline (min. top/bottom, default): band edges at the minima of the band's x and y "
            "profiles; background is a straight line between the top and bottom minima. "
            "Edges can be dragged in the profile plots."
        )
        ToolTip(background_box, background_hint)
        ToolTip(background_label, background_hint)
        measurement.columnconfigure(1, weight=1)

        detection = ttk.LabelFrame(body, text="Detection Parameters", padding=9)
        detection.pack(fill=tk.X, pady=(0, 7))
        ttk.Label(detection, text="Band Polarity").grid(row=0, column=0, sticky="w")
        polarity = ttk.Combobox(detection, state="readonly", width=15, textvariable=self.polarity_var)
        polarity["values"] = ("light", "dark", "auto")
        polarity.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        polarity.bind("<<ComboboxSelected>>", lambda _event: self._update_results())
        ToolTip(polarity, "Light (default): positive bands, bright on a dark background "
                "(fluorescence, e.g. LI-COR). Dark: negative bands, dark on a light background "
                "(chemiluminescence film, colorimetric). Auto: decide from each image.")
        ttk.Label(detection, text="Sensitivity").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(detection, from_=1, to=100, variable=self.sensitivity_var).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        self.sensitivity_label = ttk.Label(detection, text="55")
        self.sensitivity_label.grid(row=2, column=2, padx=(5, 0), pady=(8, 0))
        self.sensitivity_label.configure(text=f"{self.sensitivity_var.get():.0f}")
        trace = self.sensitivity_var.trace_add(
            "write", lambda *_: self.sensitivity_label.configure(text=f"{self.sensitivity_var.get():.0f}"),
        )
        # Remove the trace with the window so it never updates a destroyed label.
        window.bind("<Destroy>", lambda e: e.widget is window and self.sensitivity_var.trace_remove("write", trace), add="+")
        ttk.Label(detection, text="Minimum Area (px)").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(detection, from_=4, to=100000, width=10, textvariable=self.min_area_var).grid(row=3, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(detection, text="ROI Expansion (px)").grid(row=4, column=0, sticky="w", pady=(8, 0))
        expansion = ttk.Spinbox(detection, from_=0, to=100, width=10, textvariable=self.region_padding_var)
        expansion.grid(row=4, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ToolTip(expansion, "Pixels added around each box after detection or region splitting.")
        ttk.Label(detection, text="ROI Gap (px)").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(detection, from_=0, to=30, width=10, textvariable=self.roi_gap_var).grid(row=6, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Checkbutton(
            detection,
            text="Exclude candidates below average corrected ROI signal",
            variable=self.filter_below_mean_var,
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(detection, text="Marker Filter").grid(row=8, column=0, sticky="w", pady=(8, 0))
        marker_mode = ttk.Combobox(detection, state="readonly", width=15, textvariable=self.marker_mode_var)
        marker_mode["values"] = ("auto", "left", "right", "off")
        marker_mode.grid(row=8, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ToolTip(marker_mode, "Removes the marker/ladder lane from detection. Auto: find it; "
                "Left / Right: it's the outermost lane on that side; Off: keep every lane.")
        detection.columnconfigure(1, weight=1)

        ttk.Button(body, text="Close", command=window.destroy).pack(anchor="e")
        window.update_idletasks()
        x = self.winfo_rootx() + max(0, self.winfo_width() - window.winfo_reqwidth() - 40)
        y = self.winfo_rooty() + 60
        window.geometry(f"+{x}+{y}")

    def _update_settings_summary(self) -> None:
        polarity = {
            "light": "Positive bands", "dark": "Negative bands", "auto": "Auto polarity",
        }.get(self.polarity_var.get(), self.polarity_var.get())
        background = {
            "above_below": "Above/below background", "ring": "Ring background", "profile": "Profile baseline",
        }.get(self.background_mode, self.background_mode)
        if hasattr(self, "settings_summary"):
            self.settings_summary.configure(text=f"{polarity} · {background}")

    def _background_changed(self) -> None:
        label = self.background_label_var.get()
        mode = next((key for key, text in BACKGROUND_LABELS.items() if text == label), "above_below")
        if mode == self.background_mode:
            return
        self.background_mode = mode
        self._update_settings_summary()
        log.info("Background method set to %s", mode)
        self._update_results()
        self._redraw()
        self.status_var.set(f"Background method: {label}. All bands re-measured.")


    # ----------------------------------------------------------------------
    # Exports
    # ----------------------------------------------------------------------

    def export_csv(self) -> None:
        self._update_results()
        if not self.normalized_results:
            messagebox.showinfo(APP_TITLE, "There are no quantification results to export.")
            return
        initial = "WB_target_reference_normalized_by_lane.csv"
        filename = filedialog.asksaveasfilename(
            title="Export Quantification Results", defaultextension=".csv", initialfile=initial,
            filetypes=[("CSV", "*.csv")],
        )
        if not filename:
            return
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
                    self.polarity_var.get(),
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
            messagebox.showerror("Profile Export Failed", f"The results CSV was saved, but the profiles were not:\n{exc}")
            self.status_var.set(f"CSV exported: {filename} (profile export failed)")
            return
        log.info("Exported results CSV %s (%d rows)", filename, len(self.normalized_results))
        log.info("Exported profiles %s%s", profile_csv, f" and {profile_png}" if profile_png else "")
        exported = [profile_csv.name] + ([profile_png.name] if profile_png else [])
        self.status_var.set(f"CSV exported: {filename} | profiles: {', '.join(exported)}")

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

    def export_annotated(self) -> None:
        if self.original_image is None or not self.rois:
            messagebox.showinfo(APP_TITLE, "Please open an image and create ROIs first.")
            return
        role = self.image_states[self.active_image_key].label
        safe_role = role.translate(str.maketrans({"/": "_", "\\": "_", ":": "_"}))
        initial = f"{self.image_path.stem if self.image_path else 'WB'}_{safe_role}_annotated.png"
        filename = filedialog.asksaveasfilename(
            title="Export Annotated Image", defaultextension=".png", initialfile=initial,
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif")],
        )
        if not filename:
            return
        annotated = self.original_image.convert("RGB")
        draw = ImageDraw.Draw(annotated)
        line_width = max(2, round(min(annotated.size) / 350))
        for roi in self.rois:
            color = (0, 180, 230)
            draw.rectangle((roi.x, roi.y, roi.x2, roi.y2), outline=color, width=line_width)
            draw.text((roi.x + 2, max(0, roi.y - 13)), f"{roi.uid} {roi.name}", fill=color, stroke_width=1, stroke_fill=(255, 255, 255))
        annotated.save(filename)
        log.info("Exported annotated %s image %s", self.active_image_key, filename)
        self.status_var.set(f"Annotated image exported: {filename}")


    # ----------------------------------------------------------------------
    # Log viewer and shutdown
    # ----------------------------------------------------------------------

    def open_log_viewer(self) -> None:
        """Show the log (opened files, ROI files, rotations, exports) in a window."""

        if self.log_path is None:
            messagebox.showinfo(APP_TITLE, "Logging is unavailable: no writable folder was found for the log file.")
            return
        if self.log_window is not None and self.log_window.winfo_exists():
            self.log_window.lift()
            self._refresh_log_viewer()
            return
        window = tk.Toplevel(self)
        window.title("Activity Log")
        window.geometry("980x480")
        top = ttk.Frame(window, padding=(8, 6))
        top.pack(fill=tk.X)
        ttk.Label(top, text=str(self.log_path), foreground="#555").pack(side=tk.LEFT)
        ttk.Button(top, text="Refresh", command=self._refresh_log_viewer).pack(side=tk.RIGHT)
        body = ttk.Frame(window, padding=(8, 0, 8, 8))
        body.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(body, wrap="none", font=("Courier", 11))
        yscroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=text.yview)
        xscroll = ttk.Scrollbar(body, orient=tk.HORIZONTAL, command=text.xview)
        text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self.log_window = window
        self.log_text = text
        self._refresh_log_viewer()

    def _refresh_log_viewer(self) -> None:
        for handler in log.handlers:
            handler.flush()
        try:
            content = self.log_path.read_text(encoding="utf-8")
        except OSError as exc:
            content = f"Could not read the log file: {exc}"
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert("1.0", content)
        self.log_text.configure(state="disabled")
        self.log_text.see(tk.END)

    def _on_close(self) -> None:
        log.info("Session ended")
        self.destroy()


def main() -> None:
    app = WBAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
