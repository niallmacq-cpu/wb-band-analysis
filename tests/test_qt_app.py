"""End-to-end test of wb_gui_qt with real Qt mouse events, on an offscreen display."""
import csv, json, os, shutil, subprocess, sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication, QMessageBox

import wb_gui_qt as gui
from wb_analysis import measure_all

OUT = HERE / "out"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir()
DATA = HERE / "data"
if not (DATA / "channel_reference.tif").exists():  # synthetic blots are generated, not committed
    subprocess.run([sys.executable, "make_blot.py"], cwd=DATA, check=True)
for name in ("channel_reference.tif", "channel_target.tif"):
    shutil.copy(DATA / name, OUT / name)

QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)  # auto-confirm dialogs
app = QApplication.instance() or QApplication(sys.argv)
win = gui.WBAnalyzerApp()
win.resize(1480, 880)
win.show()
for _ in range(5):
    app.processEvents()

results = []
def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

def send_mouse(widget, kind, pos, button=Qt.MouseButton.LeftButton, buttons=None, mods=Qt.KeyboardModifier.NoModifier):
    types = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
             "release": QEvent.Type.MouseButtonRelease}
    if buttons is None:
        buttons = Qt.MouseButton.NoButton if kind == "release" else button
    if kind == "move":
        ev_button = Qt.MouseButton.NoButton
    else:
        ev_button = button
    gp = widget.mapToGlobal(QPointF(pos))
    ev = QMouseEvent(types[kind], QPointF(pos), gp, ev_button, buttons, mods)
    QApplication.sendEvent(widget, ev)
    app.processEvents()

def view_drag(scene_a, scene_b, mods=Qt.KeyboardModifier.NoModifier, steps=6):
    vp = win.view.viewport()
    a = win.view.mapFromScene(QPointF(*scene_a)); b = win.view.mapFromScene(QPointF(*scene_b))
    a, b = QPointF(a), QPointF(b)
    send_mouse(vp, "press", a, mods=mods)
    for i in range(1, steps + 1):
        send_mouse(vp, "move", a + (b - a) * (i / steps), buttons=Qt.MouseButton.LeftButton, mods=mods)
    send_mouse(vp, "release", b, mods=mods)

def shot(name):
    for _ in range(3):
        app.processEvents()
    win.grab().save(str(OUT / name))

# ---- reference image: load, region split by mouse drag --------------------
ok = win.load_image("reference", str(OUT / "channel_reference.tif"))
app.processEvents()
check("load reference TIFF", ok and win.gray is not None and win.gray.shape == (700, 1100))
check("auto-fit zoom sensible", 0.3 < win.zoom < 2, f"zoom {win.zoom:.3f}")
view_drag((150, 370), (1070, 470))
ref_rois = list(win.image_states["reference"].rois)
check("region split finds 9 reference bands", len(ref_rois) == 9, f"{len(ref_rois)} ROIs; status: {win.status_label.text()[:90]}")
shot("01_reference_split.png")

# ---- target image ------------------------------------------------------------
win.load_image("target", str(OUT / "channel_target.tif"))
view_drag((150, 250), (1070, 350))
tgt_rois = list(win.image_states["target"].rois)
check("region split finds 9 target bands", len(tgt_rois) == 9, f"{len(tgt_rois)} ROIs")

# ---- numbers identical to calling wb_analysis directly ------------------------
for key in ("reference", "target"):
    st = win.image_states[key]
    direct = measure_all(st.gray, st.rois, "light", None, background_mode="profile", edge_overrides=st.edge_overrides)
    gui_vals = [m.integrated_density for k, m in win.combined_measurements if k == key]
    check(f"{key}: GUI densities == wb_analysis.measure_all", np.allclose(gui_vals, [m.integrated_density for m in direct], rtol=0, atol=0))

ratios = [r.target_over_reference for r in sorted(win.normalized_results, key=lambda r: r.lane)]
truth = [float(r["normalised_to_control"]) for r in csv.DictReader(open(DATA / "ground_truth.csv"))]
ratios = np.array(ratios, dtype=float)
norm = ratios / ratios[:3].mean()
print("normalised to control:", np.round(norm, 3).tolist())
print("ground truth:         ", truth)
check("9 lanes paired, ratios within 5% of truth", len(norm) == 9 and np.max(np.abs(norm / truth - 1)) < 0.05,
      f"max error {np.max(np.abs(norm / truth - 1)) * 100:.2f}%")
check("table has 18 rows", win.table.rowCount() == 18)

# ---- select a band by clicking the table, then by clicking the blot ----------
row = win._row_by_uid[tgt_rois[4].uid]
win.table.selectRow(row); app.processEvents()
check("table click selects band", win.selected_uid == tgt_rois[4].uid)
r3 = win.image_states["target"].rois[2]
c = win.view.mapFromScene(QPointF(*r3.center))
send_mouse(win.view.viewport(), "press", QPointF(c)); send_mouse(win.view.viewport(), "release", QPointF(c))
check("clicking a box selects it", win.selected_uid == r3.uid)
shot("02_target_selected.png")

# ---- drag profile edges with real mouse events on the matplotlib canvas -----
canvas = win.profile_canvas
ax_h, ax_v = win.profile_axes
canvas.draw(); app.processEvents()
pb = win._profile_pb
check("profile baseline computed for selected band", pb is not None)
before = next(m for k, m in win.combined_measurements if m.uid == r3.uid).integrated_density
dpr = canvas.devicePixelRatioF()
fig_h = canvas.figure.bbox.height

def axes_to_widget(ax, xdata, yfrac=0.5):
    y_data = ax.get_ylim()[0] + yfrac * (ax.get_ylim()[1] - ax.get_ylim()[0])
    dx, dy = ax.transData.transform((xdata, y_data))
    return QPointF(dx / dpr, (fig_h - dy) / dpr)

old_bottom = pb.bottom
start = axes_to_widget(ax_v, pb.bottom)
end = axes_to_widget(ax_v, pb.bottom + 6)
send_mouse(canvas, "press", start)
for i in range(1, 6):
    send_mouse(canvas, "move", start + (end - start) * (i / 5), buttons=Qt.MouseButton.LeftButton)
send_mouse(canvas, "release", end)
edges = win.image_states["target"].edge_overrides.get(r3.uid, {})
check("dragging the bottom marker sets a manual edge", "bottom" in edges and abs(edges["bottom"] - (old_bottom + 6)) <= 1,
      f"edges {edges}, auto bottom was {old_bottom}")
after = next(m for k, m in win.combined_measurements if m.uid == r3.uid)
check("table re-measured after the drag", after.integrated_density != before and after.edges_source.startswith("manual"),
      f"{before:,.0f} -> {after.integrated_density:,.0f}")
start = axes_to_widget(ax_h, win._profile_pb.left)
end = axes_to_widget(ax_h, win._profile_pb.left - 3)
send_mouse(canvas, "press", start); send_mouse(canvas, "move", end, buttons=Qt.MouseButton.LeftButton); send_mouse(canvas, "release", end)
check("dragging the left marker on the x plot", "left" in win.image_states["target"].edge_overrides.get(r3.uid, {}))
shot("03_edges_dragged.png")
saved = json.loads((OUT / "channel_target.tif.rois.json").read_text())
check("manual edges written to ROI file", any("profile_edges" in r for r in saved["rois"]))
win.reset_edges()
check("Reset Edges clears them", r3.uid not in win.image_states["target"].edge_overrides)

# ---- move / resize the selected box --------------------------------------------
roi = next(r for r in win.rois if r.uid == r3.uid)
view_drag((roi.center[0], roi.y2), (roi.center[0], roi.y2 + 8))      # bottom edge down
roi2 = next(r for r in win.rois if r.uid == r3.uid)
check("dragging the bottom edge resizes the box", roi2.height == roi.height + 8 and roi2.y == roi.y,
      f"h {roi.height} -> {roi2.height}")
view_drag(roi2.center, (roi2.center[0] + 5, roi2.center[1] - 4))     # move
roi3 = next(r for r in win.rois if r.uid == r3.uid)
check("dragging inside moves the box", abs(roi3.x - roi2.x - 5) <= 1 and abs(roi3.y - roi2.y + 4) <= 1 and (roi3.width, roi3.height) == (roi2.width, roi2.height),
      f"moved by ({roi3.x - roi2.x}, {roi3.y - roi2.y}), size {roi2.width}x{roi2.height} -> {roi3.width}x{roi3.height}, source {roi3.source!r}")
n_before = len(win.rois)
win.rect_radio.setChecked(True)
view_drag((roi3.center[0] - 10, roi3.center[1]), (roi3.center[0] + 10, roi3.center[1] + 10),
          mods=Qt.KeyboardModifier.ShiftModifier)
check("Shift-drag inside the selected box draws a new ROI instead", len(win.rois) == n_before + 1)
win.delete_selected()
check("Delete ROI removes it", len(win.rois) == n_before)
win.region_radio.setChecked(True)

# ---- wheel zoom, contrast, grid ----------------------------------------------
z0 = win.zoom
vp = win.view.viewport()
pos = QPointF(vp.width() / 2, vp.height() / 2)
wheel = QWheelEvent(pos, vp.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 120), Qt.MouseButton.NoButton,
                    Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
QApplication.sendEvent(vp, wheel); app.processEvents()
check("wheel zooms in", abs(win.zoom / z0 - 1.12) < 1e-6, f"{z0:.3f} -> {win.zoom:.3f}")
win._fit_image()
win.level_sliders["white"].setValue(300); app.processEvents()
check("White slider changes display only", win.image_states["target"].display_white == 0.3)
win.mode_box.setCurrentText("Log"); win.grid_check.setChecked(True); app.processEvents()
shot("04_log_grid.png")
win.grid_check.setChecked(False); win.mode_box.setCurrentText("Linear"); win.auto_contrast()

# ---- rotation (clears ROIs after confirmation), then reset -------------------
win.rotation_spin.setValue(2.5); app.processEvents()
win._rotation_timer.timeout.emit(); app.processEvents()
st = win.image_states["target"]
check("rotation applied and ROIs cleared", st.rotation == 2.5 and not st.rois and st.gray.shape != (700, 1100),
      f"shape {st.gray.shape}")
check("rotation slider in step", win.rotation_slider.value() == 25)
win.reset_rotation(); app.processEvents(); win._rotation_timer.timeout.emit(); app.processEvents()
check("reset restores the exact original pixels", st.rotation == 0.0 and st.gray is st.source_gray)
view_drag((150, 250), (1070, 350))

# ---- exports -------------------------------------------------------------------
win.export_csv(str(OUT / "results.csv"))
rows = list(csv.DictReader(open(OUT / "results.csv", encoding="utf-8-sig")))
check("CSV export: 9 paired rows", len(rows) == 9 and all(r["target_over_reference"] for r in rows))
check("profiles CSV and PNG exported", (OUT / "results_profiles.csv").exists() and (OUT / "results_profiles.png").exists())
win.export_annotated(str(OUT / "annotated.png"))
check("annotated image exported", (OUT / "annotated.png").exists())

# ---- settings window and log viewer -------------------------------------------
win.open_settings(); app.processEvents()
win.settings_window.grab().save(str(OUT / "05_settings.png"))
combo = win.settings_window.findChildren(gui.QComboBox)[0]
combo.setCurrentText(gui.BACKGROUND_LABELS["above_below"]); app.processEvents()
check("changing background in Settings re-measures", win.background_mode == "above_below"
      and all(m.background_method == "above_below" for _k, m in win.combined_measurements))
combo.setCurrentText(gui.BACKGROUND_LABELS["profile"]); app.processEvents()
win.settings_window.hide()
win.open_log_viewer(); app.processEvents()
check("log viewer shows the session", "Exported results CSV" in win.log_text.toPlainText())
win.log_window.hide()

# ---- ROI files restore into a fresh window ---------------------------------------
win2 = gui.WBAnalyzerApp(); win2.resize(1480, 880); win2.show(); app.processEvents()
win2.load_image("reference", str(OUT / "channel_reference.tif"))
win2.load_image("target", str(OUT / "channel_target.tif"))
check("ROIs restored from sidecar files", len(win2.image_states["reference"].rois) == 9
      and len(win2.image_states["target"].rois) == 9)
r_old = sorted(r.target_over_reference for r in win.normalized_results)
r_new = sorted(r.target_over_reference for r in win2.normalized_results)
check("restored session gives identical ratios", np.allclose(r_old, r_new, rtol=0, atol=0))
win2.close()

shot("06_final.png")
print(f"\n{sum(ok for _n, ok in results)}/{len(results)} checks passed")
sys.exit(0 if all(ok for _n, ok in results) else 1)
