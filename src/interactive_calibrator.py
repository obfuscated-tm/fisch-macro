import logging
import time
import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.calibrator import Calibrator
from src.config import ROIBounds

logger = logging.getLogger("interactive_calibrator")

COLORS = {
    "bg": "#1a1a2e",
    "bg_card": "#0f3460",
    "accent": "#e94560",
    "accent_green": "#00d474",
    "accent_yellow": "#f5a623",
    "accent_blue": "#4fc3f7",
    "text": "#ffffff",
}


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


class InteractiveCalibrator(tk.Toplevel):
    """A full-screen calibration window: take a screenshot, draw the regions.

    It used to pick colours too -- the fish's, and the bar's on and off target
    -- and write them to the profile as HSV ranges. Nothing read them. The
    detector matched colour once, but moved to reading the reel track's
    structure instead (see :mod:`src.reel_vision`), which is what makes it work
    across rods that draw the bar white, dark red or as a rainbow gradient. The
    three colour steps outlived their last consumer and stayed in the window,
    asking for three careful clicks that changed nothing.
    """

    def __init__(self, parent, engine, config_manager, window_tracker, background_bgr=None):
        super().__init__(parent)
        self.engine = engine
        self.config = config_manager
        self.window_tracker = window_tracker
        self.settings = self.config.load_settings()
        self.profile = self.config.get_active_profile()
        self.window_tracker.invalidate_cache()
        self.window_bounds = self.window_tracker.get_roblox_bounds()
        self.monitor_left = 0
        self.monitor_top = 0
        self._bg_image_id = None

        self.title("Interactive Calibration")
        self.configure(bg="black")
        self.withdraw()

        # Calibration state
        self.mode = None  # 'bar_roi', 'progress_roi', 'shake_roi'
        self.rect_start = None
        self.current_rect_id = None
        self.drawn_rects = {}  # Store rect IDs for the different ROIs
        # Which steps this session has actually changed. Drives the checklist
        # and the confirmation on Cancel, so the window can say what is about
        # to be thrown away rather than silently throwing it away.
        self.completed = set()

        if background_bgr is not None:
            self._load_screenshot_frame(background_bgr)
        else:
            self._capture_screenshot_hidden()

        self._build_ui()
        self._bind_events()
        self._draw_existing_rois()

        self.update_idletasks()
        self.attributes("-fullscreen", True)
        self.attributes("-topmost", True)
        self.deiconify()
        self.lift()

    def _read_monitor_origin(self) -> None:
        import mss

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            self.monitor_left = int(monitor.get("left", 0))
            self.monitor_top = int(monitor.get("top", 0))

    def _load_screenshot_frame(self, frame: np.ndarray) -> None:
        """Prepare a pre-captured BGR frame for the canvas (no live grab)."""
        self.scale_factor = self.window_tracker.get_scale_factor()
        self._read_monitor_origin()
        self.raw_bgr = frame.copy()
        self.photo_img = self._frame_to_photo(self.raw_bgr)

    def _frame_to_photo(self, frame: np.ndarray) -> ImageTk.PhotoImage:
        h, w = frame.shape[:2]
        logical_w = int(w / self.scale_factor)
        logical_h = int(h / self.scale_factor)
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        if self.scale_factor > 1.0:
            pil_img = pil_img.resize((logical_w, logical_h), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(pil_img)

    def _capture_screenshot_hidden(self) -> None:
        """Grab the screen while this overlay is hidden (avoids black slide frames)."""
        self.withdraw()
        self.update_idletasks()
        time.sleep(0.25)
        calibrator = Calibrator(self.engine.detector, self.config)
        frame = calibrator.capture_stable_screenshot(wait_seconds=0.1, max_attempts=6)
        if frame is None:
            raise RuntimeError(
                "Could not capture a clean screenshot. Close fullscreen animation, "
                "then use Retake Screenshot or reopen calibration."
            )
        self._load_screenshot_frame(frame)

    def _retake_screenshot(self) -> None:
        """Hide overlay, wait for macOS space animation, then refresh the background."""
        self.instruction_label.config(text="Retaking screenshot — please wait…")
        self.update_idletasks()
        was_fullscreen = self.attributes("-fullscreen")
        if was_fullscreen:
            self.attributes("-fullscreen", False)
        self.withdraw()
        self.update_idletasks()
        time.sleep(0.55)
        calibrator = Calibrator(self.engine.detector, self.config)
        frame = calibrator.capture_stable_screenshot(wait_seconds=0.15, max_attempts=8)
        if frame is None:
            messagebox.showerror(
                "Screenshot Failed",
                "Capture still shows black bars from the fullscreen animation.\n\n"
                "Wait a moment, then click Retake Screenshot again.",
                parent=self,
            )
            self.deiconify()
            if was_fullscreen:
                self.attributes("-fullscreen", True)
            return
        self._load_screenshot_frame(frame)
        self.canvas.config(width=self.photo_img.width(), height=self.photo_img.height())
        if self._bg_image_id is not None:
            self.canvas.delete(self._bg_image_id)
        self._bg_image_id = self.canvas.create_image(0, 0, image=self.photo_img, anchor=tk.NW)
        self.canvas.tag_lower(self._bg_image_id)
        self.drawn_rects.clear()
        self._draw_existing_rois()
        self.instruction_label.config(text="Screenshot updated. Select a tool to continue.")
        self.deiconify()
        if was_fullscreen:
            self.attributes("-fullscreen", True)
        self.lift()

    #: (mode, button label, one-line instruction) for each calibration step.
    STEPS = (
        ("bar_roi", "1 · Reel bar area",
         "Drag a box around the whole slider track, end to end."),
        ("progress_roi", "2 · Progress bar",
         "Drag a box over the catch progress bar below the track."),
        ("shake_roi", "3 · Shake area",
         "Drag a big box over the area where SHAKE prompts appear."),
    )

    #: Toolbar button colours: (style name, background, active background).
    #:
    #: Every one carries white text and clears WCAG AA's 4.5:1 against it, which
    #: the panel's own accent shades did not: this toolbar used to put white on
    #: #4fc3f7 at 2.0:1 and, for the selected step, on #9ad9fb at 1.5:1 -- pale
    #: enough that the label was barely there even once macOS could be
    #: persuaded to draw the colour at all. A selected step is now a *darker*
    #: shade of its own colour rather than a lighter one, which reads as
    #: pressed and gains contrast instead of losing it.
    BUTTON_STYLES = (
        ("CalRoi", "#1565c0", "#11529c"),       # the box-drawing steps
        ("CalRoiOn", "#0d47a1", "#0d47a1"),     # ...while selected
        ("CalRetake", "#5c6bc0", "#4a58a8"),
        ("CalSave", "#1f8438", "#186b2d"),
        ("CalCancel", "#c62534", "#a81f2c"),
    )

    def _configure_button_styles(self):
        """Give the toolbar buttons colours macOS will actually draw.

        tk.Button takes a background and, on macOS, ignores it: the Aqua button
        is drawn by the system, so every one of these came out as a plain white
        rectangle -- with white text on it, which is to say invisible. Captured
        under Tk 9.0.3, a tk.Button asking for #e94560 renders white whether or
        not it is given a flat relief and no border.

        ttk's "clam" theme draws the button itself rather than asking the
        system to, so the colour is honoured. It is also the theme the control
        panel already uses, and being pure Tk it looks the same wherever the
        panel runs -- the same reason the fonts here are sized in pixels.
        """
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:      # pragma: no cover - clam ships with Tk
            pass
        for name, background, active in self.BUTTON_STYLES:
            style.configure(
                f"{name}.TButton",
                background=background,
                foreground="white",
                borderwidth=0,
                focuscolor=background,
                padding=(10, 5),
                font=("Helvetica Neue", -12),
            )
            style.map(
                f"{name}.TButton",
                background=[("active", active), ("pressed", active)],
                foreground=[("active", "white"), ("pressed", "white")],
            )

    def _build_ui(self):
        """Build the canvas and the control toolbar."""
        self._configure_button_styles()
        # Canvas
        self.canvas = tk.Canvas(
            self,
            width=self.photo_img.width(),
            height=self.photo_img.height(),
            highlightthickness=0,
            cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._bg_image_id = self.canvas.create_image(0, 0, image=self.photo_img, anchor=tk.NW)

        # Control Toolbar
        self.toolbar = tk.Frame(self.canvas, bg=COLORS["bg_card"], bd=2, relief=tk.RAISED)
        self.toolbar.place(relx=0.5, y=20, anchor=tk.N)

        tk.Label(
            self.toolbar,
            text=f"Calibrating rod:  {self.profile.name}",
            fg=COLORS["accent_blue"], bg=COLORS["bg_card"],
            font=("Helvetica Neue", -12),
        ).pack(side=tk.TOP, pady=(8, 0), padx=20)

        # Instruction Label
        self.instruction_label = tk.Label(
            self.toolbar,
            text="Pick a step below. Work through 1–3, then press Save & Close.",
            fg=COLORS["text"], bg=COLORS["bg_card"],
            font=("Helvetica Neue", -16, "bold")
        )
        self.instruction_label.pack(side=tk.TOP, pady=(4, 6), padx=20)

        btn_frame = tk.Frame(self.toolbar, bg=COLORS["bg_card"])
        btn_frame.pack(side=tk.TOP, pady=(0, 8), padx=10)

        self.buttons = {}
        for mode, label, _hint in self.STEPS:
            btn = ttk.Button(
                btn_frame, text=label, cursor="hand2",
                style=f"{self._step_style(mode)}.TButton",
                command=lambda m=mode: self._set_mode(m),
            )
            btn.pack(side=tk.LEFT, padx=3)
            self.buttons[mode] = btn

        action_frame = tk.Frame(self.toolbar, bg=COLORS["bg_card"])
        action_frame.pack(side=tk.TOP, pady=(0, 8), padx=10)

        ttk.Button(
            action_frame, text="Retake screenshot", cursor="hand2",
            style="CalRetake.TButton", command=self._retake_screenshot
        ).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            action_frame, text="✓ Save & Close", cursor="hand2",
            style="CalSave.TButton", command=self._save_and_close
        ).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            action_frame, text="✕ Cancel", cursor="hand2",
            style="CalCancel.TButton", command=self._cancel
        ).pack(side=tk.LEFT, padx=3)

        # Checklist: what this session has changed, and what is still untouched.
        self.checklist_label = tk.Label(
            self.toolbar, text="", fg="#a0aabf", bg=COLORS["bg_card"],
            font=("Menlo", -11),
        )
        self.checklist_label.pack(side=tk.TOP, pady=(0, 4))

        # Info readout
        self.info_label = tk.Label(
            self.toolbar, text="", fg=COLORS["accent_green"], bg=COLORS["bg_card"],
            font=("Helvetica Neue", -12)
        )
        self.info_label.pack(side=tk.TOP, pady=(0, 8))

        self._refresh_checklist()

    @staticmethod
    def _step_style(mode: str, selected: bool = False) -> str:
        """Which button style a step wears; the selected one is darker."""
        return "CalRoiOn" if selected else "CalRoi"

    def _refresh_checklist(self):
        """Redraw the ✓/· line and re-mark the step buttons."""
        marks = []
        for mode, label, _hint in self.STEPS:
            done = mode in self.completed
            marks.append(("✓ " if done else "·  ") + label.split(" · ")[1])
            # The selected step is shown by a lighter shade of its own colour.
            # It used to be a SUNKEN relief and a bolder font, neither of which
            # a ttk button takes -- and the relief was invisible anyway on a
            # button macOS was drawing itself.
            self.buttons[mode].config(
                text=("✓ " if done else "") + label,
                style=f"{self._step_style(mode, selected=mode == self.mode)}.TButton",
            )
        self.checklist_label.config(text="   ".join(marks))

    def _cancel(self):
        """Close without saving, warning first if there is anything to lose."""
        if self.completed and not messagebox.askyesno(
            "Discard Calibration",
            "Close without saving?\n\n"
            f"{len(self.completed)} step(s) you changed will be discarded.",
            parent=self,
        ):
            return
        self.destroy()

    def _draw_existing_rois(self):
        """Draw current saved ROI boxes as a starting point."""
        window_bounds = self.window_bounds
        if not window_bounds:
            return

        roi_styles = {
            "bar_roi": (self.settings.bar_roi, "green"),
            "progress_roi": (self.settings.progress_roi, "blue"),
            "shake_roi": (self.settings.shake_roi, "red"),
        }

        settings = self.settings
        inset_left = window_bounds.width * settings.window_inset_left
        inset_top = window_bounds.height * settings.window_inset_top
        eff_x = window_bounds.x + inset_left
        eff_y = window_bounds.y + inset_top
        eff_w = max(1, window_bounds.width - inset_left)
        eff_h = max(1, window_bounds.height - inset_top)

        for mode, (roi, color) in roi_styles.items():
            xs = min(1.0, max(0.0, roi.x_start + settings.roi_shift_x))
            xe = min(1.0, max(0.0, roi.x_end + settings.roi_shift_x))
            ys = min(1.0, max(0.0, roi.y_start + settings.roi_shift_y))
            ye = min(1.0, max(0.0, roi.y_end + settings.roi_shift_y))
            x0 = eff_x + eff_w * xs - self.monitor_left
            y0 = eff_y + eff_h * ys - self.monitor_top
            x1 = eff_x + eff_w * xe - self.monitor_left
            y1 = eff_y + eff_h * ye - self.monitor_top
            rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=color, width=2, dash=(6, 4)
            )
            self.drawn_rects[mode] = rect_id

    def _logical_monitor_left(self):
        return self.monitor_left

    def _logical_monitor_top(self):
        return self.monitor_top

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        # Escape is Cancel, confirmation and all — it used to destroy the
        # window outright, which threw away picked colours with no warning.
        self.bind("<Escape>", lambda _e: self._cancel())

    def _set_mode(self, mode):
        """Select a calibration step and show its instruction."""
        self.mode = mode
        for step_mode, _label, hint in self.STEPS:
            if step_mode == mode:
                self.instruction_label.config(text=hint)
                break
        self._refresh_checklist()

    def _on_press(self, event):
        if not self.mode:
            return

        if self.mode.endswith("_roi"):
            # ROI Drawing
            self.rect_start = (event.x, event.y)

            # Clear previous rect for this mode if it exists
            if self.mode in self.drawn_rects:
                self.canvas.delete(self.drawn_rects[self.mode])

            color = "red" if "shake" in self.mode else "green" if "bar" in self.mode else "blue"
            self.current_rect_id = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline=color, width=3
            )
            self.drawn_rects[self.mode] = self.current_rect_id

    def _on_drag(self, event):
        if not self.mode or not self.mode.endswith("_roi") or not self.rect_start:
            return

        x0, y0 = self.rect_start
        self.canvas.coords(self.current_rect_id, x0, y0, event.x, event.y)

    def _on_release(self, event):
        try:
            self._handle_release(event)
        except Exception as exc:
            logger.error("Interactive calibration release failed: %s", exc, exc_info=True)
            self.info_label.config(text=f"Could not save region: {exc}")
            messagebox.showerror(
                "Calibration Error",
                f"Could not save that region.\n\n{exc}",
                parent=self,
            )

    def _handle_release(self, event):
        if not self.mode or not self.mode.endswith("_roi") or not self.rect_start:
            return

        x0, y0 = self.rect_start
        x1, y1 = event.x, event.y

        # Ensure x0,y0 is top-left and x1,y1 is bottom-right
        rx = min(x0, x1)
        ry = min(y0, y1)
        rw = max(x0, x1) - rx
        rh = max(y0, y1) - ry

        self.rect_start = None

        # Only a box with no area at all is rejected — that is a stray click,
        # not a region. There used to be a 10px floor on both sides, which made
        # the genuinely thin regions impossible to mark: the catch-progress bar
        # is a few pixels tall, and every attempt at it was thrown away with
        # "too small". Capture pads the progress ROI vertically anyway (see
        # Detector.padded_progress_roi), so a tight box is the right box.
        if rw < 1 or rh < 1:
            self.info_label.config(text="That was a click, not a box. Drag to draw one.")
            self.canvas.delete(self.current_rect_id)
            self.drawn_rects.pop(self.mode, None)
            return

        # Convert to normalized bounds
        window_bounds = self.window_bounds or self.window_tracker.get_roblox_bounds()
        if not window_bounds:
            messagebox.showerror("Error", "Could not find Roblox window bounds. Make sure Roblox is visible on screen.")
            self.canvas.delete(self.current_rect_id)
            self.drawn_rects.pop(self.mode, None)
            return

        # Use calibrator utility to compute normalized bounds
        from src.calibrator import Calibrator
        calibrator = Calibrator(self.engine.detector, self.config)

        screen_rect = (
            int(rx + self._logical_monitor_left()),
            int(ry + self._logical_monitor_top()),
            int(rw),
            int(rh),
        )
        norm_bounds = calibrator.compute_roi_from_rect(screen_rect, window_bounds)
        if norm_bounds["x_end"] <= norm_bounds["x_start"] or norm_bounds["y_end"] <= norm_bounds["y_start"]:
            raise ValueError("The box must overlap the Roblox window. Draw the box inside the Roblox game area.")

        # Take the ROI shift back out. Capture applies it (see
        # Detector._compute_roi_pixels) and _draw_existing_rois draws the
        # existing boxes with it applied, but the conversion above is purely
        # window-relative — so storing its result verbatim moved every freshly
        # drawn box by the shift, away from where the user drew it.
        roi = ROIBounds(
            x_start=_clamp01(norm_bounds["x_start"] - self.settings.roi_shift_x),
            x_end=_clamp01(norm_bounds["x_end"] - self.settings.roi_shift_x),
            y_start=_clamp01(norm_bounds["y_start"] - self.settings.roi_shift_y),
            y_end=_clamp01(norm_bounds["y_end"] - self.settings.roi_shift_y),
        )

        label = dict((m, l) for m, l, _h in self.STEPS).get(self.mode, self.mode)

        if self.mode == "bar_roi":
            self.settings.bar_roi = roi
            threshold = self._estimate_active_bar_threshold(rx, ry, rw, rh)
            if threshold is not None:
                self.profile.bar_brightness_threshold = threshold
        elif self.mode == "shake_roi":
            self.settings.shake_roi = roi
        elif self.mode == "progress_roi":
            self.settings.progress_roi = roi

        extra = ""
        if self.mode == "bar_roi":
            extra = f"  ·  brightness threshold {self.profile.bar_brightness_threshold}"
        self.info_label.config(
            text=f"{label} set — x {roi.x_start * 100:.1f}–{roi.x_end * 100:.1f}%"
                 f"  y {roi.y_start * 100:.1f}–{roi.y_end * 100:.1f}%{extra}"
        )
        self.completed.add(self.mode)
        self._refresh_checklist()

    def _estimate_active_bar_threshold(self, x, y, width, height):
        """Estimate brightness threshold from the user-selected active bar ROI."""
        px = int(x * self.scale_factor)
        py = int(y * self.scale_factor)
        pw = int(width * self.scale_factor)
        ph = int(height * self.scale_factor)

        frame_h, frame_w = self.raw_bgr.shape[:2]
        px = max(0, min(px, frame_w - 1))
        py = max(0, min(py, frame_h - 1))
        pw = max(1, min(pw, frame_w - px))
        ph = max(1, min(ph, frame_h - py))

        roi_frame = self.raw_bgr[py:py + ph, px:px + pw]
        if roi_frame.size == 0:
            return None

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        return int(np.clip(mean_brightness + 35, 50, 180))

    def _save_and_close(self):
        """Write both the working regions and the rod profile, then close.

        The regions go to settings.json (what the macro captures from) *and*
        to the rod profile (what switching back to this rod restores), so the
        two cannot drift apart. Colours were written into ``self.profile`` as
        they were picked; nothing reaches disk until here, which is what makes
        Cancel able to discard them.
        """
        if not self.completed and not messagebox.askyesno(
            "Nothing Changed",
            "No steps were changed in this session.\n\nSave anyway?",
            parent=self,
        ):
            return

        self.profile.bar_roi = self.settings.bar_roi
        self.profile.progress_roi = self.settings.progress_roi
        self.profile.shake_roi = self.settings.shake_roi

        self.config.save_settings(self.settings)
        self.config.save_profile(self.profile)
        logger.info(
            "Calibration saved to rod %r (%d step(s) changed)",
            self.profile.name, len(self.completed),
        )
        self.destroy()
