import logging
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

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

class InteractiveCalibrator(tk.Toplevel):
    """
    A full-screen interactive calibration window.
    Takes a screenshot and allows the user to pick colors and draw ROIs.
    """

    def __init__(self, parent, engine, config_manager, window_tracker):
        super().__init__(parent)
        self.engine = engine
        self.config = config_manager
        self.window_tracker = window_tracker
        self.settings = self.config.load_settings()
        self.profile = self.config.get_active_profile()
        self.window_bounds = self.window_tracker.get_roblox_bounds()
        self.monitor_left = 0
        self.monitor_top = 0

        self.title("Interactive Calibration")
        self.attributes("-fullscreen", True)
        self.attributes("-topmost", True)
        self.configure(bg="black")
        
        # Calibration state
        self.mode = None  # 'fish_color', 'bar_color', 'bar_roi', 'shake_roi', 'progress_roi'
        self.rect_start = None
        self.current_rect_id = None
        self.drawn_rects = {}  # Store rect IDs for the different ROIs
        
        self._take_screenshot()
        self._build_ui()
        self._bind_events()
        self._draw_existing_rois()

    def _take_screenshot(self):
        """Take a full screen screenshot and prepare it for the canvas."""
        self.scale_factor = self.window_tracker.get_scale_factor()
        
        # Use mss to capture
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # primary monitor
            self.monitor_left = int(monitor.get("left", 0))
            self.monitor_top = int(monitor.get("top", 0))
            sct_img = sct.grab(monitor)
            self.raw_bgr = np.array(sct_img)
            self.raw_bgr = cv2.cvtColor(self.raw_bgr, cv2.COLOR_BGRA2BGR)
            
        # Resize to logical coordinates for Tkinter display
        h, w = self.raw_bgr.shape[:2]
        logical_w = int(w / self.scale_factor)
        logical_h = int(h / self.scale_factor)
        
        # Convert to PIL and resize
        rgb_img = cv2.cvtColor(self.raw_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Only resize if scale factor is > 1
        if self.scale_factor > 1.0:
            pil_img = pil_img.resize((logical_w, logical_h), Image.Resampling.LANCZOS)
            
        self.photo_img = ImageTk.PhotoImage(pil_img)

    def _build_ui(self):
        """Build the canvas and the control toolbar."""
        # Canvas
        self.canvas = tk.Canvas(
            self, 
            width=self.photo_img.width(), 
            height=self.photo_img.height(), 
            highlightthickness=0,
            cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, image=self.photo_img, anchor=tk.NW)
        
        # Control Toolbar
        self.toolbar = tk.Frame(self.canvas, bg=COLORS["bg_card"], bd=2, relief=tk.RAISED)
        self.toolbar.place(relx=0.5, y=20, anchor=tk.N)
        
        # Instruction Label
        self.instruction_label = tk.Label(
            self.toolbar, 
            text="Select a tool below to begin calibration.",
            fg=COLORS["text"], bg=COLORS["bg_card"],
            font=("Helvetica Neue", 16, "bold")
        )
        self.instruction_label.pack(side=tk.TOP, pady=(10, 5), padx=20)
        
        btn_frame = tk.Frame(self.toolbar, bg=COLORS["bg_card"])
        btn_frame.pack(side=tk.TOP, pady=(0, 10), padx=10)
        
        # Buttons
        self.buttons = {}
        
        def create_btn(text, mode, color):
            btn = tk.Button(
                btn_frame, text=text, font=("Helvetica Neue", 12),
                bg=color, fg="white", cursor="hand2",
                command=lambda: self._set_mode(mode)
            )
            btn.pack(side=tk.LEFT, padx=3)
            self.buttons[mode] = btn
            return btn
            
        create_btn("1. Fish Color", "fish_color", COLORS["accent"])
        create_btn("2. On-Target Bar", "on_target_color", COLORS["accent_green"])
        create_btn("3. Off-Target Bar", "off_target_color", COLORS["accent_yellow"])
        create_btn("4. Bar Bounds", "bar_roi", COLORS["accent_blue"])
        create_btn("5. Progress", "progress_roi", COLORS["accent_blue"])
        create_btn("6. Shake", "shake_roi", COLORS["accent_blue"])
        
        tk.Label(btn_frame, text=" | ", bg=COLORS["bg_card"], fg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame, text="Save & Close", font=("Helvetica Neue", 12, "bold"),
            bg="#28a745", fg="white", cursor="hand2", command=self._save_and_close
        ).pack(side=tk.LEFT, padx=3)
        
        tk.Button(
            btn_frame, text="Cancel", font=("Helvetica Neue", 12),
            bg="#dc3545", fg="white", cursor="hand2", command=self.destroy
        ).pack(side=tk.LEFT, padx=3)
        
        # Info readout
        self.info_label = tk.Label(
            self.toolbar, text="", fg="#a0aabf", bg=COLORS["bg_card"], font=("Helvetica Neue", 12)
        )
        self.info_label.pack(side=tk.TOP, pady=(0, 5))

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

        for mode, (roi, color) in roi_styles.items():
            # Coordinates are absolute logical screen points
            x0 = window_bounds.x + window_bounds.width * roi.x_start
            y0 = window_bounds.y + window_bounds.height * roi.y_start
            x1 = window_bounds.x + window_bounds.width * roi.x_end
            y1 = window_bounds.y + window_bounds.height * roi.y_end
            
            # Since the window is fullscreen on the primary monitor, 
            # absolute screen points == canvas points.
            rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=color, width=2, dash=(6, 4)
            )
            self.drawn_rects[mode] = rect_id

    def _logical_monitor_left(self):
        # We assume the calibration window is on the primary monitor (0,0)
        return 0

    def _logical_monitor_top(self):
        return 0

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        # Also bind escape to cancel
        self.bind("<Escape>", lambda e: self.destroy())

    def _set_mode(self, mode):
        self.mode = mode
        
        # Reset button styles
        for m, btn in self.buttons.items():
            btn.config(relief=tk.RAISED, font=("Helvetica Neue", 12))
            
        # Highlight active button
        if mode in self.buttons:
            self.buttons[mode].config(relief=tk.SUNKEN, font=("Helvetica Neue", 12, "bold"))
            
        # Update instructions
        if mode == "fish_color":
            self.instruction_label.config(text="1. PICK FISH: Click exactly on the PINK fish icon in the bar.")
        elif mode == "on_target_color":
            self.instruction_label.config(text="2. ON-TARGET: Click the bar when it turns GREEN (on the fish).")
        elif mode == "off_target_color":
            self.instruction_label.config(text="3. OFF-TARGET: Click the bar when it is WHITE/ORANGE (off the fish).")
        elif mode == "bar_roi":
            self.instruction_label.config(text="4. BAR AREA: Draw a box covering the entire slider track.")
        elif mode == "progress_roi":
            self.instruction_label.config(text="5. PROGRESS: Draw a box over the catch progress bar at the bottom.")
        elif mode == "shake_roi":
            self.instruction_label.config(text="6. SHAKE: Draw a large box over the middle where 'SHAKE' buttons appear.")
        elif mode.endswith("_roi"):
            name = mode.split("_")[0].capitalize()
            self.instruction_label.config(text=f"Click and drag to draw a box around the {name} region.")

    def _on_press(self, event):
        if not self.mode:
            return
            
        if self.mode.endswith("_color"):
            # Color picking
            physical_x = int(event.x * self.scale_factor)
            physical_y = int(event.y * self.scale_factor)
            
            # Ensure within bounds
            h, w = self.raw_bgr.shape[:2]
            if 0 <= physical_x < w and 0 <= physical_y < h:
                bgr = self.raw_bgr[physical_y, physical_x]
                hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                
                h_val, s_val, v_val = hsv
                
                # Dynamic padding based on mode
                if self.mode == "fish_color":
                    # Pink fish needs specific hue but can be bright
                    h_low, h_high = max(0, int(h_val)-20), min(179, int(h_val)+20)
                    s_low, s_high = max(60, int(s_val)-60), min(255, int(s_val)+60)
                    v_low, v_high = max(50, int(v_val)-60), min(255, int(v_val)+60)
                else:
                    # Bar colors (green/white/orange)
                    h_low, h_high = max(0, int(h_val)-15), min(179, int(h_val)+15)
                    s_low, s_high = max(40, int(s_val)-50), min(255, int(s_val)+50)
                    v_low, v_high = max(40, int(v_val)-50), min(255, int(v_val)+50)
                
                low_arr = [h_low, s_low, v_low]
                high_arr = [h_high, s_high, v_high]
                
                if self.mode == "fish_color":
                    self.profile.fish_hsv_low = low_arr
                    self.profile.fish_hsv_high = high_arr
                    self.info_label.config(text=f"Fish Color Saved! HSV: {low_arr} to {high_arr}")
                elif self.mode == "on_target_color":
                    self.profile.on_target_hsv_low = low_arr
                    self.profile.on_target_hsv_high = high_arr
                    # Also update the general bar color for backward compatibility
                    self.profile.bar_hsv_low = low_arr
                    self.profile.bar_hsv_high = high_arr
                    self.info_label.config(text=f"On-Target Color Saved! HSV: {low_arr} to {high_arr}")
                elif self.mode == "off_target_color":
                    self.profile.off_target_hsv_low = low_arr
                    self.profile.off_target_hsv_high = high_arr
                    self.info_label.config(text=f"Off-Target Color Saved! HSV: {low_arr} to {high_arr}")
        
        elif self.mode.endswith("_roi"):
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
        
        if rw < 10 or rh < 10:
            self.info_label.config(text="Drawn region too small. Try again.")
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

        roi = ROIBounds(**norm_bounds)
        
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
            extra = f", brightness threshold: {self.profile.bar_brightness_threshold}"
        self.info_label.config(
            text=f"Saved {self.mode}! X: {roi.x_start:.2f}-{roi.x_end:.2f}, Y: {roi.y_start:.2f}-{roi.y_end:.2f}{extra}"
        )

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
        """Save settings and profile, then destroy the window."""
        self.profile.bar_roi = self.settings.bar_roi
        self.profile.progress_roi = self.settings.progress_roi
        self.profile.shake_roi = self.settings.shake_roi
        
        # New: color profile persistence is already done in _on_press directly to self.profile
        self.config.save_settings(self.settings)
        self.config.save_profile(self.profile)
        self.destroy()
