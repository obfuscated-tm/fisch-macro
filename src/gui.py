"""
Fisch Macro GUI — tkinter Control Panel

Provides a compact, always-on-top window with:
- Control tab: Start/Stop, status, stats
- Settings tab: Timing sliders, toggles
- Calibration tab: ROI + color profile management
- Debug tab: Live detection preview
"""

import logging
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

logger = logging.getLogger("gui")


# ─── Color Palette ─────────────────────────────────────────────────
COLORS = {
    "bg": "#1a1a2e",
    "bg_secondary": "#16213e",
    "bg_card": "#183b63",  # Brighter card background
    "accent": "#e94560",
    "accent_green": "#00d474",
    "accent_yellow": "#f5a623",
    "accent_blue": "#4fc3f7",
    "text": "#ffffff",    # Brighter text
    "text_dim": "#b0bacb", # Brighter dim text
    "border": "#2a2a4a",
    "success": "#00d474",
    "danger": "#e94560",
    "warning": "#f5a623",
    # Settings panel — light card surface so black text is readable
    "settings_bg": "#eef1f5",
    "settings_text": "#000000",
    "settings_dim": "#444444",
}

STATE_COLORS = {
    "Idle": COLORS["text_dim"],
    "Casting": COLORS["accent_blue"],
    "Waiting for Bite": COLORS["accent_yellow"],
    "Shaking": COLORS["accent_yellow"],
    "Reeling": COLORS["accent"],
    "Catch Complete": COLORS["accent_green"],
    "Stopped": COLORS["danger"],
}

STATE_ICONS = {
    "Idle": "⏸",
    "Casting": "🎣",
    "Waiting for Bite": "💤",
    "Shaking": "👆",
    "Reeling": "🐟",
    "Catch Complete": "✅",
    "Stopped": "🔴",
}


class MacroGUI:
    """
    Main GUI window for the Fisch macro.

    Handles user interaction and displays macro status/stats.
    All GUI updates are thread-safe via root.after().
    """

    def __init__(self, macro_engine, config_manager, window_tracker):
        """
        Args:
            macro_engine: MacroEngine instance
            config_manager: ConfigManager instance
            window_tracker: WindowTracker instance
        """
        self.engine = macro_engine
        self.config = config_manager
        self.window_tracker = window_tracker
        self.settings = self.config.load_settings()
        self._last_hotkey_toggle = 0.0

        # Build the GUI
        self.root = tk.Tk()

        # Initialize common variables (must be after tk.Tk())
        self.profile_var = tk.StringVar(value=self.settings.active_profile)
        self.root.title("Fisch Macro")
        self.root.geometry("440x780")
        self.root.minsize(400, 640)
        self._vision_photo = None
        self.root.configure(bg=COLORS["bg"])
        self.root.attributes("-topmost", True)

        # Try to set the window style
        try:
            self.root.tk.call("tk", "scaling", 1.0)
        except Exception:
            pass

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self._configure_styles()

        # Build UI components
        self._build_header()
        self._build_notebook()
        self._build_control_tab()
        self._build_settings_tab()
        self._build_calibration_tab()
        self._build_log_tab()

        # Register callbacks with the macro engine
        self.engine.on_state_change(self._on_state_change)
        self.engine.on_stats_update(self._on_stats_update)
        self.engine.on_log(self._on_log_message)
        self.engine.controller.on_hotkey(
            lambda: self.root.after(0, self._toggle_from_hotkey)
        )
        self.root.bind_all("<KeyRelease-F6>", lambda _event: self._toggle_from_hotkey())

        # Auto-save setup
        self._setup_auto_save()

        # Periodic UI update
        self._update_interval = 500  # ms
        self._vision_interval = 80  # ms
        self._schedule_update()
        self._schedule_vision_update()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_auto_save(self):
        """Add traces to all settings variables for automatic saving."""
        for var in [
            self.cast_time_var,
            self.recast_delay_var,
            self.scan_interval_var,
            self.auto_recast_var,
            self.shake_enabled_var,
            self.profile_var,
            self.fish_prediction_ms_var,
            self.prediction_weight_var,
            self.fish_velocity_smoothing_var,
            self.bar_velocity_smoothing_var,
            self.bar_momentum_factor_var,
            self.control_kp_var,
            self.control_kd_var,
            self.reeling_guard_var,
            self.min_catch_seconds_var,
            self.fast_catch_min_seconds_var,
            self.post_catch_lockout_var,
            self.min_midgame_progress_var,
            self.finish_progress_threshold_var,
            self.roi_shift_x_var,
            self.roi_shift_y_var,
            self.window_inset_top_var,
            self.window_inset_left_var,
            self.prediction_use_accel_var,
            self.prediction_arrival_lead_var,
            self.show_live_vision_var,
            self.off_target_chase_var,
        ]:
            var.trace_add("write", lambda *args: self._save_settings())

    # ─── Style Configuration ──────────────────────────────────────

    def _configure_styles(self):
        """Configure ttk styles for a dark modern look."""
        s = self.style

        s.configure(".", background=COLORS["bg"], foreground=COLORS["text"])
        s.configure("TFrame", background=COLORS["bg"])
        s.configure("Card.TFrame", background=COLORS["bg_card"])

        s.configure(
            "TNotebook",
            background=COLORS["bg"],
            borderwidth=0,
        )
        s.configure(
            "TNotebook.Tab",
            background=COLORS["bg_secondary"],
            foreground=COLORS["text_dim"],
            padding=[12, 6],
            borderwidth=0,
        )
        s.map(
            "TNotebook.Tab",
            background=[("selected", COLORS["bg_card"])],
            foreground=[("selected", COLORS["text"])],
        )

        s.configure(
            "TLabel",
            background=COLORS["bg"],
            foreground=COLORS["text"],
            font=("Helvetica Neue", 13),
        )
        s.configure(
            "Header.TLabel",
            font=("Helvetica Neue", 22, "bold"),
            foreground=COLORS["text"],
        )
        s.configure(
            "Status.TLabel",
            font=("Helvetica Neue", 15, "bold"),
            foreground=COLORS["accent"],
        )
        s.configure(
            "Stat.TLabel",
            font=("Menlo", 14),
            foreground=COLORS["accent_blue"],
        )
        s.configure(
            "Dim.TLabel",
            font=("Helvetica Neue", 12),
            foreground=COLORS["text_dim"],
        )
        s.configure(
            "Card.TLabel",
            background=COLORS["bg_card"],
            foreground=COLORS["text"],
            font=("Helvetica Neue", 13),
        )
        s.configure(
            "CardDim.TLabel",
            background=COLORS["bg_card"],
            foreground=COLORS["text_dim"],
            font=("Helvetica Neue", 12),
        )
        s.configure(
            "CardStat.TLabel",
            background=COLORS["bg_card"],
            foreground=COLORS["accent_blue"],
            font=("Menlo", 16, "bold"),
        )

        s.configure(
            "Start.TButton",
            font=("Helvetica Neue", 14, "bold"),
            padding=[20, 10],
        )
        s.configure(
            "TScale",
            background=COLORS["bg"],
            troughcolor=COLORS["bg_secondary"],
        )
        s.configure(
            "TCheckbutton",
            background=COLORS["bg"],
            foreground=COLORS["text"],
            font=("Helvetica Neue", 12),
        )

        # ── Settings panel — light card with black text ──
        s.configure(
            "Settings.TFrame",
            background=COLORS["settings_bg"],
        )
        s.configure(
            "Settings.TLabel",
            background=COLORS["settings_bg"],
            foreground=COLORS["settings_text"],
            font=("Helvetica Neue", 13),
        )
        s.configure(
            "SettingsDim.TLabel",
            background=COLORS["settings_bg"],
            foreground=COLORS["settings_dim"],
            font=("Helvetica Neue", 10, "bold"),
        )
        s.configure(
            "SettingsStat.TLabel",
            background=COLORS["settings_bg"],
            foreground=COLORS["settings_text"],
            font=("Menlo", 12),
        )
        s.configure(
            "Settings.TScale",
            background=COLORS["settings_bg"],
            troughcolor=COLORS["border"],
        )
        s.layout(
            "Horizontal.Settings.TScale",
            s.layout("Horizontal.TScale"),
        )
        s.layout(
            "Vertical.Settings.TScale",
            s.layout("Vertical.TScale"),
        )
        s.configure(
            "Settings.TCheckbutton",
            background=COLORS["settings_bg"],
            foreground=COLORS["settings_text"],
            font=("Helvetica Neue", 12),
        )

    # ─── Header ───────────────────────────────────────────────────

    def _build_header(self):
        """Build the title header with hotkey info."""
        header = ttk.Frame(self.root, style="TFrame")
        header.pack(fill=tk.X, padx=16, pady=(12, 4))

        title_frame = ttk.Frame(header)
        title_frame.pack(fill=tk.X)

        ttk.Label(
            title_frame, text="🐟 Fisch Macro", style="Header.TLabel"
        ).pack(side=tk.LEFT)

        self.header_killswitch_label = ttk.Label(
            title_frame,
            text=f"⚡ Toggle: {self.settings.killswitch_key.upper()}",
            style="Dim.TLabel",
        )
        self.header_killswitch_label.pack(side=tk.RIGHT)

    # ─── Notebook (Tabs) ──────────────────────────────────────────

    def _build_notebook(self):
        """Build the tabbed notebook."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))

        self.control_tab = ttk.Frame(self.notebook, style="TFrame")
        self.settings_tab = ttk.Frame(self.notebook, style="TFrame")
        self.calibration_tab = ttk.Frame(self.notebook, style="TFrame")
        self.log_tab = ttk.Frame(self.notebook, style="TFrame")

        self.notebook.add(self.control_tab, text="  Control  ")
        self.notebook.add(self.settings_tab, text="  Settings  ")
        self.notebook.add(self.calibration_tab, text="  Calibrate  ")
        self.notebook.add(self.log_tab, text="  Log  ")

    # ─── Control Tab ──────────────────────────────────────────────

    def _build_control_tab(self):
        """Build the main control panel."""
        tab = self.control_tab

        # ── Status Section ──
        status_frame = ttk.Frame(tab, style="Card.TFrame")
        status_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        # Status icon and text
        status_inner = ttk.Frame(status_frame, style="Card.TFrame")
        status_inner.pack(fill=tk.X, padx=16, pady=12)

        self.status_icon = ttk.Label(
            status_inner, text="⏸", style="Card.TLabel",
            font=("Helvetica Neue", 28),
        )
        self.status_icon.pack(side=tk.LEFT, padx=(0, 12))

        status_text = ttk.Frame(status_inner, style="Card.TFrame")
        status_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_label = ttk.Label(
            status_text, text="Stopped", style="Card.TLabel",
            font=("Helvetica Neue", 16, "bold"),
        )
        self.status_label.pack(anchor=tk.W)

        self.status_detail = ttk.Label(
            status_text, text="Press Start to begin fishing",
            style="CardDim.TLabel",
        )
        self.status_detail.pack(anchor=tk.W)

        self.on_target_label = ttk.Label(
            status_text, text="",
            style="Card.TLabel", font=("Helvetica Neue", 11, "bold")
        )
        self.on_target_label.pack(anchor=tk.W)

        # ── Live vision (digmacro-style preview) ──
        vision_card = ttk.Frame(tab, style="Card.TFrame")
        vision_card.pack(fill=tk.X, padx=8, pady=(4, 4))

        vision_inner = ttk.Frame(vision_card, style="Card.TFrame")
        vision_inner.pack(fill=tk.X, padx=12, pady=10)

        ttk.Label(
            vision_inner,
            text="👁 Live Vision",
            style="Card.TLabel",
            font=("Helvetica Neue", 12, "bold"),
        ).pack(anchor=tk.W)

        self.vision_image_label = tk.Label(
            vision_inner,
            text="Start macro to see detection…",
            bg="#0a0a14",
            fg=COLORS["text"],
            font=("Menlo", 10),
            height=4,
        )
        self.vision_image_label.pack(fill=tk.X)

        self._build_vision_key(vision_inner)

        self.vision_telemetry = tk.Text(
            vision_inner,
            height=6,
            bg="#0a0a14",
            fg="#a8d4ff",
            font=("Menlo", 9),
            relief=tk.FLAT,
            wrap=tk.WORD,
        )
        self.vision_telemetry.pack(fill=tk.X, pady=(6, 0))
        self.vision_telemetry.config(state=tk.DISABLED)

        # ── Start/Stop Button ──
        self.start_button = tk.Button(
            tab,
            text="▶  START",
            font=("Helvetica Neue", 15, "bold"),
            bg=COLORS["accent_green"],
            fg="#ffffff",
            activebackground="#00b563",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            height=2,
            command=self._toggle_macro,
        )
        self.start_button.pack(fill=tk.X, padx=8, pady=8)

        self.kill_button = tk.Button(
            tab,
            text="EMERGENCY STOP",
            font=("Helvetica Neue", 13, "bold"),
            bg=COLORS["danger"],
            fg="#ffffff",
            activebackground="#b83045",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            height=1,
            command=self._emergency_stop,
        )
        self.kill_button.pack(fill=tk.X, padx=8, pady=(0, 8))

        # ── Stats Grid ──
        stats_frame = ttk.Frame(tab, style="Card.TFrame")
        stats_frame.pack(fill=tk.X, padx=8, pady=4)

        stats_inner = ttk.Frame(stats_frame, style="Card.TFrame")
        stats_inner.pack(fill=tk.X, padx=16, pady=12)
        stats_inner.columnconfigure((0, 1, 2, 3), weight=1)

        # Row: labels
        for i, label in enumerate(["Caught", "Failed", "Casts", "Rate"]):
            ttk.Label(
                stats_inner, text=label, style="CardDim.TLabel"
            ).grid(row=0, column=i, sticky=tk.N)

        # Row: values
        self.stat_caught = ttk.Label(stats_inner, text="0", style="CardStat.TLabel")
        self.stat_caught.grid(row=1, column=0, pady=(2, 0))

        self.stat_failed = ttk.Label(stats_inner, text="0", style="CardStat.TLabel")
        self.stat_failed.grid(row=1, column=1, pady=(2, 0))

        self.stat_casts = ttk.Label(stats_inner, text="0", style="CardStat.TLabel")
        self.stat_casts.grid(row=1, column=2, pady=(2, 0))

        self.stat_rate = ttk.Label(stats_inner, text="—", style="CardStat.TLabel")
        self.stat_rate.grid(row=1, column=3, pady=(2, 0))

        # ── Session Time ──
        time_frame = ttk.Frame(tab, style="Card.TFrame")
        time_frame.pack(fill=tk.X, padx=8, pady=4)

        time_inner = ttk.Frame(time_frame, style="Card.TFrame")
        time_inner.pack(fill=tk.X, padx=16, pady=8)

        ttk.Label(time_inner, text="⏱ Session", style="CardDim.TLabel").pack(
            side=tk.LEFT
        )
        self.session_time = ttk.Label(
            time_inner, text="00:00:00", style="Card.TLabel",
            font=("Menlo", 13),
        )
        self.session_time.pack(side=tk.RIGHT)

        # ── Window Status ──
        win_frame = ttk.Frame(tab, style="Card.TFrame")
        win_frame.pack(fill=tk.X, padx=8, pady=4)

        win_inner = ttk.Frame(win_frame, style="Card.TFrame")
        win_inner.pack(fill=tk.X, padx=16, pady=8)

        ttk.Label(win_inner, text="🖥 Roblox Window", style="CardDim.TLabel").pack(
            side=tk.LEFT
        )
        self.window_status = ttk.Label(
            win_inner, text="Searching...", style="Card.TLabel",
            font=("Helvetica Neue", 11),
        )
        self.window_status.pack(side=tk.RIGHT)

    def _build_vision_key(self, parent):
        """Draw the overlay key from the palette the overlay itself uses.

        The key used to be a hand-written sentence and had drifted out of step
        with the drawing: it called the off-target bracket blue when it is
        amber, and never mentioned the aim marker or the progress strip at all.
        Building it from :data:`detector.OVERLAY_LEGEND` means a colour cannot
        change on one side without changing on the other.
        """
        from src.detector import OVERLAY_LEGEND, legend_hex

        key = ttk.Frame(parent, style="Card.TFrame")
        key.pack(fill=tk.X, pady=(6, 0))

        row = None
        for index, (name, label) in enumerate(OVERLAY_LEGEND):
            if index % 2 == 0:
                row = ttk.Frame(key, style="Card.TFrame")
                row.pack(fill=tk.X)

            cell = ttk.Frame(row, style="Card.TFrame")
            cell.pack(side=tk.LEFT, fill=tk.X, expand=True)

            tk.Frame(
                cell,
                bg=legend_hex(name),
                width=12,
                height=4,
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(0, 5), pady=3)

            ttk.Label(
                cell,
                text=label,
                style="CardDim.TLabel",
                font=("Helvetica Neue", 9),
            ).pack(side=tk.LEFT)

    # ─── Settings Tab ─────────────────────────────────────────────

    def _build_settings_tab(self):
        """Build the settings panel with sliders and toggles."""
        tab = self.settings_tab

        # Scrollable content
        canvas = tk.Canvas(tab, bg=COLORS["settings_bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        content = ttk.Frame(canvas, style="Settings.TFrame")

        content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=content, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Timing Section ──
        self._add_section_header(content, "Timing")

        self.cast_time_var = tk.DoubleVar(value=self.settings.cast_hold_time)
        self._add_slider(
            content,
            "Cast Hold Time",
            self.cast_time_var,
            0.5, 4.0, 0.1,
            suffix="s",
        )

        self.recast_delay_var = tk.DoubleVar(value=self.settings.recast_delay)
        self._add_slider(
            content,
            "Recast Delay",
            self.recast_delay_var,
            0.5, 5.0, 0.1,
            suffix="s",
        )

        self.scan_interval_var = tk.IntVar(value=self.settings.scan_interval_ms)
        self._add_slider(
            content,
            "Scan Interval",
            self.scan_interval_var,
            20, 100, 5,
            suffix="ms",
        )

        # ── Toggles Section ──
        self._add_section_header(content, "Automation")

        self.auto_recast_var = tk.BooleanVar(value=self.settings.auto_recast)
        self._add_toggle(content, "Auto Recast", self.auto_recast_var)

        self.shake_enabled_var = tk.BooleanVar(value=self.settings.shake_enabled)
        self._add_toggle(content, "Auto Shake", self.shake_enabled_var)

        # ── Reeling / Prediction ──
        self._add_section_header(content, "Reeling & Prediction")

        self.fish_prediction_ms_var = tk.DoubleVar(value=self.settings.fish_prediction_ms)
        self._add_slider(content, "Fish Lookahead", self.fish_prediction_ms_var, 0, 200, 10, suffix="ms")

        self.prediction_weight_var = tk.DoubleVar(value=self.settings.prediction_weight)
        self._add_slider(content, "Prediction Blend", self.prediction_weight_var, 0.0, 1.0, 0.05)

        self.fish_velocity_smoothing_var = tk.DoubleVar(value=self.settings.fish_velocity_smoothing)
        self._add_slider(content, "Fish Momentum", self.fish_velocity_smoothing_var, 0.10, 0.90, 0.05)

        self.bar_velocity_smoothing_var = tk.DoubleVar(value=self.settings.bar_velocity_smoothing)
        self._add_slider(content, "Bar Drift Smoothing", self.bar_velocity_smoothing_var, 0.10, 0.90, 0.05)

        self.bar_momentum_factor_var = tk.DoubleVar(value=self.settings.bar_momentum_factor)
        self._add_slider(content, "Bar Momentum", self.bar_momentum_factor_var, 0.0, 1.0, 0.05)

        self.control_kp_var = tk.DoubleVar(value=self.settings.control_kp)
        self._add_slider(content, "Control Strength (Kp)", self.control_kp_var, 0.10, 0.60, 0.02)

        self.control_kd_var = tk.DoubleVar(value=self.settings.control_kd)
        self._add_slider(content, "Response (Kd)", self.control_kd_var, 0.50, 3.00, 0.10)

        # ── Catch Detection ──
        self._add_section_header(content, "Catch Detection")

        self.reeling_guard_var = tk.DoubleVar(value=self.settings.reeling_guard_seconds)
        self._add_slider(content, "Reel Start Guard", self.reeling_guard_var, 1.0, 5.0, 0.25, suffix="s")

        self.min_catch_seconds_var = tk.DoubleVar(value=self.settings.min_catch_seconds)
        self._add_slider(content, "Min Catch Time", self.min_catch_seconds_var, 1.0, 10.0, 0.5, suffix="s")

        self.fast_catch_min_seconds_var = tk.DoubleVar(
            value=getattr(self.settings, "fast_catch_min_seconds", 1.0)
        )
        self._add_slider(
            content,
            "Fast Rod Min Reel",
            self.fast_catch_min_seconds_var,
            0.5,
            4.0,
            0.25,
            suffix="s",
        )

        self.post_catch_lockout_var = tk.DoubleVar(
            value=getattr(self.settings, "post_catch_lockout_seconds", 1.5)
        )
        self._add_slider(
            content,
            "Post-Catch Lockout",
            self.post_catch_lockout_var,
            0.5,
            5.0,
            0.25,
            suffix="s",
        )

        self.min_midgame_progress_var = tk.DoubleVar(value=self.settings.min_midgame_progress)
        self._add_slider(content, "Min Progress Before Catch", self.min_midgame_progress_var, 0.20, 0.60, 0.05)

        self.finish_progress_threshold_var = tk.DoubleVar(value=self.settings.finish_progress_threshold)
        self._add_slider(content, "Finish Progress", self.finish_progress_threshold_var, 0.90, 0.99, 0.01)

        # ── Calibration alignment ──
        self._add_section_header(content, "Calibration Alignment")

        self.roi_shift_x_var = tk.DoubleVar(value=self.settings.roi_shift_x)
        self._add_slider(content, "ROI Shift X", self.roi_shift_x_var, -0.08, 0.08, 0.005)

        self.roi_shift_y_var = tk.DoubleVar(value=self.settings.roi_shift_y)
        self._add_slider(content, "ROI Shift Y", self.roi_shift_y_var, -0.08, 0.08, 0.005)

        self.window_inset_top_var = tk.DoubleVar(value=self.settings.window_inset_top)
        self._add_slider(content, "Window Inset Top", self.window_inset_top_var, 0.0, 0.08, 0.005)

        self.window_inset_left_var = tk.DoubleVar(value=self.settings.window_inset_left)
        self._add_slider(content, "Window Inset Left", self.window_inset_left_var, 0.0, 0.08, 0.005)

        self.prediction_use_accel_var = tk.BooleanVar(value=self.settings.prediction_use_acceleration)
        self._add_toggle(content, "Use Acceleration Predict", self.prediction_use_accel_var)

        self.prediction_arrival_lead_var = tk.BooleanVar(value=self.settings.prediction_arrival_lead)
        self._add_toggle(content, "Arrival Lead (digmacro-style)", self.prediction_arrival_lead_var)

        self.show_live_vision_var = tk.BooleanVar(value=self.settings.show_live_vision)
        self._add_toggle(content, "Live Vision Preview", self.show_live_vision_var)

        self.off_target_chase_var = tk.DoubleVar(value=self.settings.off_target_chase_gain)
        self._add_slider(content, "Off-Target Chase Gain", self.off_target_chase_var, 1.0, 2.0, 0.05)

        # ── Hotkey Section ──
        self._add_section_header(content, "Hotkey")

        ks_frame = ttk.Frame(content, style="Settings.TFrame")
        ks_frame.pack(fill=tk.X, padx=12, pady=4)

        ttk.Label(ks_frame, text="Start/Stop Key:", style="Settings.TLabel").pack(side=tk.LEFT)
        self.killswitch_label = ttk.Label(
            ks_frame, text=self.settings.killswitch_key.upper(), style="SettingsStat.TLabel"
        )
        self.killswitch_label.pack(side=tk.LEFT, padx=10)

        rebind_btn = tk.Button(
            ks_frame, text="Rebind", font=("Helvetica Neue", 12),
            bg=COLORS["accent"], fg="white", cursor="hand2",
            command=self._rebind_killswitch
        )
        rebind_btn.pack(side=tk.RIGHT)

    def _add_section_header(self, parent, text):
        """Add a section header label."""
        frame = ttk.Frame(parent, style="Settings.TFrame")
        frame.pack(fill=tk.X, padx=12, pady=(16, 4))
        ttk.Label(
            frame,
            text=text.upper(),
            style="SettingsDim.TLabel",
        ).pack(anchor=tk.W)

    def _add_slider(self, parent, label, variable, from_, to, resolution, suffix=""):
        """Add a labeled slider with value display."""
        frame = ttk.Frame(parent, style="Settings.TFrame")
        frame.pack(fill=tk.X, padx=12, pady=4)

        top = ttk.Frame(frame, style="Settings.TFrame")
        top.pack(fill=tk.X)

        ttk.Label(top, text=label, style="Settings.TLabel").pack(side=tk.LEFT)

        value_label = ttk.Label(
            top,
            text=f"{variable.get()}{suffix}",
            style="SettingsStat.TLabel",
        )
        value_label.pack(side=tk.RIGHT)

        scale = ttk.Scale(
            frame,
            from_=from_,
            to=to,
            variable=variable,
            style="Settings.TScale",
            orient=tk.HORIZONTAL,
        )
        scale.pack(fill=tk.X, pady=(2, 0))

        def update_label(*_args):
            val = variable.get()
            if isinstance(val, float):
                value_label.config(text=f"{val:.1f}{suffix}")
            else:
                value_label.config(text=f"{val}{suffix}")

        variable.trace_add("write", update_label)

    def _add_toggle(self, parent, label, variable):
        """Add a labeled toggle checkbox."""
        frame = ttk.Frame(parent, style="Settings.TFrame")
        frame.pack(fill=tk.X, padx=12, pady=4)
        cb = ttk.Checkbutton(
            frame, text=label, variable=variable, style="Settings.TCheckbutton"
        )
        cb.pack(anchor=tk.W)

    # ─── Calibration Tab ──────────────────────────────────────────

    def _build_calibration_tab(self):
        """Build the calibration panel."""
        tab = self.calibration_tab

        # Profile selector
        profile_frame = ttk.Frame(tab, style="Card.TFrame")
        profile_frame.pack(fill=tk.X, padx=8, pady=8)

        pf_inner = ttk.Frame(profile_frame, style="Card.TFrame")
        pf_inner.pack(fill=tk.X, padx=16, pady=12)

        ttk.Label(pf_inner, text="🎣 Rod Profile", style="Card.TLabel",
                  font=("Helvetica Neue", 13, "bold")).pack(anchor=tk.W)

        selector_frame = ttk.Frame(pf_inner, style="Card.TFrame")
        selector_frame.pack(fill=tk.X, pady=(8, 0))

        profiles = self.config.list_profiles()
        self.profile_combo = ttk.Combobox(
            selector_frame,
            textvariable=self.profile_var,
            values=profiles,
            state="readonly",
            width=20,
        )
        self.profile_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_change)

        delete_profile_btn = tk.Button(
            selector_frame,
            text="🗑",
            font=("Helvetica Neue", 12),
            bg=COLORS["danger"],
            fg="#ffffff",
            activebackground="#c73a50",
            relief=tk.FLAT,
            cursor="hand2",
            width=3,
            command=self._delete_profile,
        )
        delete_profile_btn.pack(side=tk.RIGHT)

        save_as_profile_btn = tk.Button(
            pf_inner,
            text="💾  Save Current Calibration As Rod Profile",
            font=("Helvetica Neue", 12),
            bg=COLORS["accent_blue"],
            fg="#ffffff",
            activebackground="#3aa3d7",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._save_calibration_as_profile,
        )
        save_as_profile_btn.pack(fill=tk.X, pady=(10, 0))

        # HSV Color Preview (informational)
        hsv_frame = ttk.Frame(tab, style="Card.TFrame")
        hsv_frame.pack(fill=tk.X, padx=8, pady=4)

        hsv_inner = ttk.Frame(hsv_frame, style="Card.TFrame")
        hsv_inner.pack(fill=tk.X, padx=16, pady=12)

        ttk.Label(hsv_inner, text="🎨 Color Ranges (HSV)", style="Card.TLabel",
                  font=("Helvetica Neue", 13, "bold")).pack(anchor=tk.W)

        self.hsv_info_label = ttk.Label(
            hsv_inner,
            text="Loading profile colors...",
            style="CardDim.TLabel",
        )
        self.hsv_info_label.pack(anchor=tk.W, pady=(4, 0))
        self._update_hsv_display()

        # Interactive calibrate button
        interactive_cal_btn = tk.Button(
            tab,
            text="👁️  Interactive Calibration (Eyedropper)",
            font=("Helvetica Neue", 12, "bold"),
            bg="#f5a623",
            fg="#ffffff",
            activebackground="#f5b853",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._interactive_calibrate,
        )
        interactive_cal_btn.pack(fill=tk.X, padx=8, pady=4)

        # Save calibration
        save_cal_btn = tk.Button(
            tab,
            text="💾  Save Calibration To Selected Rod",
            font=("Helvetica Neue", 12),
            bg=COLORS["accent_blue"],
            fg="#ffffff",
            activebackground="#3aa3d7",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._save_calibration,
        )
        save_cal_btn.pack(fill=tk.X, padx=8, pady=4)

    # ─── Log Tab ──────────────────────────────────────────────────

    def _build_log_tab(self):
        """Build the activity log panel."""
        tab = self.log_tab

        # Log text area
        log_frame = ttk.Frame(tab)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.log_text = tk.Text(
            log_frame,
            bg=COLORS["bg_secondary"],
            fg=COLORS["text"],
            font=("Menlo", 11),
            relief=tk.FLAT,
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=20,
            padx=8,
            pady=8,
        )
        log_scrollbar = ttk.Scrollbar(
            log_frame, orient=tk.VERTICAL, command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Clear log button
        clear_btn = tk.Button(
            tab,
            text="🗑  Clear Log",
            font=("Helvetica Neue", 11),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_dim"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._clear_log,
        )
        clear_btn.pack(fill=tk.X, padx=8, pady=(0, 8))

    # ─── Actions ──────────────────────────────────────────────────

    def _toggle_macro(self):
        """Start or stop the macro."""
        if self.engine.is_running():
            # Repaint first. The button used to change only after stop()
            # returned, and stop() waited on the worker thread, so pressing
            # Stop looked like nothing had happened for as long as that took.
            self.start_button.config(
                text="▶  START",
                bg=COLORS["accent_green"],
                activebackground="#00b563",
            )
            self.start_button.update_idletasks()
            self.engine.stop()
        else:
            # Update window tracking
            bounds = self.window_tracker.get_roblox_bounds()
            if bounds is None:
                messagebox.showwarning(
                    "Roblox Not Found",
                    "Could not find the Roblox window.\n\n"
                    "Make sure Roblox is open and in WINDOWED mode "
                    "(not fullscreen).",
                )
                return

            scale = self.window_tracker.get_scale_factor()
            self.engine.detector.set_window_info(bounds, scale)

            self._append_log(
                f"Roblox window: {bounds.width}×{bounds.height} "
                f"at ({bounds.x}, {bounds.y}), scale={scale}x"
            )

            self.engine.start()
            self.start_button.config(
                text="⏹  STOP",
                bg=COLORS["danger"],
                activebackground="#c73a50",
            )

    def _toggle_from_hotkey(self):
        """Toggle the macro from F6/global hotkey on the Tk main thread.

        Two paths deliver the same press — the global listener and the Tk
        binding below it — so the first one through wins for a moment. Kept
        short deliberately: see Controller._hotkey_debounce_seconds.
        """
        now = time.monotonic()
        if now - self._last_hotkey_toggle < 0.2:
            return
        self._last_hotkey_toggle = now
        self._append_log(f"Hotkey {self.settings.killswitch_key.upper()} pressed")
        self._toggle_macro()

    def _save_settings(self):
        """Save current settings to disk (triggered by auto-save)."""
        try:
            self.settings.cast_hold_time = self.cast_time_var.get()
            self.settings.recast_delay = self.recast_delay_var.get()
            self.settings.scan_interval_ms = int(self.scan_interval_var.get())
            self.settings.auto_recast = self.auto_recast_var.get()
            self.settings.shake_enabled = self.shake_enabled_var.get()
            self.settings.active_profile = self.profile_var.get()
            self.settings.fish_prediction_ms = self.fish_prediction_ms_var.get()
            self.settings.prediction_weight = self.prediction_weight_var.get()
            self.settings.fish_velocity_smoothing = self.fish_velocity_smoothing_var.get()
            self.settings.bar_velocity_smoothing = self.bar_velocity_smoothing_var.get()
            self.settings.bar_momentum_factor = self.bar_momentum_factor_var.get()
            self.settings.control_kp = self.control_kp_var.get()
            self.settings.control_kd = self.control_kd_var.get()
            self.settings.reeling_guard_seconds = self.reeling_guard_var.get()
            self.settings.min_catch_seconds = self.min_catch_seconds_var.get()
            self.settings.fast_catch_min_seconds = self.fast_catch_min_seconds_var.get()
            self.settings.post_catch_lockout_seconds = self.post_catch_lockout_var.get()
            self.settings.min_midgame_progress = self.min_midgame_progress_var.get()
            self.settings.finish_progress_threshold = self.finish_progress_threshold_var.get()
            self.settings.roi_shift_x = self.roi_shift_x_var.get()
            self.settings.roi_shift_y = self.roi_shift_y_var.get()
            self.settings.window_inset_top = self.window_inset_top_var.get()
            self.settings.window_inset_left = self.window_inset_left_var.get()
            self.settings.prediction_use_acceleration = self.prediction_use_accel_var.get()
            self.settings.prediction_arrival_lead = self.prediction_arrival_lead_var.get()
            self.settings.show_live_vision = self.show_live_vision_var.get()
            self.settings.off_target_chase_gain = self.off_target_chase_var.get()

            self.config.save_settings(self.settings)
        except Exception as e:
            logger.error(f"Auto-save settings failed: {e}")

    def _save_calibration(self):
        """Save current calibration settings into the selected rod profile."""
        try:
            self._sync_bar_roi_from_entries()
            profile = self.config.load_profile(self.profile_var.get())
            self._store_current_calibration_in_profile(profile, copy_active_colors=False)
            self.config.save_settings(self.settings)
            self.config.save_profile(profile)
            self._refresh_roi_fields()
            self._update_hsv_display()
            self._append_log(f"Calibration saved to rod profile: {profile.name}")
        except Exception as e:
            messagebox.showerror("Calibration Error", f"Invalid values: {e}")

    def _on_profile_change(self, event):
        """Handle profile selection change by recalling colors and bounds."""
        profile_name = self.profile_var.get()
        profile = self.config.load_profile(profile_name)
        self.settings.active_profile = profile_name
        self._apply_profile_calibration(profile)
        self.config.save_settings(self.settings)
        self._refresh_roi_fields()
        self._update_hsv_display()
        self._append_log(f"Loaded rod profile: {profile_name}")

    def _delete_profile(self):
        """Delete the currently selected profile."""
        name = self.profile_var.get()
        if name == "default":
            messagebox.showwarning("Cannot Delete", "The 'default' profile cannot be deleted.")
            return

        if not messagebox.askyesno("Delete Profile", f"Are you sure you want to delete the '{name}' profile?"):
            return

        if self.config.delete_profile(name):
            self._append_log(f"Deleted rod profile: {name}")
            # Switch back to default
            self.profile_var.set("default")
            self.profile_combo.configure(values=self.config.list_profiles())
            self._on_profile_change(None)
        else:
            messagebox.showerror("Error", f"Could not delete profile '{name}'.")

    def _sync_bar_roi_from_entries(self):
        """No longer used as manual entries were removed."""
        pass

    def _store_current_calibration_in_profile(self, profile, copy_active_colors=True):
        """Persist current colors plus all ROI bounds into a rod profile."""
        if copy_active_colors:
            active_profile = self.config.get_active_profile()
            profile.fish_hsv_low = list(active_profile.fish_hsv_low)
            profile.fish_hsv_high = list(active_profile.fish_hsv_high)
            profile.bar_hsv_low = list(active_profile.bar_hsv_low)
            profile.bar_hsv_high = list(active_profile.bar_hsv_high)
            profile.bar_brightness_threshold = active_profile.bar_brightness_threshold
        profile.bar_roi = self.settings.bar_roi
        profile.progress_roi = self.settings.progress_roi
        profile.shake_roi = self.settings.shake_roi

    def _apply_profile_calibration(self, profile):
        """Apply saved ROI bounds from a rod profile if present."""
        if profile.bar_roi is not None:
            self.settings.bar_roi = profile.bar_roi
        if profile.progress_roi is not None:
            self.settings.progress_roi = profile.progress_roi
        if profile.shake_roi is not None:
            self.settings.shake_roi = profile.shake_roi

    def _save_calibration_as_profile(self):
        """Create a named rod profile from the current calibration."""
        import re
        from src.config import ColorProfile

        name = simpledialog.askstring(
            "Save Rod Profile",
            "Enter the rod name for this calibration:",
            initialvalue=self.profile_var.get(),
            parent=self.root,
        )
        if name is None:
            return

        safe_name = re.sub(r"[^A-Za-z0-9_. -]+", "", name.strip()).strip()
        safe_name = safe_name.replace("/", "-")
        if not safe_name:
            messagebox.showwarning("Invalid Name", "Rod profile name cannot be empty.")
            return

        self._sync_bar_roi_from_entries()
        profile = ColorProfile(
            name=safe_name,
            description=f"Calibration for {safe_name}",
        )
        self._store_current_calibration_in_profile(profile)
        self.config.save_profile(profile)

        self.settings.active_profile = safe_name
        self.config.save_settings(self.settings)
        self.profile_var.set(safe_name)
        self.profile_combo.configure(values=self.config.list_profiles())
        self._refresh_roi_fields()
        self._update_hsv_display()
        self._append_log(f"Saved rod profile: {safe_name}")

    def _interactive_calibrate(self):
        """Open the screenshot-based eyedropper and ROI calibration window."""
        try:
            from src.interactive_calibrator import InteractiveCalibrator

            # Force fresh bounds lookup
            self.window_tracker.invalidate_cache()
            bounds = self.window_tracker.get_roblox_bounds()
            if bounds is None:
                messagebox.showwarning(
                    "Roblox Not Found",
                    "Could not find the Roblox window.\n\n"
                    "Open Roblox in windowed mode, start the Fisch minigame, "
                    "then run interactive calibration again.",
                )
                return

            scale = self.window_tracker.get_scale_factor()
            self.engine.detector.set_window_info(bounds, scale)

            from src.calibrator import Calibrator

            calibrator = Calibrator(self.engine.detector, self.config)
            background = calibrator.capture_stable_screenshot(
                wait_seconds=0.12,
                max_attempts=5,
            )
            if background is None:
                messagebox.showwarning(
                    "Screenshot Not Ready",
                    "Could not grab a clean screen image (black bars from fullscreen "
                    "animation).\n\nWait for the desktop to settle, then try again.",
                )
                return

            self._append_log("Opened interactive calibration")

            window = InteractiveCalibrator(
                self.root,
                self.engine,
                self.config,
                self.window_tracker,
                background_bgr=background,
            )
            self.root.wait_window(window)
            self.settings = self.config.load_settings()
            self._refresh_roi_fields()
            self._update_hsv_display()
            self._append_log("Interactive calibration closed")
        except Exception as e:
            logger.error(f"Interactive calibration error: {e}", exc_info=True)
            messagebox.showerror("Calibration Error", str(e))

    def _rebind_killswitch(self):
        """Prompt for a new killswitch key and restart the listener."""
        key_name = simpledialog.askstring(
            "Rebind Toggle Hotkey",
            "Enter a key name such as esc, q, x, f8, or f12.\n\n"
            "This key toggles Start/Stop. F6 may require Fn+F6 on some Macs.",
            initialvalue=self.settings.killswitch_key,
            parent=self.root,
        )
        if key_name is None:
            return

        key_name = key_name.strip().lower()
        if not key_name:
            messagebox.showwarning("Invalid Key", "Hotkey cannot be empty.")
            return

        self.settings.killswitch_key = key_name
        self.config.save_settings(self.settings)
        self.engine.controller.setup_killswitch(key_name)
        self._refresh_killswitch_labels()
        self._append_log(f"Toggle hotkey rebound to {key_name.upper()}")

    def _emergency_stop(self):
        """Stop the macro from the GUI without relying on global hotkeys."""
        self.start_button.config(
            text="▶  START",
            bg=COLORS["accent_green"],
            activebackground="#00b563",
        )
        self.start_button.update_idletasks()
        self.engine.stop()
        self._append_log("Emergency stop pressed")

    def _refresh_killswitch_labels(self):
        """Refresh all visible hotkey labels."""
        label = self.settings.killswitch_key.upper()
        self.header_killswitch_label.config(text=f"⚡ Toggle: {label}")
        self.killswitch_label.config(text=label)
        self.kill_button.config(text="EMERGENCY STOP")

    def _refresh_roi_fields(self):
        """No longer used as manual entries were removed."""
        pass

    def _update_hsv_display(self):
        """Update the HSV color range display."""
        try:
            profile = self.config.get_active_profile()
            text = (
                f"Fish: H{profile.fish_hsv_low[0]}-{profile.fish_hsv_high[0]}  "
                f"S{profile.fish_hsv_low[1]}-{profile.fish_hsv_high[1]}  "
                f"V{profile.fish_hsv_low[2]}-{profile.fish_hsv_high[2]}\n"
                f"Bar:   H{profile.bar_hsv_low[0]}-{profile.bar_hsv_high[0]}  "
                f"S{profile.bar_hsv_low[1]}-{profile.bar_hsv_high[1]}  "
                f"V{profile.bar_hsv_low[2]}-{profile.bar_hsv_high[2]}"
            )
            self.hsv_info_label.config(text=text)
        except Exception:
            self.hsv_info_label.config(text="No profile loaded")

    # ─── Thread-Safe Callbacks ────────────────────────────────────

    def _on_state_change(self, state):
        """Handle macro state change (called from macro thread)."""
        self.root.after(0, self._update_status_display, state)

    def _on_stats_update(self, stats):
        """Handle stats update (called from macro thread)."""
        self.root.after(0, self._update_stats_display, stats)

    def _on_log_message(self, message):
        """Handle log message (called from macro thread)."""
        self.root.after(0, self._append_log, message)

    def _update_status_display(self, state):
        """Update the status display (must be called on main thread)."""
        state_name = state.value
        icon = STATE_ICONS.get(state_name, "❓")
        color = STATE_COLORS.get(state_name, COLORS["text"])

        self.status_icon.config(text=icon)
        self.status_label.config(text=state_name, foreground=color)

        if state_name == "Stopped":
            self.start_button.config(
                text="▶  START",
                bg=COLORS["accent_green"],
                activebackground="#00b563",
            )
            self.status_detail.config(text="Press Start to begin fishing")

        elif state_name == "Reeling":
            self.status_detail.config(text="Keep the fish in the bar!")

        elif state_name == "Waiting for Bite":
            self.status_detail.config(text="Waiting for a fish to bite...")

        elif state_name == "Casting":
            self.status_detail.config(text="Casting the rod...")

        elif state_name == "Catch Complete":
            self.status_detail.config(text="Nice catch! Preparing to recast...")

    def _update_stats_display(self, stats):
        """Update the stats display (must be called on main thread)."""
        self.stat_caught.config(text=str(stats.fish_caught))
        self.stat_failed.config(text=str(stats.fish_failed))
        self.stat_casts.config(text=str(stats.casts))

        rate = stats.success_rate
        if rate > 0:
            self.stat_rate.config(text=f"{rate:.0%}")
        else:
            self.stat_rate.config(text="—")

    def _append_log(self, message):
        """Append a message to the log (must be called on main thread)."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

        # Keep log size manageable
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 500:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete("1.0", "100.0")
            self.log_text.config(state=tk.DISABLED)

    def _clear_log(self):
        """Clear the activity log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ─── Periodic Updates ─────────────────────────────────────────

    def _schedule_update(self):
        """Schedule periodic UI updates."""
        self._periodic_update()
        self.root.after(self._update_interval, self._schedule_update)

    def _schedule_vision_update(self):
        """Refresh live vision overlay at higher FPS while running."""
        self._refresh_vision_panel()
        self.root.after(self._vision_interval, self._schedule_vision_update)

    def _refresh_vision_panel(self):
        """Draw the latest detection frame and telemetry on the Control tab."""
        if not self.settings.show_live_vision:
            return
        try:
            import cv2
            from PIL import Image, ImageTk

            running = self.engine.is_running()
            if not running:
                # Nothing is looking at the screen, so whatever is on the panel
                # is from a fight that has already ended. Leaving it up made
                # the bars look like a live reading of a bar that is not there.
                if self._vision_photo is not None:
                    self._vision_photo = None
                    self.vision_image_label.config(
                        image="", text="Start macro to see detection…"
                    )
                    self._set_vision_telemetry("")
                return

            snap = self.engine.get_vision_snapshot()

            if snap.frame_bgr is None:
                self._vision_photo = None
                self.vision_image_label.config(image="", text="Waiting for the reel ROI…")
            else:
                rgb = cv2.cvtColor(snap.frame_bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                target_w = max(320, self.vision_image_label.winfo_width() or 380)
                scale = min(1.0, target_w / max(pil.width, 1))
                target_h = max(1, int(pil.height * scale))
                if pil.width != target_w or pil.height != target_h:
                    pil = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                self._vision_photo = ImageTk.PhotoImage(pil)
                self.vision_image_label.config(image=self._vision_photo, text="")

            # Updated even with no frame: while hunting, the numbers are the
            # only thing there is to see, and holding the previous fight's
            # telemetry there was actively misleading.
            self._set_vision_telemetry("\n".join(snap.summary_lines()))
        except Exception as exc:
            logger.debug("Vision panel refresh failed: %s", exc)

    def _set_vision_telemetry(self, text: str) -> None:
        """Replace the telemetry readout text."""
        self.vision_telemetry.config(state=tk.NORMAL)
        self.vision_telemetry.delete("1.0", tk.END)
        if text:
            self.vision_telemetry.insert(tk.END, text)
        self.vision_telemetry.config(state=tk.DISABLED)

    def _periodic_update(self):
        """Periodic tasks: update session time, check window status."""
        # Update session time and target status
        if self.engine.is_running():
            duration = self.engine.stats.session_duration
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            self.session_time.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")

            hint = getattr(self.engine, "status_hint", "") or ""
            if self.engine.state.value == "Reeling":
                if self.engine.last_on_target:
                    self.on_target_label.config(
                        text="🎯 ON TARGET", foreground=COLORS["accent_green"]
                    )
                else:
                    self.on_target_label.config(
                        text="⚠️ OFF TARGET", foreground=COLORS["accent_yellow"]
                    )
            elif hint:
                self.on_target_label.config(
                    text=f"📡 {hint[:72]}",
                    foreground=COLORS["accent_blue"],
                )
            else:
                self.on_target_label.config(text="")
            if hint and self.engine.state.value != "Reeling":
                self.status_detail.config(text=hint[:120])
        else:
            self.on_target_label.config(text="")

        # Update window detection status
        bounds = self.window_tracker.get_roblox_bounds()
        if bounds is not None:
            self.window_status.config(
                text=f"{bounds.width}×{bounds.height}",
                foreground=COLORS["accent_green"],
            )
            # Update detector with latest bounds
            scale = self.window_tracker.get_scale_factor()
            self.engine.detector.set_window_info(bounds, scale)
        else:
            self.window_status.config(
                text="Not found",
                foreground=COLORS["danger"],
            )

    # ─── Window Management ────────────────────────────────────────

    def _on_close(self):
        """Handle window close event.

        The loop thread is a daemon and its stop flag is already set, so the
        window can go now; a short join keeps the common case tidy without the
        multi-second stall that made quitting feel hung.
        """
        if self.engine.is_running():
            self.engine.stop(wait=0.3)
        self.root.destroy()

    def run(self):
        """Start the GUI main loop."""
        self._append_log("Fisch Macro initialized")
        self._append_log(f"Start/stop hotkey: {self.settings.killswitch_key.upper()}")
        self._append_log("Ready — press Start to begin")
        self.root.mainloop()
