"""
Fisch Macro GUI — tkinter Control Panel

Provides a compact, always-on-top window with:
- Control tab: Start/Stop, status, live vision, stats
- Settings tab: everyday controls, then fold-out sections for tuning
- Calibration tab: ROI + color profile management
- Log tab: activity log

Sized to sit beside a windowed Roblox rather than cover it. Nothing was
dropped to get there: the panel shows the same readouts and offers the same
controls, packed into single-row sliders, paired toggles and shorter cards,
with the tuning parameters folded away behind their section headers.
"""

import logging
import sys
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
    "settings_head": "#dde3ec",
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
        # Two thirds the old footprint: the panel sits beside a windowed
        # Roblox rather than covering it.
        self.root.geometry("372x572")
        self.root.minsize(340, 500)
        self._vision_photo = None
        self.root.configure(bg=COLORS["bg"])
        self.root.attributes("-topmost", True)

        # Try to set the window style. Note this does not settle text size:
        # every font in the panel is specified in pixels (a negative size)
        # rather than points, which is what actually keeps the layout stable
        # across Tk versions. Tk 8.6 on Aqua renders a point as a pixel, while
        # Tk 9 converts at ~96dpi, so the same positive size comes out a third
        # larger there -- the shipped .app bundles 8.6 and a Homebrew Python
        # 3.14 checkout runs 9.0, and the two panels did not look alike.
        # Setting the scaling factor does not fix it; measured, a size-10 font
        # is 16px of linespace under Tk 9 at either scaling, and 12px at -10.
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
        self._bound_keysym = None
        self._bind_hotkey_key()

        # Auto-save setup
        self._setup_auto_save()

        # Periodic UI update
        self._update_interval = 500  # ms
        self._vision_interval = 80  # ms
        self._schedule_update()
        self._schedule_vision_update()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Last, so the log tab it writes to exists.
        self._check_input_permission()

    def _setup_auto_save(self):
        """Add traces to all settings variables for automatic saving."""
        for var in [
            self.cast_time_var,
            self.recast_delay_var,
            self.scan_interval_var,
            self.auto_recast_var,
            self.shake_enabled_var,
            self.humanize_var,
            self.duty_kp_var,
            self.duty_ki_var,
            self.neutral_duty_var,
            self.lead_seconds_var,
            self.stationary_speed_var,
            self.bar_accel_var,
            self.latency_var,
            self.max_brake_var,
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
            self.show_live_vision_var,
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
            padding=[9, 4],
            borderwidth=0,
            font=("Helvetica Neue", -10),
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
            font=("Helvetica Neue", -11),
        )
        s.configure(
            "Header.TLabel",
            font=("Helvetica Neue", -16, "bold"),
            foreground=COLORS["text"],
        )
        s.configure(
            "Status.TLabel",
            font=("Helvetica Neue", -12, "bold"),
            foreground=COLORS["accent"],
        )
        s.configure(
            "Stat.TLabel",
            font=("Menlo", -11),
            foreground=COLORS["accent_blue"],
        )
        s.configure(
            "Dim.TLabel",
            font=("Helvetica Neue", -10),
            foreground=COLORS["text_dim"],
        )
        s.configure(
            "Card.TLabel",
            background=COLORS["bg_card"],
            foreground=COLORS["text"],
            font=("Helvetica Neue", -11),
        )
        s.configure(
            "CardDim.TLabel",
            background=COLORS["bg_card"],
            foreground=COLORS["text_dim"],
            font=("Helvetica Neue", -9),
        )
        s.configure(
            "CardStat.TLabel",
            background=COLORS["bg_card"],
            foreground=COLORS["accent_blue"],
            font=("Menlo", -13, "bold"),
        )

        s.configure(
            "Start.TButton",
            font=("Helvetica Neue", -12, "bold"),
            padding=[12, 6],
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
            font=("Helvetica Neue", -10),
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
            font=("Helvetica Neue", -11),
        )
        s.configure(
            "SettingsDim.TLabel",
            background=COLORS["settings_bg"],
            foreground=COLORS["settings_dim"],
            font=("Helvetica Neue", -9, "bold"),
        )
        s.configure(
            "SettingsStat.TLabel",
            background=COLORS["settings_bg"],
            foreground=COLORS["settings_text"],
            font=("Menlo", -10),
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
            font=("Helvetica Neue", -10),
        )

    # ─── Header ───────────────────────────────────────────────────

    def _build_header(self):
        """Build the title header with hotkey info."""
        header = ttk.Frame(self.root, style="TFrame")
        header.pack(fill=tk.X, padx=10, pady=(8, 2))

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
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))

        self.control_tab = ttk.Frame(self.notebook, style="TFrame")
        self.settings_tab = ttk.Frame(self.notebook, style="TFrame")
        self.calibration_tab = ttk.Frame(self.notebook, style="TFrame")
        self.log_tab = ttk.Frame(self.notebook, style="TFrame")

        self.notebook.add(self.control_tab, text=" Control ")
        self.notebook.add(self.settings_tab, text=" Settings ")
        self.notebook.add(self.calibration_tab, text=" Calibrate ")
        self.notebook.add(self.log_tab, text=" Log ")

        # The calibration readout reflects the ROI-shift sliders on the
        # Settings tab, so it is re-read on the way in rather than on every
        # drag event.
        self.notebook.bind(
            "<<NotebookTabChanged>>", lambda _e: self._refresh_calibration_view()
        )

    # ─── Control Tab ──────────────────────────────────────────────

    def _add_row_card(self, parent, pady=(3, 3), ipady=7, expand=False):
        """Add a card to a container and return its padded inner frame."""
        fill = tk.BOTH if expand else tk.X
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill=fill, padx=6, pady=pady, expand=expand)

        inner = ttk.Frame(card, style="Card.TFrame")
        inner.pack(fill=fill, padx=10, pady=ipady, expand=expand)
        return inner

    def _build_control_tab(self):
        """Build the main control panel.

        Everything the panel showed before is still here — state, detail line,
        on-target readout, live vision with its key, telemetry, both buttons,
        the four counters, session time and window size. It is packed into
        shorter cards and paired rows so it all fits without scrolling in a
        window small enough to sit beside Roblox.
        """
        tab = self.control_tab

        # ── Status ──
        status_inner = self._add_row_card(tab, pady=(6, 3))

        self.status_icon = ttk.Label(
            status_inner, text="⏸", style="Card.TLabel",
            font=("Helvetica Neue", -19),
        )
        self.status_icon.pack(side=tk.LEFT, padx=(0, 8))

        status_text = ttk.Frame(status_inner, style="Card.TFrame")
        status_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status_label = ttk.Label(
            status_text, text="Stopped", style="Card.TLabel",
            font=("Helvetica Neue", -13, "bold"),
        )
        self.status_label.pack(anchor=tk.W)

        self.status_detail = ttk.Label(
            status_text, text="Press Start to begin fishing",
            style="CardDim.TLabel",
        )
        self.status_detail.pack(anchor=tk.W)

        self.on_target_label = ttk.Label(
            status_text, text="",
            style="Card.TLabel", font=("Helvetica Neue", -10, "bold")
        )
        self.on_target_label.pack(anchor=tk.W)

        # ── Live vision (digmacro-style preview) ──
        # Given the leftover height, so shrinking the window trims the
        # telemetry rather than pushing the buttons off the bottom.
        vision_inner = self._add_row_card(tab, pady=(3, 3), ipady=6, expand=True)

        ttk.Label(
            vision_inner,
            text="👁 Live Vision",
            style="Card.TLabel",
            font=("Helvetica Neue", -10, "bold"),
        ).pack(anchor=tk.W)

        # height=3 is three *lines*, for the placeholder text. Tk reads the
        # same option as three *pixels* once an image is showing, so it has to
        # be cleared before the frame goes in — see _show_vision_image.
        self.vision_image_label = tk.Label(
            vision_inner,
            text="Start macro to see detection…",
            bg="#0a0a14",
            fg=COLORS["text"],
            font=("Menlo", -9),
            height=3,
        )
        self.vision_image_label.pack(fill=tk.X)

        self._build_vision_key(vision_inner)

        self.vision_telemetry = tk.Text(
            vision_inner,
            height=5,
            width=1,
            bg="#0a0a14",
            fg="#a8d4ff",
            font=("Menlo", -8),
            relief=tk.FLAT,
            wrap=tk.WORD,
            padx=4,
            pady=2,
        )
        self.vision_telemetry.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.vision_telemetry.config(state=tk.DISABLED)

        # ── Start/Stop + emergency stop, side by side ──
        buttons = ttk.Frame(tab, style="TFrame")
        buttons.pack(fill=tk.X, padx=6, pady=(4, 3))

        self.start_button = tk.Button(
            buttons,
            text="▶  START",
            font=("Helvetica Neue", -13, "bold"),
            bg=COLORS["accent_green"],
            fg="#ffffff",
            activebackground="#00b563",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            height=1,
            command=self._toggle_macro,
        )
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.kill_button = tk.Button(
            buttons,
            text="■ E-STOP",
            font=("Helvetica Neue", -11, "bold"),
            bg=COLORS["danger"],
            fg="#ffffff",
            activebackground="#b83045",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            height=1,
            command=self._emergency_stop,
        )
        self.kill_button.pack(side=tk.LEFT, padx=(6, 0))

        # ── Stats Grid ──
        stats_inner = self._add_row_card(tab, pady=(3, 3), ipady=6)
        stats_inner.columnconfigure((0, 1, 2, 3), weight=1)

        for i, label in enumerate(["Caught", "Failed", "Casts", "Rate"]):
            ttk.Label(
                stats_inner, text=label, style="CardDim.TLabel"
            ).grid(row=0, column=i, sticky=tk.N)

        self.stat_caught = ttk.Label(stats_inner, text="0", style="CardStat.TLabel")
        self.stat_caught.grid(row=1, column=0)

        self.stat_failed = ttk.Label(stats_inner, text="0", style="CardStat.TLabel")
        self.stat_failed.grid(row=1, column=1)

        self.stat_casts = ttk.Label(stats_inner, text="0", style="CardStat.TLabel")
        self.stat_casts.grid(row=1, column=2)

        self.stat_rate = ttk.Label(stats_inner, text="—", style="CardStat.TLabel")
        self.stat_rate.grid(row=1, column=3)

        # ── Session time + Roblox window, one row instead of two cards ──
        footer = self._add_row_card(tab, pady=(3, 6), ipady=5)

        ttk.Label(footer, text="⏱", style="CardDim.TLabel").pack(side=tk.LEFT)
        self.session_time = ttk.Label(
            footer, text="00:00:00", style="Card.TLabel",
            font=("Menlo", -11),
        )
        self.session_time.pack(side=tk.LEFT, padx=(4, 0))

        self.window_status = ttk.Label(
            footer, text="Searching...", style="Card.TLabel",
            font=("Helvetica Neue", -10),
        )
        self.window_status.pack(side=tk.RIGHT)
        ttk.Label(footer, text="🖥", style="CardDim.TLabel").pack(
            side=tk.RIGHT, padx=(0, 4)
        )

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
        key.pack(fill=tk.X, pady=(4, 0))

        # Three across rather than two: the same entries in two rows instead
        # of three or four.
        per_row = 3
        row = None
        for index, (name, label) in enumerate(OVERLAY_LEGEND):
            if index % per_row == 0:
                row = ttk.Frame(key, style="Card.TFrame")
                row.pack(fill=tk.X)

            cell = ttk.Frame(row, style="Card.TFrame")
            cell.pack(side=tk.LEFT, fill=tk.X, expand=True)

            tk.Frame(
                cell,
                bg=legend_hex(name),
                width=10,
                height=3,
                highlightthickness=0,
            ).pack(side=tk.LEFT, padx=(0, 4), pady=2)

            ttk.Label(
                cell,
                text=label,
                style="CardDim.TLabel",
                font=("Helvetica Neue", -8),
            ).pack(side=tk.LEFT)

    # ─── Settings Tab ─────────────────────────────────────────────

    def _build_settings_tab(self):
        """Build the settings panel.

        Ordered by how often a control is touched: the handful that get
        changed between sessions sit at the top, always visible, and the
        tuning parameters live in collapsed sections underneath. Everything
        that was here before is still here — nothing has been dropped, it is
        one click away instead of ten scroll wheels down.
        """
        tab = self.settings_tab

        # Scrollable content
        canvas = tk.Canvas(tab, bg=COLORS["settings_bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        content = ttk.Frame(canvas, style="Settings.TFrame")

        content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        window_id = canvas.create_window((0, 0), window=content, anchor=tk.NW)
        # Without this the content keeps its natural width, so the sliders
        # stopped short of the panel edge and the value column floated.
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(window_id, width=e.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._bind_mousewheel(canvas)

        # ══ Everyday controls, always visible ══
        self._add_section_header(content, "Fishing")

        self.cast_time_var = tk.DoubleVar(value=self.settings.cast_hold_time)
        self._add_slider(
            content, "Cast Hold Time", self.cast_time_var, 0.5, 4.0, 0.1, suffix="s"
        )

        self.recast_delay_var = tk.DoubleVar(value=self.settings.recast_delay)
        self._add_slider(
            content, "Recast Delay", self.recast_delay_var, 0.5, 5.0, 0.1, suffix="s"
        )

        self.scan_interval_var = tk.IntVar(value=self.settings.scan_interval_ms)
        self._add_slider(
            content, "Scan Interval", self.scan_interval_var, 20, 100, 5, suffix="ms"
        )

        self.auto_recast_var = tk.BooleanVar(value=self.settings.auto_recast)
        self.shake_enabled_var = tk.BooleanVar(value=self.settings.shake_enabled)
        self._add_toggle_pair(
            content,
            "Auto Recast", self.auto_recast_var,
            "Auto Shake", self.shake_enabled_var,
        )

        self.show_live_vision_var = tk.BooleanVar(value=self.settings.show_live_vision)
        self._add_toggle(content, "Live Vision Preview", self.show_live_vision_var)

        # Off means every cast is charged for the identical number of
        # microseconds and every shake click lands on the identical pixel.
        self.humanize_var = tk.BooleanVar(value=self.settings.humanize)
        self._add_toggle(content, "Humanize Inputs", self.humanize_var)

        # Hotkey — one row, next to the toggles it belongs with.
        ks_frame = ttk.Frame(content, style="Settings.TFrame")
        ks_frame.pack(fill=tk.X, padx=10, pady=(2, 4))

        ttk.Label(ks_frame, text="Start/Stop Key", style="Settings.TLabel").pack(
            side=tk.LEFT
        )
        rebind_btn = tk.Button(
            ks_frame, text="Rebind", font=("Helvetica Neue", -10),
            bg=COLORS["accent"], fg="white", relief=tk.FLAT, cursor="hand2",
            padx=8, pady=0,
            command=self._rebind_killswitch,
        )
        rebind_btn.pack(side=tk.RIGHT)

        self.killswitch_label = ttk.Label(
            ks_frame,
            text=self.settings.killswitch_key.upper(),
            style="SettingsStat.TLabel",
        )
        self.killswitch_label.pack(side=tk.RIGHT, padx=8)

        # ══ Everything else, folded away ══
        self._add_section_header(content, "Advanced")

        # ── Reeling ──
        # These are ReelController's own parameters. The sliders that used to
        # sit here belonged to the PD controller it replaced and had not been
        # wired to anything for some time: half of them were read nowhere at
        # all, and the rest only inside helpers with no callers. Tuning them
        # did nothing, which is worse than not offering them.
        reeling = self._add_collapsible(content, "Reeling & Control")

        self.duty_kp_var = tk.DoubleVar(value=self.settings.control_duty_kp)
        self._add_slider(reeling, "Steering Strength", self.duty_kp_var, 2.0, 25.0, 0.5)

        self.duty_ki_var = tk.DoubleVar(value=self.settings.control_duty_ki)
        self._add_slider(reeling, "Steering Trim", self.duty_ki_var, 0.0, 3.0, 0.1)

        self.neutral_duty_var = tk.DoubleVar(value=self.settings.control_neutral_duty)
        self._add_slider(reeling, "Neutral Hold", self.neutral_duty_var, 0.30, 0.70, 0.005)

        self.lead_seconds_var = tk.DoubleVar(value=self.settings.control_lead_seconds)
        self._add_slider(reeling, "Fish Lookahead", self.lead_seconds_var, 0.0, 0.25, 0.01, suffix="s")

        self.stationary_speed_var = tk.DoubleVar(value=self.settings.control_stationary_speed)
        self._add_slider(reeling, "Stationary Speed", self.stationary_speed_var, 0.0, 0.20, 0.01)

        self.bar_accel_var = tk.DoubleVar(value=self.settings.control_bar_accel)
        self._add_slider(reeling, "Bar Acceleration", self.bar_accel_var, 0.2, 3.0, 0.05)

        self.latency_var = tk.DoubleVar(value=self.settings.control_latency_seconds)
        self._add_slider(reeling, "Input Latency", self.latency_var, 0.0, 0.20, 0.01, suffix="s")

        self.max_brake_var = tk.DoubleVar(value=self.settings.control_max_brake_distance)
        self._add_slider(reeling, "Max Brake Distance", self.max_brake_var, 0.05, 0.60, 0.05)

        # ── Catch Detection ──
        catch = self._add_collapsible(content, "Catch Detection")

        self.reeling_guard_var = tk.DoubleVar(value=self.settings.reeling_guard_seconds)
        self._add_slider(catch, "Reel Start Guard", self.reeling_guard_var, 1.0, 5.0, 0.25, suffix="s")

        self.min_catch_seconds_var = tk.DoubleVar(value=self.settings.min_catch_seconds)
        self._add_slider(catch, "Min Catch Time", self.min_catch_seconds_var, 1.0, 10.0, 0.5, suffix="s")

        self.fast_catch_min_seconds_var = tk.DoubleVar(
            value=getattr(self.settings, "fast_catch_min_seconds", 1.0)
        )
        self._add_slider(
            catch, "Fast Rod Min Reel", self.fast_catch_min_seconds_var,
            0.5, 4.0, 0.25, suffix="s",
        )

        self.post_catch_lockout_var = tk.DoubleVar(
            value=getattr(self.settings, "post_catch_lockout_seconds", 1.5)
        )
        self._add_slider(
            catch, "Post-Catch Lockout", self.post_catch_lockout_var,
            0.5, 5.0, 0.25, suffix="s",
        )

        self.min_midgame_progress_var = tk.DoubleVar(value=self.settings.min_midgame_progress)
        self._add_slider(catch, "Min Catch Progress", self.min_midgame_progress_var, 0.20, 0.60, 0.05)

        self.finish_progress_threshold_var = tk.DoubleVar(value=self.settings.finish_progress_threshold)
        self._add_slider(catch, "Finish Progress", self.finish_progress_threshold_var, 0.90, 0.99, 0.01)

        # ── Calibration alignment ──
        align = self._add_collapsible(content, "Calibration Alignment")

        self.roi_shift_x_var = tk.DoubleVar(value=self.settings.roi_shift_x)
        self._add_slider(align, "ROI Shift X", self.roi_shift_x_var, -0.08, 0.08, 0.005)

        self.roi_shift_y_var = tk.DoubleVar(value=self.settings.roi_shift_y)
        self._add_slider(align, "ROI Shift Y", self.roi_shift_y_var, -0.08, 0.08, 0.005)

        self.window_inset_top_var = tk.DoubleVar(value=self.settings.window_inset_top)
        self._add_slider(align, "Window Inset Top", self.window_inset_top_var, 0.0, 0.08, 0.005)

        self.window_inset_left_var = tk.DoubleVar(value=self.settings.window_inset_left)
        self._add_slider(align, "Window Inset Left", self.window_inset_left_var, 0.0, 0.08, 0.005)

    def _bind_mousewheel(self, canvas):
        """Scroll a canvas under the pointer with the wheel/trackpad."""
        def on_wheel(event):
            delta = event.delta
            if delta == 0:
                delta = 1 if event.num == 4 else -1
            canvas.yview_scroll(-1 if delta > 0 else 1, "units")

        canvas.bind("<Enter>", lambda _e: (
            canvas.bind_all("<MouseWheel>", on_wheel),
            canvas.bind_all("<Button-4>", on_wheel),
            canvas.bind_all("<Button-5>", on_wheel),
        ))
        canvas.bind("<Leave>", lambda _e: (
            canvas.unbind_all("<MouseWheel>"),
            canvas.unbind_all("<Button-4>"),
            canvas.unbind_all("<Button-5>"),
        ))

    def _add_section_header(self, parent, text):
        """Add a section header label."""
        frame = ttk.Frame(parent, style="Settings.TFrame")
        frame.pack(fill=tk.X, padx=10, pady=(10, 2))
        ttk.Label(
            frame,
            text=text.upper(),
            style="SettingsDim.TLabel",
        ).pack(anchor=tk.W)

    def _add_collapsible(self, parent, title, expanded=False):
        """Add a fold-out section and return the frame its controls go in.

        Collapsed by default: these are the knobs that get set once during
        tuning and then left alone, so they cost a click when wanted and no
        height at all the rest of the time.
        """
        wrapper = ttk.Frame(parent, style="Settings.TFrame")
        wrapper.pack(fill=tk.X, padx=10, pady=(2, 0))

        head = tk.Label(
            wrapper,
            anchor=tk.W,
            bg=COLORS["settings_head"],
            fg=COLORS["settings_text"],
            font=("Helvetica Neue", -11, "bold"),
            padx=8,
            pady=4,
            cursor="hand2",
        )
        head.pack(fill=tk.X)

        body = ttk.Frame(wrapper, style="Settings.TFrame")
        is_open = tk.BooleanVar(value=expanded)

        def render():
            arrow = "▾" if is_open.get() else "▸"
            head.config(text=f"{arrow}  {title}")
            if is_open.get():
                body.pack(fill=tk.X, pady=(2, 6))
            else:
                body.pack_forget()

        def toggle(_event=None):
            is_open.set(not is_open.get())
            render()

        head.bind("<Button-1>", toggle)
        render()
        return body

    def _add_slider(self, parent, label, variable, from_, to, resolution, suffix=""):
        """Add a labeled slider on a single row: name, track, value.

        The name and value used to sit on a row of their own above the track,
        which cost twice the height per control. ``resolution`` now picks how
        many decimals the value shows — it was accepted and ignored before, so
        a 0.005-step slider read as one decimal and looked stuck while moving.
        """
        row = ttk.Frame(parent, style="Settings.TFrame")
        row.pack(fill=tk.X, padx=10, pady=1)

        ttk.Label(
            row, text=label, style="Settings.TLabel", width=19, anchor=tk.W
        ).pack(side=tk.LEFT)

        decimals = 0
        if isinstance(resolution, float):
            step = abs(resolution)
            decimals = 1 if step >= 0.1 else 2 if step >= 0.01 else 3

        value_label = ttk.Label(
            row,
            text="",
            style="SettingsStat.TLabel",
            width=6,
            anchor=tk.E,
        )
        value_label.pack(side=tk.RIGHT)

        scale = ttk.Scale(
            row,
            from_=from_,
            to=to,
            variable=variable,
            style="Settings.TScale",
            orient=tk.HORIZONTAL,
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 4))

        def update_label(*_args):
            val = variable.get()
            if isinstance(val, float):
                value_label.config(text=f"{val:.{decimals}f}{suffix}")
            else:
                value_label.config(text=f"{val}{suffix}")

        variable.trace_add("write", update_label)
        update_label()

    def _add_toggle(self, parent, label, variable):
        """Add a labeled toggle checkbox."""
        frame = ttk.Frame(parent, style="Settings.TFrame")
        frame.pack(fill=tk.X, padx=10, pady=1)
        cb = ttk.Checkbutton(
            frame, text=label, variable=variable, style="Settings.TCheckbutton"
        )
        cb.pack(anchor=tk.W)

    def _add_toggle_pair(self, parent, left_label, left_var, right_label, right_var):
        """Add two toggles side by side, for the ones that read as a pair."""
        frame = ttk.Frame(parent, style="Settings.TFrame")
        frame.pack(fill=tk.X, padx=10, pady=1)
        frame.columnconfigure((0, 1), weight=1)

        ttk.Checkbutton(
            frame, text=left_label, variable=left_var, style="Settings.TCheckbutton"
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(
            frame, text=right_label, variable=right_var, style="Settings.TCheckbutton"
        ).grid(row=0, column=1, sticky=tk.W)

    # ─── Calibration Tab ──────────────────────────────────────────

    def _build_calibration_tab(self):
        """Build the calibration panel.

        Laid out in the order the job is actually done — pick the rod, mark the
        regions, read back what was stored, save it — because the previous
        layout offered two similarly worded save buttons and no way to see what
        either of them had done.
        """
        tab = self.calibration_tab

        # Scrollable: the readout is taller than the tab on a small window.
        canvas = tk.Canvas(tab, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        content = ttk.Frame(canvas, style="TFrame")
        content.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        window_id = canvas.create_window((0, 0), window=content, anchor=tk.NW)
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(window_id, width=e.width),
        )
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._bind_mousewheel(canvas)

        # ── Step 1: which rod ──
        step1 = self._add_card(content, "1  Rod profile")

        ttk.Label(
            step1,
            text="One profile per rod, holding that rod's colours and the three\n"
                 "regions below. Switching rods loads its saved calibration back.",
            style="CardDim.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 6))

        selector = ttk.Frame(step1, style="Card.TFrame")
        selector.pack(fill=tk.X)

        self.profile_combo = ttk.Combobox(
            selector,
            textvariable=self.profile_var,
            values=self.config.list_profiles(),
            state="readonly",
            width=14,
            font=("Helvetica Neue", -10),
        )
        self.profile_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_change)

        tk.Button(
            selector,
            text="＋ New rod",
            font=("Helvetica Neue", -10),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text"],
            activebackground=COLORS["border"],
            activeforeground=COLORS["text"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._save_calibration_as_profile,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            selector,
            text="🗑",
            font=("Helvetica Neue", -10),
            bg=COLORS["danger"],
            fg="#ffffff",
            activebackground="#c73a50",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            width=3,
            command=self._delete_profile,
        ).pack(side=tk.LEFT)

        # ── Step 2: mark the regions ──
        step2 = self._add_card(content, "2  Mark the regions")

        ttk.Label(
            step2,
            text="Opens a full-screen snapshot. Click the fish and bar colours,\n"
                 "then drag a box around each region and press Save & Close.\n"
                 "Roblox must be windowed with the minigame on screen.",
            style="CardDim.TLabel",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 6))

        tk.Button(
            step2,
            text="👁  Open calibration overlay",
            font=("Helvetica Neue", -11, "bold"),
            bg=COLORS["accent_yellow"],
            fg="#ffffff",
            activebackground="#f5b853",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._interactive_calibrate,
        ).pack(fill=tk.X)

        # ── Step 3: what the macro will actually use ──
        step3 = self._add_card(content, "3  Current calibration")

        self.calibration_readout = tk.Text(
            step3,
            height=8,
            width=1,
            bg="#0a0a14",
            fg="#a8d4ff",
            font=("Menlo", -9),
            relief=tk.FLAT,
            wrap=tk.NONE,
            padx=6,
            pady=4,
            highlightthickness=0,
        )
        self.calibration_readout.pack(fill=tk.X, pady=(4, 0))
        self.calibration_readout.config(state=tk.DISABLED)

        self.calibration_status = ttk.Label(
            step3,
            text="",
            style="Card.TLabel",
            font=("Helvetica Neue", -10, "bold"),
            wraplength=290,
            justify=tk.LEFT,
        )
        self.calibration_status.pack(anchor=tk.W, pady=(6, 0))

        buttons = ttk.Frame(step3, style="Card.TFrame")
        buttons.pack(fill=tk.X, pady=(8, 0))

        self.save_calibration_btn = tk.Button(
            buttons,
            text="💾  Save to rod",
            font=("Helvetica Neue", -11, "bold"),
            bg=COLORS["accent_blue"],
            fg="#ffffff",
            activebackground="#3aa3d7",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            command=self._save_calibration,
        )
        self.save_calibration_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.revert_calibration_btn = tk.Button(
            buttons,
            text="↩  Revert",
            font=("Helvetica Neue", -11),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text"],
            activebackground=COLORS["border"],
            activeforeground=COLORS["text"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._revert_calibration,
        )
        self.revert_calibration_btn.pack(side=tk.LEFT)

        self._refresh_calibration_view()

    def _add_card(self, parent, title):
        """Add a titled card to a container and return its inner frame."""
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill=tk.X, padx=6, pady=4)

        inner = ttk.Frame(card, style="Card.TFrame")
        inner.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(
            inner, text=title, style="Card.TLabel",
            font=("Helvetica Neue", -11, "bold"),
        ).pack(anchor=tk.W)
        return inner

    # ─── Log Tab ──────────────────────────────────────────────────

    def _build_log_tab(self):
        """Build the activity log panel."""
        tab = self.log_tab

        # Log text area
        log_frame = ttk.Frame(tab)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.log_text = tk.Text(
            log_frame,
            width=1,
            bg=COLORS["bg_secondary"],
            fg=COLORS["text"],
            font=("Menlo", -9),
            relief=tk.FLAT,
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=14,
            padx=6,
            pady=6,
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
            font=("Helvetica Neue", -10),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_dim"],
            relief=tk.FLAT,
            cursor="hand2",
            command=self._clear_log,
        )
        clear_btn.pack(fill=tk.X, padx=6, pady=(0, 6))

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

    @staticmethod
    def _hotkey_keysym(key_name: str) -> str:
        """Tk keysym for a configured hotkey name ('f8' → 'F8', 'esc' → 'Escape')."""
        name = (key_name or "").strip().lower()
        named = {
            "esc": "Escape",
            "escape": "Escape",
            "space": "space",
            "enter": "Return",
            "return": "Return",
            "tab": "Tab",
            "backspace": "BackSpace",
            "delete": "Delete",
        }
        if name in named:
            return named[name]
        if len(name) > 1 and name[0] == "f" and name[1:].isdigit():
            return name.upper()
        return name

    def _bind_hotkey_key(self):
        """Point the panel's own key binding at the configured hotkey.

        This is the path that works while the panel has focus; the global
        listener in Controller is the one that works from inside Roblox. The
        binding used to be a hardcoded <KeyRelease-F6>, so rebinding the hotkey
        moved the global listener and left the panel still answering to F6.
        """
        keysym = self._hotkey_keysym(self.settings.killswitch_key)
        if keysym == self._bound_keysym:
            return
        if self._bound_keysym:
            try:
                self.root.unbind_all(f"<KeyRelease-{self._bound_keysym}>")
            except Exception:
                pass
        if not keysym:
            self._bound_keysym = None
            return
        try:
            self.root.bind_all(
                f"<KeyRelease-{keysym}>", lambda _event: self._toggle_from_hotkey()
            )
        except tk.TclError:
            logger.warning("Tk could not bind the hotkey key %r", keysym)
            self._bound_keysym = None
            return
        self._bound_keysym = keysym

    def _check_input_permission(self):
        """Warn when macOS will not let this process see or send input.

        Without Accessibility, pynput's listener never receives a key — it logs
        "This process is not trusted!" and nothing else — and pyautogui's clicks
        go nowhere. The panel still looks and behaves completely normally, so
        the only visible symptom is the hotkey doing nothing from inside the
        game. A downloaded .app needs its own grant: the permission belongs to
        the bundle's signature, not to the terminal that ran the checkout, and a
        fresh build is a new signature even at the same path.
        """
        if sys.platform != "darwin":
            return
        try:
            from ApplicationServices import AXIsProcessTrusted
        except ImportError:
            return
        try:
            trusted = bool(AXIsProcessTrusted())
        except Exception:
            return
        if trusted:
            return

        key = self.settings.killswitch_key.upper()
        self.header_killswitch_label.config(
            text="⚠ No Accessibility", foreground=COLORS["warning"]
        )
        self._append_log(
            f"⚠ Accessibility is not granted — {key} only works while this "
            "panel has focus, and clicks will not reach Roblox."
        )
        self._append_log(
            "Grant it in System Settings → Privacy & Security → Accessibility, "
            "then restart. Remove and re-add the entry after an update."
        )
        self.root.after(
            400,
            lambda: messagebox.showwarning(
                "Accessibility Permission Needed",
                "macOS is not letting Fisch Macro see or send input.\n\n"
                f"{key} will do nothing from inside Roblox, and the macro "
                "cannot click.\n\n"
                "System Settings → Privacy & Security → Accessibility, add "
                "this app, then restart it. If it is already listed after an "
                "update, remove the entry and add it again — the permission is "
                "tied to the build it was granted to.",
                parent=self.root,
            ),
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
        """Save current settings to disk (triggered by auto-save).

        ``active_profile`` is deliberately not written from here. Switching
        rods has to be able to ask before discarding unsaved regions, and to
        put the combobox back if the answer is no; an autosave trace on the
        same variable committed the switch before the question was asked.
        """
        try:
            self.settings.cast_hold_time = self.cast_time_var.get()
            self.settings.recast_delay = self.recast_delay_var.get()
            self.settings.scan_interval_ms = int(self.scan_interval_var.get())
            self.settings.auto_recast = self.auto_recast_var.get()
            self.settings.shake_enabled = self.shake_enabled_var.get()
            self.settings.humanize = self.humanize_var.get()
            self.settings.control_duty_kp = self.duty_kp_var.get()
            self.settings.control_duty_ki = self.duty_ki_var.get()
            self.settings.control_neutral_duty = self.neutral_duty_var.get()
            self.settings.control_lead_seconds = self.lead_seconds_var.get()
            self.settings.control_stationary_speed = self.stationary_speed_var.get()
            self.settings.control_bar_accel = self.bar_accel_var.get()
            self.settings.control_latency_seconds = self.latency_var.get()
            self.settings.control_max_brake_distance = self.max_brake_var.get()
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
            self.settings.show_live_vision = self.show_live_vision_var.get()

            self.config.save_settings(self.settings)
        except Exception as e:
            logger.error(f"Auto-save settings failed: {e}")

    # ─── Calibration ──────────────────────────────────────────────

    #: Region label, Settings attribute, ColorProfile attribute.
    CALIBRATION_REGIONS = (
        ("Reel bar", "bar_roi"),
        ("Progress", "progress_roi"),
        ("Shake", "shake_roi"),
    )

    @staticmethod
    def _roi_equal(a, b) -> bool:
        """Compare two ROIBounds, tolerating float round-trips through JSON."""
        if a is None or b is None:
            return a is b
        return all(
            abs(getattr(a, f) - getattr(b, f)) < 1e-9
            for f in ("x_start", "x_end", "y_start", "y_end")
        )

    def _unsaved_regions(self, profile):
        """Regions where the working calibration differs from the rod's own.

        Returns a list of (label, reason) pairs. ``reason`` is "never saved"
        when the rod has nothing stored for that region at all, which is what
        an older profile looks like.
        """
        out = []
        for label, attr in self.CALIBRATION_REGIONS:
            saved = getattr(profile, attr, None)
            if saved is None:
                out.append((label, "never saved"))
            elif not self._roi_equal(saved, getattr(self.settings, attr)):
                out.append((label, "changed"))
        return out

    def _calibration_readout_text(self, profile) -> str:
        """The numbers the macro will actually use, as displayed text."""
        lines = []
        for label, attr in self.CALIBRATION_REGIONS:
            roi = getattr(self.settings, attr)
            lines.append(
                f"{label:<9} x {roi.x_start * 100:5.1f}–{roi.x_end * 100:5.1f}%"
                f"   y {roi.y_start * 100:5.1f}–{roi.y_end * 100:5.1f}%"
            )

        shift = ""
        if self.settings.roi_shift_x or self.settings.roi_shift_y:
            shift = (
                f"  (shifted {self.settings.roi_shift_x * 100:+.1f}%, "
                f"{self.settings.roi_shift_y * 100:+.1f}%)"
            )
        if shift:
            lines.append(f"{'':<9}{shift.strip()}")

        lines.append("")
        lines.append(
            f"Fish      H {profile.fish_hsv_low[0]:>3}–{profile.fish_hsv_high[0]:<3}"
            f" S {profile.fish_hsv_low[1]:>3}–{profile.fish_hsv_high[1]:<3}"
            f" V {profile.fish_hsv_low[2]:>3}–{profile.fish_hsv_high[2]:<3}"
        )
        lines.append(
            f"Bar       H {profile.bar_hsv_low[0]:>3}–{profile.bar_hsv_high[0]:<3}"
            f" S {profile.bar_hsv_low[1]:>3}–{profile.bar_hsv_high[1]:<3}"
            f" V {profile.bar_hsv_low[2]:>3}–{profile.bar_hsv_high[2]:<3}"
        )
        lines.append(f"Brightness threshold  {profile.bar_brightness_threshold}")
        return "\n".join(lines)

    def _refresh_calibration_view(self):
        """Redraw the calibration readout, status line and button labels.

        This is the whole answer to "did my calibration stick?" — the tab now
        shows the numbers in use and says plainly whether the selected rod
        holds them.
        """
        if not hasattr(self, "calibration_readout"):
            return

        name = self.profile_var.get()
        try:
            profile = self.config.load_profile(name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Could not load profile %r: %s", name, exc)
            profile = None

        self.calibration_readout.config(state=tk.NORMAL)
        self.calibration_readout.delete("1.0", tk.END)
        if profile is None:
            self.calibration_readout.insert(tk.END, "No profile loaded.")
        else:
            self.calibration_readout.insert(
                tk.END, self._calibration_readout_text(profile)
            )
        self.calibration_readout.config(state=tk.DISABLED)

        self.save_calibration_btn.config(text=f'💾  Save to "{name}"')

        if profile is None:
            self.calibration_status.config(
                text="Rod profile could not be read.",
                foreground=COLORS["danger"],
            )
            self.revert_calibration_btn.config(state=tk.DISABLED)
            return

        pending = self._unsaved_regions(profile)
        if not pending:
            self.calibration_status.config(
                text=f'✓ Saved — rod "{name}" holds this calibration.',
                foreground=COLORS["accent_green"],
            )
            self.revert_calibration_btn.config(state=tk.DISABLED)
        else:
            detail = ", ".join(f"{label} ({why})" for label, why in pending)
            self.calibration_status.config(
                text=f'● Not saved yet — {detail}.\nSave to "{name}" to keep it.',
                foreground=COLORS["accent_yellow"],
            )
            self.revert_calibration_btn.config(state=tk.NORMAL)

    def _save_calibration(self):
        """Store the working ROI bounds in the selected rod profile.

        Colours are left alone: the calibration overlay writes those straight
        into the rod's own file when it closes, so re-copying them here from
        whatever happens to be active could overwrite one rod's colours with
        another's.
        """
        name = self.profile_var.get()
        try:
            profile = self.config.load_profile(name)
            profile.bar_roi = self.settings.bar_roi
            profile.progress_roi = self.settings.progress_roi
            profile.shake_roi = self.settings.shake_roi

            self.settings.active_profile = name
            self.config.save_settings(self.settings)
            self.config.save_profile(profile)
            self._refresh_calibration_view()
            self._append_log(f'Calibration saved to rod "{name}"')
        except Exception as e:
            logger.error("Saving calibration failed: %s", e, exc_info=True)
            messagebox.showerror("Calibration Error", f"Could not save: {e}")

    def _revert_calibration(self):
        """Put the selected rod's saved regions back into use."""
        name = self.profile_var.get()
        profile = self.config.load_profile(name)
        if all(
            getattr(profile, attr, None) is None
            for _label, attr in self.CALIBRATION_REGIONS
        ):
            messagebox.showinfo(
                "Nothing To Revert To",
                f'Rod "{name}" has no saved regions yet.\n\n'
                "Mark them with the calibration overlay, then save.",
            )
            return

        self._apply_profile_calibration(profile)
        self.config.save_settings(self.settings)
        self._refresh_calibration_view()
        self._append_log(f'Reverted to the calibration saved in rod "{name}"')

    def _confirm_discard_calibration(self, action: str) -> bool:
        """Ask before an action that would drop unsaved region changes."""
        previous = self.settings.active_profile
        profile = self.config.load_profile(previous)
        pending = self._unsaved_regions(profile)
        if not pending:
            return True

        detail = ", ".join(label for label, _why in pending)
        return messagebox.askyesno(
            "Unsaved Calibration",
            f'Rod "{previous}" does not have your current {detail} region(s).\n\n'
            f"{action} anyway and lose them?",
        )

    def _on_profile_change(self, _event):
        """Load the selected rod's saved regions into use."""
        profile_name = self.profile_var.get()
        if profile_name == self.settings.active_profile:
            self._refresh_calibration_view()
            return

        if not self._confirm_discard_calibration("Switch rod"):
            self.profile_var.set(self.settings.active_profile)
            return

        profile = self.config.load_profile(profile_name)
        self.settings.active_profile = profile_name
        self._apply_profile_calibration(profile)
        self.config.save_settings(self.settings)
        self._refresh_calibration_view()
        self._append_log(f'Loaded rod profile "{profile_name}"')

    def _delete_profile(self):
        """Delete the currently selected profile."""
        name = self.profile_var.get()
        if name == "default":
            messagebox.showwarning(
                "Cannot Delete", 'The "default" rod profile cannot be deleted.'
            )
            return

        if not messagebox.askyesno(
            "Delete Rod Profile",
            f'Delete the rod profile "{name}"?\n\n'
            "Its colours and saved regions are removed permanently.",
        ):
            return

        if not self.config.delete_profile(name):
            messagebox.showerror("Error", f'Could not delete rod profile "{name}".')
            return

        self._append_log(f'Deleted rod profile "{name}"')
        # The rod is gone, so there is nothing left to lose by switching.
        self.settings.active_profile = "default"
        self.profile_var.set("default")
        self.profile_combo.configure(values=self.config.list_profiles())
        self._apply_profile_calibration(self.config.load_profile("default"))
        self.config.save_settings(self.settings)
        self._refresh_calibration_view()

    def _apply_profile_calibration(self, profile):
        """Apply saved ROI bounds from a rod profile if present."""
        for _label, attr in self.CALIBRATION_REGIONS:
            saved = getattr(profile, attr, None)
            if saved is not None:
                setattr(self.settings, attr, saved)

    def _save_calibration_as_profile(self):
        """Create a new rod profile from the current calibration.

        The new rod is a full copy of the active one — every colour field, not
        a hand-listed subset. The old hand-written copy silently dropped the
        on-target and off-target colours back to library defaults, so a rod
        saved this way lost part of its calibration the moment it was created.
        """
        import dataclasses
        import re

        name = simpledialog.askstring(
            "New Rod Profile",
            "Name this rod:",
            initialvalue=self.profile_var.get(),
            parent=self.root,
        )
        if name is None:
            return

        safe_name = re.sub(r"[^A-Za-z0-9_. -]+", "", name.strip()).strip()
        safe_name = safe_name.replace("/", "-")
        if not safe_name:
            messagebox.showwarning("Invalid Name", "A rod profile needs a name.")
            return

        if safe_name in self.config.list_profiles() and not messagebox.askyesno(
            "Overwrite Rod Profile",
            f'A rod profile named "{safe_name}" already exists.\n\nOverwrite it?',
        ):
            return

        source = self.config.get_active_profile()
        profile = dataclasses.replace(
            source,
            name=safe_name,
            description=f"Calibration for {safe_name}",
            bar_roi=self.settings.bar_roi,
            progress_roi=self.settings.progress_roi,
            shake_roi=self.settings.shake_roi,
        )
        self.config.save_profile(profile)

        self.settings.active_profile = safe_name
        self.config.save_settings(self.settings)
        self.profile_var.set(safe_name)
        self.profile_combo.configure(values=self.config.list_profiles())
        self._refresh_calibration_view()
        self._append_log(f'Saved rod profile "{safe_name}"')

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
            self.profile_var.set(self.settings.active_profile)
            self._refresh_calibration_view()
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
        self._bind_hotkey_key()
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
        self.kill_button.config(text="■ E-STOP")

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
                    self._show_vision_placeholder("Start macro to see detection…")
                    self._set_vision_telemetry("")
                return

            snap = self.engine.get_vision_snapshot()

            if snap.frame_bgr is None:
                self._show_vision_placeholder("Waiting for the reel ROI…")
            else:
                rgb = cv2.cvtColor(snap.frame_bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                target_w = max(240, self.vision_image_label.winfo_width() or 320)
                scale = min(1.0, target_w / max(pil.width, 1))
                target_h = max(1, int(pil.height * scale))
                if pil.width != target_w or pil.height != target_h:
                    pil = pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                self._vision_photo = ImageTk.PhotoImage(pil)
                self._show_vision_image(self._vision_photo)

            # Updated even with no frame: while hunting, the numbers are the
            # only thing there is to see, and holding the previous fight's
            # telemetry there was actively misleading.
            self._set_vision_telemetry("\n".join(snap.summary_lines()))
        except Exception as exc:
            logger.debug("Vision panel refresh failed: %s", exc)

    def _show_vision_image(self, photo) -> None:
        """Put a rendered frame on the vision label at its full height.

        The label carries height=3 for its placeholder text, which Tk counts in
        lines. With an image in the label it counts the same number in pixels
        instead, so the panel asked for three pixels and cropped the frame to a
        sliver across the middle of the reel track — the catch-progress strip
        get_debug_frame pads onto the bottom was cut off entirely and looked
        like it was never drawn. Clearing the option lets the image size the
        label.
        """
        self.vision_image_label.config(image=photo, text="", height=0)

    def _show_vision_placeholder(self, text: str) -> None:
        """Drop back to the placeholder message, height in lines again."""
        self._vision_photo = None
        self.vision_image_label.config(image="", text=text, height=3)

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
