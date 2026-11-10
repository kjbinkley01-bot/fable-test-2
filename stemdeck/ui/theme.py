"""Colours and the application style sheet (dark theme)."""

BG = "#14161a"
PANEL = "#1b1e24"
PANEL_ALT = "#22262d"
BORDER = "#2e333c"
TEXT = "#e6e9ee"
TEXT_DIM = "#8a93a0"
ACCENT = "#4ea8ff"
WARN = "#ffb020"
ERROR = "#ff5c5c"
OK = "#4fd6a0"

STYLE = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QMainWindow, QDialog {{ background: {BG}; }}
QLabel {{ background: transparent; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QFrame#strip {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QPushButton {{
    background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 5px;
    padding: 6px 12px; min-height: 18px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: #5a626e; border-color: {PANEL_ALT}; }}
QPushButton#play {{ font-size: 15px; font-weight: 600; min-width: 96px; padding: 8px 14px; }}
QPushButton#play:checked {{ background: {ACCENT}; color: #0b1220; border-color: {ACCENT}; }}
QPushButton#small {{ min-width: 30px; max-width: 30px; padding: 4px 0; }}
QPushButton#mute, QPushButton#solo {{ min-width: 26px; max-width: 26px; padding: 3px 0; font-weight: 700; }}
QPushButton#mute:checked {{ background: {ERROR}; color: #1a0b0b; border-color: {ERROR}; }}
QPushButton#solo:checked {{ background: {WARN}; color: #1a1205; border-color: {WARN}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #5b6472; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {TEXT}; width: 12px; height: 16px; margin: -7px 0; border-radius: 3px;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}
QProgressBar {{
    background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 4px;
    text-align: center; height: 16px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QComboBox {{ background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 5px; padding: 4px 8px; }}
QComboBox QAbstractItemView {{ background: {PANEL_ALT}; selection-background-color: {ACCENT}; }}
QScrollBar:horizontal {{ background: {PANEL}; height: 12px; }}
QScrollBar::handle:horizontal {{ background: #4a515c; border-radius: 5px; min-width: 30px; margin: 2px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QCheckBox, QRadioButton {{ background: transparent; spacing: 6px; }}
QToolTip {{ background: {PANEL_ALT}; color: {TEXT}; border: 1px solid {BORDER}; }}
QLabel#big {{ font-size: 22px; font-weight: 600; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
QLabel#title {{ font-size: 16px; font-weight: 600; }}
QLabel#banner {{
    background: #3a2c0c; color: {WARN}; border: 1px solid #6b5216; border-radius: 5px; padding: 6px 10px;
}}
"""
