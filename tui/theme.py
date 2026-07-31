"""FORGE's colour identity.

Two registered Textual themes rather than one, so `ctrl+t` has somewhere to go
and the app is legible on a light terminal.

The palette is deliberately *desaturated*: a coding agent's transcript is read
for minutes at a time, and stock terminal primaries (pure #00ff00, #ff0000) are
exhausting at that duration. Every hue here is pulled toward grey, and the
foreground is a soft grey rather than white so that genuinely important things —
the accent rail, an error — have somewhere brighter to go.

The rule that makes this real: **`forge.tcss` must never contain a literal
colour.** Anything hardcoded there is a spot the theme switch cannot reach.
"""

from __future__ import annotations

from textual.theme import Theme

# Slate + periwinkle. Background carries a faint blue cast rather than being
# pure black, which reads as a surface instead of a hole.
FORGE_DARK = Theme(
    name="forge-dark",
    primary="#8b9cf7",      # periwinkle — the one saturated note
    secondary="#7fbf9f",    # sage
    accent="#8b9cf7",
    foreground="#d2d6e0",   # soft grey, never #ffffff
    background="#15171c",
    surface="#1b1e25",
    panel="#232730",
    success="#7fbf9f",
    warning="#e0b877",      # sand
    error="#e08c8c",        # dusty rose
    dark=True,
    variables={
        "text-muted": "#767d8d",
        "block-cursor-background": "#8b9cf7",
        "block-cursor-foreground": "#15171c",
        "block-cursor-text-style": "none",
        "input-selection-background": "#8b9cf7 35%",
        "scrollbar": "#2b2f39",
        "scrollbar-hover": "#3a3f4b",
        "scrollbar-active": "#8b9cf7",
        "border": "#2b2f39",
        # The gutter rails. Named here rather than in CSS so both themes can
        # tune them independently.
        "rail-user": "#8b9cf7",
        "rail-assistant": "#3a3f4b",
    },
)

# The same hues inverted for a light terminal. Accents darken rather than
# lighten — periwinkle on white is unreadable at the same value.
FORGE_LIGHT = Theme(
    name="forge-light",
    primary="#4c5bb5",
    secondary="#3f8f68",
    accent="#4c5bb5",
    foreground="#2b2f38",
    background="#f4f5f7",
    surface="#ffffff",
    panel="#e8eaee",
    success="#3f8f68",
    warning="#a3701f",
    error="#b05252",
    dark=False,
    variables={
        "text-muted": "#6b7280",
        "block-cursor-background": "#4c5bb5",
        "block-cursor-foreground": "#ffffff",
        "block-cursor-text-style": "none",
        "input-selection-background": "#4c5bb5 25%",
        "scrollbar": "#d3d6dc",
        "scrollbar-hover": "#bcc0c8",
        "scrollbar-active": "#4c5bb5",
        "border": "#d3d6dc",
        "rail-user": "#4c5bb5",
        "rail-assistant": "#c8ccd4",
    },
)

THEMES = (FORGE_DARK, FORGE_LIGHT)
DEFAULT_THEME = FORGE_DARK.name


def next_theme(current: str) -> str:
    """Cycle within FORGE's own themes.

    Deliberately not Textual's stock light/dark: switching into `textual-dark`
    would drop every variable `forge.tcss` depends on (`$rail-user` and friends)
    and half-render the app.
    """
    names = [theme.name for theme in THEMES]
    if current not in names:
        return DEFAULT_THEME
    return names[(names.index(current) + 1) % len(names)]
