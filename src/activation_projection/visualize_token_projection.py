import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from typing import List, Optional


def plot_token_badges(
    tokens: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    *,
    # two-block mode (stacked vertically)
    prompt_tokens: Optional[List[str]] = None,
    prompt_scores: Optional[List[float]] = None,
    response_tokens: Optional[List[str]] = None,
    response_scores: Optional[List[float]] = None,
    group_vgap_lines: float = 0.8,      # vertical gap (in line-heights) between prompt and response

    # optional masks marking special tokens (rendered with dashed borders)
    special_mask: Optional[List[bool]] = None,
    prompt_special_mask: Optional[List[bool]] = None,
    response_special_mask: Optional[List[bool]] = None,

    # layout/appearance
    wrap_chars: int = 200,
    cmap: str = "RdBu_r",
    fontsize: int = 10,
    alpha: float = 0.9,
    family: str = "DejaVu Sans Mono",
    line_height_mult: float = 1.25,
    box_pad: float = 0.12,
    gap_em: float = 0.45,
    border_pad_pts: float = 4,
    show_colorbar: bool = True,
):
    """
    Renders either a single sequence (tokens/scores) or two sequences
    (prompt_* above response_*) stacked vertically with a configurable gap.
    """

    two_block = (prompt_tokens is not None or response_tokens is not None)


    norm = Normalize(vmin=-30.0, vmax=30.0)
    C = cm.get_cmap(cmap)

    fp = FontProperties(family=family, size=fontsize)

    def token_width_pts(t) -> float:
        if not isinstance(t, str):
            t = "".join(map(str, t)) if isinstance(t, (list, tuple)) else str(t)
        bb = TextPath((0, 0), t, prop=fp, usetex=False).get_extents()
        return bb.width

    em = fontsize
    gap_pts = gap_em * em
    line_h = line_height_mult * fontsize

    approx_char_w = 0.6 * fontsize
    wrap_width_pts = wrap_chars * approx_char_w

    def layout_rows(seq_tokens: List[str]):
        widths_local = [token_width_pts(t) for t in seq_tokens]
        rows = []
        i = 0
        while i < len(seq_tokens):
            row = []
            used = 0.0
            while i < len(seq_tokens):
                w = widths_local[i]
                if row and (used + (gap_pts if used > 0 else 0) + w) > wrap_width_pts:
                    break
                if used > 0:
                    used += gap_pts
                row.append(i)
                used += w
                i += 1
            rows.append(row)
        row_widths = [
            (sum(widths_local[j] for j in row) + gap_pts * (len(row) - 1)) if row else 0.0
            for row in rows
        ]
        return rows, widths_local, row_widths

    if two_block:
        p_rows, p_widths, p_row_widths = layout_rows(prompt_tokens)
        r_rows, r_widths, r_row_widths = layout_rows(response_tokens)

        vgap_pts = group_vgap_lines * line_h
        total_height = (len(p_rows) * line_h) + vgap_pts + (len(r_rows) * line_h)
        max_row_width = max(p_row_widths + r_row_widths) if (p_row_widths or r_row_widths) else 0.0
    else:
        rows, widths, row_widths = layout_rows(tokens)
        total_height = len(rows) * line_h
        max_row_width = max(row_widths) if row_widths else 0.0

    fig_w_in = max_row_width / 72 + 1.0
    fig_h_in = total_height / 72 + 1.0
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=120)
    ax.axis("off")

    ax.set_xlim(0 - border_pad_pts, max_row_width + border_pad_pts)
    ax.set_ylim(total_height + border_pad_pts, 0 - border_pad_pts)  # invert y

    def _bbox(score: float, is_special: bool) -> dict:
        r, g, b, _ = C(norm(score))
        if is_special:
            return dict(
                boxstyle=f"round,pad={box_pad},rounding_size=0.15",
                facecolor=(r, g, b, float(np.clip(abs(score), 0.0, 1.0)) * alpha),
                edgecolor=(0, 0, 0, 0.85),
                linewidth=1.2,
                linestyle="--",
            )
        return dict(
            boxstyle=f"round,pad={box_pad},rounding_size=0.15",
            facecolor=(r, g, b, float(np.clip(abs(score), 0.0, 1.0)) * alpha),
            edgecolor=(0, 0, 0, 0.12),
        )

    # draw
    y = 0.0
    if two_block:
        # --- prompt ---
        for row in p_rows:
            x = 0.0
            for k, idx in enumerate(row):
                t = prompt_tokens[idx]
                s = float(prompt_scores[idx])
                is_special = bool(prompt_special_mask[idx]) if prompt_special_mask is not None else False
                ax.text(
                    x, y, t, fontproperties=fp, va="top", ha="left",
                    bbox=_bbox(s, is_special),
                )
                x += p_widths[idx]
                if k < len(row) - 1:
                    x += gap_pts
            y += line_h

        # vertical gap
        y += group_vgap_lines * line_h

        # --- response ---
        for row in r_rows:
            x = 0.0
            for k, idx in enumerate(row):
                t = response_tokens[idx]
                s = float(response_scores[idx])
                is_special = bool(response_special_mask[idx]) if response_special_mask is not None else False
                ax.text(
                    x, y, t, fontproperties=fp, va="top", ha="left",
                    bbox=_bbox(s, is_special),
                )
                x += r_widths[idx]
                if k < len(row) - 1:
                    x += gap_pts
            y += line_h

    else:
        for row in rows:
            x = 0.0
            for k, idx in enumerate(row):
                t = tokens[idx]
                s = float(scores[idx])
                is_special = bool(special_mask[idx]) if special_mask is not None else False
                ax.text(
                    x, y, t, fontproperties=fp, va="top", ha="left",
                    bbox=_bbox(s, is_special),
                )
                x += widths[idx]
                if k < len(row) - 1:
                    x += gap_pts
            y += line_h

    # compact colorbar
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.04, pad=0.02, shrink=0.7)
        cbar.set_label("Score", rotation=270, labelpad=10)
        cbar.ax.set_yticks([-100, 0, 100])
        cbar.ax.set_yticklabels(["-100", "0", "+100"])
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(pad=0.01)
    return fig

