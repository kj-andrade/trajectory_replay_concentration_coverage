from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL = 5.60
HALF = 2.72

C = {
    "blue":"#0072B2","orange":"#E69F00","green":"#009E73",
    "red":"#D55E00","purple":"#CC79A7","sky":"#56B4E9",
    "yellow":"#F0E442","black":"#000000","grey":"#666666",
    "light":"#B7B7B7",
}
CYCLE=[C["blue"],C["orange"],C["green"],C["red"],C["purple"],C["sky"]]
MARKERS=["o","s","^","D","v","P"]
LINES=["-","--","-.",":",(0,(3,1,1,1)),(0,(5,1))]

def use_style():
    """Apply the figure rcParams. Call once before any plotting."""
    plt.rcParams.update({
        # --- fonts -------------------------------------------------------
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,

        # --- axes --------------------------------------------------------
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#333333",
        "axes.labelpad": 3.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=CYCLE),

        # --- grid --------------------------------------------------------
        "axes.grid": True,
        "grid.color": "#B0B0B0",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "axes.axisbelow": True,

        # --- ticks -------------------------------------------------------
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,

        # --- lines -------------------------------------------------------
        "lines.linewidth": 1.3,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.0,

        # --- legend ------------------------------------------------------
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "legend.facecolor": "white",
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.9,
        "legend.handletextpad": 0.6,
        "legend.borderaxespad": 0.4,

        # --- output ------------------------------------------------------
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # embed TrueType, keeps text selectable
        "ps.fonttype": 42,
    })


def legend(ax, **kw):
    """Legend with the house defaults; pass loc/ncol as needed."""
    kw.setdefault("frameon", True)
    kw.setdefault("fancybox", False)
    leg = ax.legend(**kw)
    leg.get_frame().set_linewidth(0.5)
    return leg


def legend_above(ax, ncol, **kw):
    """Horizontal legend lifted clear of the axes, for full frames."""
    kw.setdefault("frameon", False)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                    ncol=ncol, columnspacing=1.4, **kw)
    return leg


def panel_label(ax, text, x=-0.16, y=1.03):
    ax.text(x,y,text,transform=ax.transAxes,fontsize=8.5,
            fontweight="bold",va="bottom",ha="left")

def save(fig,outdir,stem,tight=True):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    if tight: fig.tight_layout(pad=0.25)
    pdf=outdir/f"{stem}.pdf"; png=outdir/f"{stem}.png"
    fig.savefig(pdf); fig.savefig(png); plt.close(fig)
    return pdf,png

def style_key(i):
    return dict(color=CYCLE[i%len(CYCLE)],
                marker=MARKERS[i%len(MARKERS)],
                linestyle=LINES[i%len(LINES)])
