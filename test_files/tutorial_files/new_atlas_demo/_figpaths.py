"""Shared two-version output config for the figure-creation render scripts.

The committed docs images are rendered at **150 DPI** (small, web-friendly). A
full-resolution **600 DPI "publication"** copy is written to a *git-ignored*
`publication/` subdir alongside them, so you keep print-quality figures locally
without bloating the repo.

Both are driven by environment variables so a driver can run each render script
once per target without editing it:

  HLP_FIG_DPI  -> render DPI                  (default 150)
  HLP_FIG_PUB  -> "" for the committed tree,  (default "")
                  "publication" for the git-ignored full-res copy

Typical use::

  HLP_FIG_DPI=150                     python render_figures.py   # committed
  HLP_FIG_DPI=600 HLP_FIG_PUB=publication python render_figures.py   # local, git-ignored
"""
import os
from pathlib import Path

FIG_DPI = int(os.environ.get("HLP_FIG_DPI", "150"))
_PUB = os.environ.get("HLP_FIG_PUB", "").strip()


def fig_root(repo_root) -> Path:
    """Return the figure-creation image root for the current target.

    ``docs/images/figure_creation`` for the committed tree, or
    ``docs/images/figure_creation/publication`` when ``HLP_FIG_PUB=publication``.
    """
    base = Path(repo_root) / "docs" / "images" / "figure_creation"
    return base / _PUB if _PUB else base
