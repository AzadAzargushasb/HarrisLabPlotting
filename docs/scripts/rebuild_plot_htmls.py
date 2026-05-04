"""Rebuild interactive Plotly HTMLs to use a CDN-hosted plotly.js.

The HTMLs in this repo were exported with the full plotly.min.js inlined
(~4.1 MB each, ~62 MB across the 15 plots). For docs we want them to load
plotly.js from the CDN instead, dropping each file to ~200 KB while keeping
full interactivity.

This script:

1. Walks the source HTMLs in
       tutorial/output_legend_views/*.html
       test_files/tutorial_files/brain_modularity.html
2. Finds the inlined ``plotly.min.js`` ``<script>`` block.
3. Replaces it with ``<script src="https://cdn.plot.ly/plotly-<VERSION>.min.js"></script>``
   where ``<VERSION>`` is read from the bundle header.
4. Writes the rewritten file to ``docs/_static/plots/<name>.html``.

Run from the repo root::

    python docs/scripts/rebuild_plot_htmls.py

It is safe to run multiple times — already-CDN files are detected and
re-copied as-is. No data inputs are required: the postprocess works purely
on the existing HTML.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Files to migrate.
SOURCES: list[Path] = [
    *Path("tutorial/output_legend_views").glob("*.html"),
    Path("test_files/tutorial_files/brain_modularity.html"),
]
DEST_DIR = Path("docs/_static/plots")

# Match the inlined plotly bundle. The bundle starts with "/**\n* plotly.js vX.Y.Z"
# and is the single largest <script> block in the file.
INLINE_RE = re.compile(
    r"<script\s+type=\"text/javascript\">\s*/\*\*\s*"
    r"\*\s*plotly\.js\s+v(?P<version>[\w.\-+]+).*?"
    r"</script>",
    re.DOTALL,
)

# Detect already-CDN files so we don't double-rewrite.
ALREADY_CDN_RE = re.compile(r"<script\s+[^>]*src=\"https?://cdn\.plot\.ly/[^\"]+\"")


def rewrite(text: str) -> tuple[str, str | None]:
    """Return (rewritten_html, plotly_version) or (text, None) if no inline bundle."""
    if ALREADY_CDN_RE.search(text):
        return text, "cdn"
    match = INLINE_RE.search(text)
    if not match:
        return text, None
    version = match.group("version")
    cdn_tag = f'<script src="https://cdn.plot.ly/plotly-{version}.min.js" charset="utf-8"></script>'
    return INLINE_RE.sub(cdn_tag, text, count=1), version


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    DEST_DIR_ABS = repo_root / DEST_DIR
    DEST_DIR_ABS.mkdir(parents=True, exist_ok=True)

    sources = []
    for s in SOURCES:
        abs_path = repo_root / s
        if abs_path.exists():
            sources.append(abs_path)
        else:
            print(f"  skip (missing): {s}", file=sys.stderr)

    if not sources:
        print("No source HTMLs found.", file=sys.stderr)
        return 1

    converted = 0
    skipped = 0
    for src in sources:
        text = src.read_text(encoding="utf-8", errors="ignore")
        new_text, version = rewrite(text)
        out = DEST_DIR_ABS / src.name
        out.write_text(new_text, encoding="utf-8")
        if version == "cdn":
            print(f"  copy (already CDN): {src.name}")
            skipped += 1
        elif version:
            ratio = len(new_text) / max(len(text), 1)
            print(
                f"  rewrite (v{version}): {src.name} "
                f"[{len(text)/1024:.0f} KB → {len(new_text)/1024:.0f} KB, {ratio:.1%}]"
            )
            converted += 1
        else:
            print(f"  WARNING: no plotly bundle found in {src.name} (copied verbatim)")
            skipped += 1

    print(f"\nDone: {converted} rewritten, {skipped} copied as-is, into {DEST_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
