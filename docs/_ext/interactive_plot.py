"""Custom Sphinx directive: ``interactive-plot``.

Renders a paired screenshot + lazy-loaded interactive Plotly iframe inside a
sphinx-design tab block.

Usage (MyST)::

    :::{interactive-plot}
    :html: plots/mv_mod_dummy.html
    :image: images/legend_tutorial/09_mv_modular.png
    :alt: 3-view modular brain network
    :caption: Multi-view stitched export with modular edge coloring.
    :height: 520
    :::

Both ``:html:`` and ``:image:`` are resolved relative to ``_static/``. Either
may be omitted: missing ``:html:`` falls back to a screenshot-only block;
missing ``:image:`` falls back to interactive-only.
"""

from __future__ import annotations

import itertools

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective

# Each directive instance gets a unique id so its tab radio group does not
# collide with sibling blocks on the same page.
_uid_counter = itertools.count()


def _flag(arg):
    """Treat ``yes/true/1`` as truthy, anything else as falsy."""
    if arg is None:
        return True
    return str(arg).strip().lower() in {"yes", "true", "1", "on"}


class InteractivePlot(SphinxDirective):
    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        "html": directives.unchanged,
        "image": directives.unchanged,
        "alt": directives.unchanged,
        "caption": directives.unchanged,
        "height": directives.unchanged,
        "width": directives.unchanged,
        "default": directives.unchanged,  # "screenshot" (default) or "interactive"
        "no-download": _flag,
    }

    def run(self):
        html_rel = (self.options.get("html") or "").strip()
        image_rel = (self.options.get("image") or "").strip()
        alt = self.options.get("alt", "").strip()
        caption = self.options.get("caption", "").strip()
        height = (self.options.get("height") or "520").strip()
        width = (self.options.get("width") or "100%").strip()
        default = (self.options.get("default") or "screenshot").strip().lower()
        no_download = self.options.get("no-download", False)

        if not html_rel and not image_rel:
            err = self.state.document.reporter.error(
                "interactive-plot directive requires at least one of :html: or :image:.",
                line=self.lineno,
            )
            return [err]

        # Resolve paths against _static/, prefixed with enough "../" segments
        # to reach the docs root from the current page. Without this, embeds
        # on any doc deeper than index.html resolve to <currentdir>/_static/...
        # and 404. self.env.docname is e.g. "getting_started/quickstart" or
        # "gallery/notebooks/quickstart"; its slash count is the depth.
        depth = self.env.docname.count("/")
        prefix = "../" * depth
        html_url = f"{prefix}_static/{html_rel}" if html_rel else ""
        image_url = f"{prefix}_static/{image_rel}" if image_rel else ""

        screenshot_active = (default != "interactive") and bool(image_url)
        interactive_active = not screenshot_active and bool(html_url)

        # If user gave only one asset, render a single-pane block, not tabs.
        uid = next(_uid_counter)

        if html_url and not image_url:
            block_html = self._iframe_only_html(html_url, height, width, no_download)
        elif image_url and not html_url:
            block_html = self._image_only_html(image_url, alt)
        else:
            block_html = self._tab_html(
                html_url,
                image_url,
                alt,
                height,
                width,
                screenshot_active,
                interactive_active,
                no_download,
                uid,
            )

        if caption:
            block_html += (
                f'\n<div class="plot-caption">{caption}</div>'
            )

        wrapper = nodes.container(classes=["interactive-plot"])
        wrapper += nodes.raw("", block_html, format="html")
        return [wrapper]

    @staticmethod
    def _iframe_only_html(html_url, height, width, no_download):
        download = ""
        if not no_download:
            download = (
                f'<div class="plot-links">'
                f'<a href="{html_url}" target="_blank" rel="noopener">↗ Open in new tab</a>'
                f'<a href="{html_url}" download>⬇ Download HTML</a>'
                f"</div>"
            )
        return (
            f'<iframe loading="lazy" src="{html_url}" '
            f'style="width:{width}; height:{height}px; border:0;" '
            f'allowfullscreen></iframe>{download}'
        )

    @staticmethod
    def _image_only_html(image_url, alt):
        alt_attr = alt or ""
        return f'<img src="{image_url}" alt="{alt_attr}" loading="lazy" />'

    @staticmethod
    def _tab_html(
        html_url,
        image_url,
        alt,
        height,
        width,
        screenshot_active,
        interactive_active,
        no_download,
        uid,
    ):
        ss_active = " active" if screenshot_active else ""
        iv_active = " active" if interactive_active else ""

        download = ""
        if not no_download:
            download = (
                f'<div class="plot-links">'
                f'<a href="{html_url}" target="_blank" rel="noopener">↗ Open in new tab</a>'
                f'<a href="{html_url}" download>⬇ Download HTML</a>'
                f"</div>"
            )

        # Each tab block gets its own radio-group name (`ip-group-{uid}`) so
        # blocks on the same page don't toggle each other.
        group = f"ip-group-{uid}"
        ss_id = f"ip-tab-ss-{uid}"
        iv_id = f"ip-tab-iv-{uid}"

        return f"""
<div class="sd-tab-set">
  <input type="radio" name="{group}"
         id="{ss_id}"
         class="sd-tab-input"{' checked' if screenshot_active else ''}>
  <label class="sd-tab-label sd-btn sd-btn-primary{ss_active}"
         for="{ss_id}">Screenshot</label>
  <div class="sd-tab-content">
    <img src="{image_url}" alt="{alt}" loading="lazy" />
  </div>

  <input type="radio" name="{group}"
         id="{iv_id}"
         class="sd-tab-input"{' checked' if interactive_active else ''}>
  <label class="sd-tab-label sd-btn sd-btn-primary{iv_active}"
         for="{iv_id}">Interactive</label>
  <div class="sd-tab-content">
    <iframe loading="lazy" src="{html_url}"
            style="width:{width}; height:{height}px; border:0;"
            allowfullscreen></iframe>
    {download}
  </div>
</div>
""".strip()


def setup(app):
    app.add_directive("interactive-plot", InteractivePlot)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
