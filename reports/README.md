# reports/

Scripts that convert pipeline **output data** (JSON artifacts) into
human-readable deliverables: PDFs, plots, tables, summary assets.

| File | Purpose |
|------|---------|
| `pdf_utils.py` | Shared PDF rendering helpers (matplotlib setup, text/command/info pages, tables, manifest) |
| `testing3d.py` | 3-D energy-visualization PDF from hardcoded pipeline JSON |
| `compare_jsons.py` | Side-by-side JSON consistency checker (terminal + optional PDF) |
| `guide_assets.py` | Generates repo-implementation-guide PNGs and summary JSONs |

## `pdf_utils.py` — shared rendering module

All four pipeline files (`hubbard_pipeline.py`, `adapt_pipeline.py`,
`compare_hc_vs_qk.py`, `qiskit_baseline.py`) import from this module
instead of duplicating ~80 lines of matplotlib setup and page helpers.

Key exports:
- `HAS_MATPLOTLIB`, `require_matplotlib()`, `get_plt()`, `get_PdfPages()`
- `render_text_page()`, `render_command_page()`, `render_info_page()`
- `render_compact_table()`, `render_parameter_manifest()`
- `current_command_string()`
