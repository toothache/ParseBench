"""Unit tests for Azure Content Understanding provider pure helpers.

Covers the deterministic post-processing logic that does NOT require network:
  * Chart.js figure config -> markdown table rendering (the chart-understanding path)
  * CU ``source`` polygon parsing + normalized-bbox conversion
  * ``contents`` extraction + layout_pages construction from a CU result

No live API calls — all inputs are hand-built CU-shaped dicts.
"""

from __future__ import annotations

import unittest

from parse_bench.inference.providers.parse.azure_content_understanding import (
    _build_layout_pages,
    _chartjs_to_markdown_table,
    _get_contents,
    _parse_source_polygon,
    _polygon_to_normalized_bbox,
    _render_chart_tables,
)


class TestChartJsToMarkdownTable(unittest.TestCase):
    """Chart.js config -> markdown table (the data the chart metric scores)."""

    def test_bar_chart_two_series_renders_all_values(self) -> None:
        content = {
            "type": "bar",
            "data": {
                "labels": ["Significantly lower", "About the same", "Higher"],
                "datasets": [
                    {"label": "2024", "data": [9, 27, 35]},
                    {"label": "2022", "data": [5, 45, 24]},
                ],
            },
        }
        md = _chartjs_to_markdown_table(content, caption="Partner turnover")
        # caption is emitted as a heading for context-aware label matching
        self.assertIn("### Partner turnover", md)
        # header carries both series names
        self.assertIn("| Category | 2024 | 2022 |", md)
        # every (category, series) value is present in its row
        self.assertIn("| Significantly lower | 9 | 5 |", md)
        self.assertIn("| About the same | 27 | 45 |", md)
        self.assertIn("| Higher | 35 | 24 |", md)

    def test_missing_series_name_falls_back_to_series_index(self) -> None:
        content = {
            "type": "bar",
            "data": {"labels": ["A", "B"], "datasets": [{"data": [1, 2]}]},
        }
        md = _chartjs_to_markdown_table(content)
        self.assertIn("Series 1", md)
        self.assertIn("| A | 1 |", md)
        self.assertIn("| B | 2 |", md)

    def test_scatter_points_flattened_to_x_y(self) -> None:
        content = {
            "type": "scatter",
            "data": {
                "labels": ["p1", "p2"],
                "datasets": [{"label": "S", "data": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}],
            },
        }
        md = _chartjs_to_markdown_table(content)
        self.assertIn("| p1 | 1, 2 |", md)
        self.assertIn("| p2 | 3, 4 |", md)

    def test_malformed_content_returns_empty_string(self) -> None:
        self.assertEqual(_chartjs_to_markdown_table({}), "")
        self.assertEqual(_chartjs_to_markdown_table({"data": {}}), "")
        self.assertEqual(_chartjs_to_markdown_table({"data": {"labels": ["a"], "datasets": []}}), "")

    def test_pipe_in_labels_and_values_is_escaped(self) -> None:
        # A literal "|" must not break markdown table structure (extra columns).
        content = {
            "type": "bar",
            "data": {
                "labels": ["Profit | Loss"],
                "datasets": [{"label": "FY | 24", "data": ["1|2"]}],
            },
        }
        md = _chartjs_to_markdown_table(content, caption="Rev | Cost")
        # every emitted line must have the same pipe count (structurally valid)
        body_lines = [ln for ln in md.splitlines() if ln.startswith("|")]
        pipe_counts = {ln.count("|") - ln.count("\\|") for ln in body_lines}
        self.assertEqual(len(pipe_counts), 1, f"ragged table: {body_lines}")
        # literal pipes are backslash-escaped, not dropped
        self.assertIn("Profit \\| Loss", md)
        self.assertIn("FY \\| 24", md)
        self.assertIn("1\\|2", md)
        self.assertIn("Rev \\| Cost", md)

    def test_newline_in_cell_flattened_to_space(self) -> None:
        content = {
            "type": "bar",
            "data": {"labels": ["a\nb"], "datasets": [{"label": "s", "data": [1]}]},
        }
        md = _chartjs_to_markdown_table(content)
        self.assertNotIn("a\nb", md)
        self.assertIn("a b", md)

    def test_series_shorter_than_labels_pads_blank(self) -> None:
        # dataset has fewer values than labels -> trailing rows get blank cells,
        # never an IndexError.
        content = {
            "type": "bar",
            "data": {"labels": ["A", "B", "C"], "datasets": [{"label": "s", "data": [1]}]},
        }
        md = _chartjs_to_markdown_table(content)
        self.assertIn("| A | 1 |", md)
        self.assertIn("| B |  |", md)
        self.assertIn("| C |  |", md)

    def test_render_chart_tables_only_processes_chart_figures(self) -> None:
        contents = [
            {
                "figures": [
                    {
                        "kind": "chart",
                        "content": {"data": {"labels": ["x"], "datasets": [{"label": "s", "data": [7]}]}},
                    },
                    {"kind": "image", "content": None},  # non-chart figure ignored
                    {"kind": "mermaid", "content": "graph TD; A-->B"},  # mermaid str ignored by table renderer
                ]
            }
        ]
        out = _render_chart_tables(contents)
        self.assertIn("| x | 7 |", out)
        self.assertNotIn("graph TD", out)


class TestSourcePolygon(unittest.TestCase):
    """CU ``source`` string parsing + normalized bbox math."""

    def test_parse_valid_source(self) -> None:
        src = "D(1,0.9866,1.0963,7.2713,1.0963,7.2713,1.4953,0.9866,1.4953)"
        parsed = _parse_source_polygon(src)
        self.assertIsNotNone(parsed)
        page, poly = parsed
        self.assertEqual(page, 1)
        self.assertEqual(len(poly), 8)
        self.assertAlmostEqual(poly[0], 0.9866)

    def test_parse_rejects_garbage(self) -> None:
        self.assertIsNone(_parse_source_polygon(""))
        self.assertIsNone(_parse_source_polygon("not-a-polygon"))
        self.assertIsNone(_parse_source_polygon("D(1,0.1,0.2)"))  # too few coords

    def test_polygon_to_normalized_bbox(self) -> None:
        # a rectangle from (1,1) to (3,2) on an 8x10 page
        poly = [1.0, 1.0, 3.0, 1.0, 3.0, 2.0, 1.0, 2.0]
        nx, ny, nw, nh = _polygon_to_normalized_bbox(poly, 8.0, 10.0)
        self.assertAlmostEqual(nx, 1.0 / 8.0)
        self.assertAlmostEqual(ny, 1.0 / 10.0)
        self.assertAlmostEqual(nw, 2.0 / 8.0)
        self.assertAlmostEqual(nh, 1.0 / 10.0)

    def test_zero_page_dims_safe(self) -> None:
        poly = [1.0, 1.0, 3.0, 1.0, 3.0, 2.0, 1.0, 2.0]
        nx, ny, nw, nh = _polygon_to_normalized_bbox(poly, 0.0, 0.0)
        self.assertEqual((nx, ny, nw, nh), (0.0, 0.0, 0.0, 0.0))


class TestContentsAndLayout(unittest.TestCase):
    """contents extraction + layout_pages construction."""

    def _sample_result(self) -> dict:
        return {
            "result": {
                "contents": [
                    {
                        "markdown": "# Title\n\nbody",
                        "startPageNumber": 1,
                        "pages": [{"pageNumber": 1, "width": 8.0, "height": 10.0}],
                        "paragraphs": [
                            {"role": "title", "content": "Title", "source": "D(1,1.0,1.0,3.0,1.0,3.0,2.0,1.0,2.0)"},
                            {"role": None, "content": "body", "source": "D(1,1.0,3.0,5.0,3.0,5.0,4.0,1.0,4.0)"},
                        ],
                        "tables": [],
                        "figures": [],
                    }
                ]
            }
        }

    def test_get_contents_handles_nested_and_flat(self) -> None:
        nested = self._sample_result()
        self.assertEqual(len(_get_contents(nested)), 1)
        flat = nested["result"]
        self.assertEqual(len(_get_contents(flat)), 1)
        self.assertEqual(_get_contents({}), [])

    def test_build_layout_pages_maps_roles_and_bboxes(self) -> None:
        contents = _get_contents(self._sample_result())
        pages = _build_layout_pages(contents)
        self.assertEqual(len(pages), 1)
        page = pages[0]
        self.assertEqual(page.page_number, 1)
        labels = {item.bbox.label for item in page.items if item.bbox}
        # 'title' role -> Canonical 'Title'; None role -> default 'Text'
        self.assertIn("Title", labels)
        self.assertIn("Text", labels)
        # bbox normalized against the 8x10 page
        title_item = next(it for it in page.items if it.bbox and it.bbox.label == "Title")
        self.assertAlmostEqual(title_item.bbox.x, 1.0 / 8.0)
        self.assertAlmostEqual(title_item.bbox.w, 2.0 / 8.0)


if __name__ == "__main__":
    unittest.main()
