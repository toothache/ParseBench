"""Regression tests for Azure Content Understanding normalization."""

from types import SimpleNamespace

import pytest
from azure.ai.contentunderstanding.models import AnalysisResult

from parse_bench.inference.providers.parse.azure_content_understanding import (
    _build_layout_pages,
    render_content_markdown,
)


def _paragraph(markdown: str, role: str, source_text: str, content: str) -> SimpleNamespace:
    start = markdown.index(source_text)
    return SimpleNamespace(
        role=role,
        content=content,
        span=SimpleNamespace(offset=start, length=len(source_text)),
    )


def test_page_furniture_uses_typed_paragraph_content() -> None:
    header = "<!-- PageHeader: R&amp;D -->"
    footer = "<!-- PageFooter: Confidential -->"
    page_number = "<!-- PageNumber: 3/4 -->"
    markdown = f"{header}\nBody\n{footer}\n{page_number}"
    content = SimpleNamespace(
        markdown=markdown,
        figures=[],
        paragraphs=[
            _paragraph(markdown, "pageHeader", header, "R&D"),
            _paragraph(markdown, "pageFooter", footer, "Confidential"),
            _paragraph(markdown, "pageNumber", page_number, "3/4"),
        ],
    )

    assert render_content_markdown(content) == "R&D\nBody\nConfidential\n3/4"


def test_other_paragraph_roles_and_comments_are_unchanged() -> None:
    title = "# Visible title"
    note = "<!-- ReviewerNote: keep me -->"
    markdown = f"{title}\n{note}"
    content = SimpleNamespace(
        markdown=markdown,
        figures=[],
        paragraphs=[_paragraph(markdown, "title", title, "Visible title")],
    )

    assert render_content_markdown(content) == markdown


def test_page_furniture_and_chart_edits_share_original_offsets() -> None:
    header = "<!-- PageHeader: Report -->"
    chart = "chart placeholder"
    page_number = "<!-- PageNumber: 2 -->"
    markdown = f"{header}\n{chart}\n{page_number}"
    content = SimpleNamespace(
        markdown=markdown,
        paragraphs=[
            _paragraph(markdown, "pageHeader", header, "Report"),
            _paragraph(markdown, "pageNumber", page_number, "2"),
        ],
        figures=[
            {
                "kind": "chart",
                "content": {
                    "type": "bar",
                    "data": {
                        "labels": ["Q1"],
                        "datasets": [{"label": "Revenue", "data": [10]}],
                    },
                },
                "span": {"offset": markdown.index(chart), "length": len(chart)},
            }
        ],
    )

    rendered = render_content_markdown(content)

    assert rendered.startswith("Report\n")
    assert "| Label | Revenue |" in rendered
    assert rendered.endswith("\n2")


def test_invalid_page_furniture_span_fails_explicitly() -> None:
    content = SimpleNamespace(
        markdown="Body",
        figures=[],
        paragraphs=[
            SimpleNamespace(
                role="pageHeader",
                content="Header",
                span=SimpleNamespace(offset=10, length=4),
            )
        ],
    )

    with pytest.raises(ValueError, match="invalid Markdown span"):
        render_content_markdown(content)


def test_layout_pages_keep_page_furniture_roles_and_fields() -> None:
    result = AnalysisResult(
        {
            "contents": [
                {
                    "kind": "document",
                    "markdown": "Body",
                    "pages": [{"pageNumber": 1, "width": 8.5, "height": 11}],
                    "paragraphs": [
                        {
                            "content": "Quarterly Report",
                            "role": "pageHeader",
                            "source": "D(1,0,0,1,0,1,1,0,1)",
                            "span": {"offset": 0, "length": 4},
                        },
                        {
                            "content": "Confidential",
                            "role": "pageFooter",
                            "source": "D(1,0,9,1,9,1,10,0,10)",
                            "span": {"offset": 0, "length": 4},
                        },
                        {
                            "content": "3/4",
                            "role": "pageNumber",
                            "source": "D(1,7,9,8,9,8,10,7,10)",
                            "span": {"offset": 0, "length": 4},
                        },
                    ],
                }
            ]
        }
    )

    page = _build_layout_pages(result.contents)[0]

    assert page.page_header_markdown == "Quarterly Report"
    assert page.page_footer_markdown == "Confidential\n3/4"
    assert page.printed_page_number == "3/4"
    assert [item.bbox.label for item in page.items] == [
        "Page-header",
        "Page-footer",
        "Page-footer",
    ]
