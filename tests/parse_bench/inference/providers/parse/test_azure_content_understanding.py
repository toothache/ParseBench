"""Regression tests for Azure Content Understanding normalization."""

from azure.ai.contentunderstanding.models import AnalysisResult

from parse_bench.inference.providers.parse.azure_content_understanding import (
    _build_layout_pages,
    _make_page_furniture_visible,
)


def test_page_furniture_comments_become_visible_markdown() -> None:
    markdown = (
        "<!-- PageHeader: Quarterly Report 2026 -->\n"
        "Body text\n"
        "<!-- PageFooter: [Example](https://example.com) -->\n"
        "<!-- PageNumber: 3/4 -->"
    )

    assert _make_page_furniture_visible(markdown) == (
        "Quarterly Report 2026\n"
        "Body text\n"
        "[Example](https://example.com)\n"
        "3/4"
    )


def test_unknown_and_malformed_comments_are_preserved() -> None:
    markdown = "<!-- ReviewerNote: keep me -->\n<!-- PageHeader missing colon -->"

    assert _make_page_furniture_visible(markdown) == markdown


def test_layout_pages_keep_page_furniture_roles_and_dedicated_fields() -> None:
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
                        },
                        {
                            "content": "Confidential",
                            "role": "pageFooter",
                            "source": "D(1,0,9,1,9,1,10,0,10)",
                        },
                        {
                            "content": "3/4",
                            "role": "pageNumber",
                            "source": "D(1,7,9,8,9,8,10,7,10)",
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
