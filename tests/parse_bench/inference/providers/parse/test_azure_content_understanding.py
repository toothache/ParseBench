from types import SimpleNamespace

from parse_bench.inference.providers.parse.azure_content_understanding import (
    render_content_markdown,
)


def _paragraph(markdown: str, text: str, *, role: str, content: str | None = None):
    start = markdown.index(text)
    return SimpleNamespace(
        role=role,
        content=text if content is None else content,
        span=SimpleNamespace(offset=start, length=len(text)),
    )


def _content(markdown: str, *paragraphs):
    return SimpleNamespace(markdown=markdown, paragraphs=paragraphs, figures=[])


def test_render_content_markdown_exposes_standalone_page_furniture():
    markdown = "Body\n\n<!-- PageFooter: Company -->"
    paragraph = _paragraph(
        markdown,
        "<!-- PageFooter: Company -->",
        role="pageFooter",
        content="Company",
    )

    assert render_content_markdown(_content(markdown, paragraph)) == "Body\n\nCompany"


def test_render_content_markdown_keeps_figure_alt_handling_unchanged():
    markdown = "![Claude<!-- PageFooter: ABNASIA.ORG -->\n](figures/1.1)"
    paragraph = _paragraph(
        markdown,
        "<!-- PageFooter: ABNASIA.ORG -->",
        role="pageFooter",
        content="ABNASIA.ORG",
    )

    assert render_content_markdown(_content(markdown, paragraph)) == (
        "![ClaudeABNASIA.ORG\n](figures/1.1)"
    )


def test_render_content_markdown_keeps_parent_and_skips_nested_edit(caplog):
    markdown = "<!-- PageHeader: Parent CHILD -->"
    parent = _paragraph(
        markdown,
        markdown,
        role="pageHeader",
        content="Parent CHILD",
    )
    child = _paragraph(
        markdown,
        "CHILD",
        role="pageHeader",
        content="child",
    )

    assert render_content_markdown(_content(markdown, parent, child)) == "Parent CHILD"
    assert "skipping_overlapping_cu_markdown_edit" in caplog.text


def test_render_content_markdown_skips_partial_overlap(caplog):
    markdown = "abcdefghij"
    parent = SimpleNamespace(
        role="pageHeader",
        content="PARENT",
        span=SimpleNamespace(offset=4, length=6),
    )
    partial = SimpleNamespace(
        role="pageHeader",
        content="partial",
        span=SimpleNamespace(offset=2, length=4),
    )

    assert render_content_markdown(_content(markdown, parent, partial)) == "abcdPARENT"
    assert "skipping_overlapping_cu_markdown_edit" in caplog.text


def test_render_content_markdown_keeps_reverse_applied_edits_disjoint():
    markdown = "<!-- PageHeader: Header -->\n\nBody\n\n<!-- PageNumber: 3 -->"
    header = _paragraph(
        markdown,
        "<!-- PageHeader: Header -->",
        role="pageHeader",
        content="Header",
    )
    page_number = _paragraph(
        markdown,
        "<!-- PageNumber: 3 -->",
        role="pageNumber",
        content="3",
    )

    assert render_content_markdown(_content(markdown, header, page_number)) == (
        "Header\n\nBody\n\n3"
    )
