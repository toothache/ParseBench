from azure.ai.contentunderstanding.models import AnalysisResult

from parse_bench.evaluation.layout_adapters.adapters import _build_azure_cu_content
from parse_bench.inference.providers.parse.azure_content_understanding import (
    _build_layout_pages,
)


def test_layout_items_follow_wire_span_order():
    result = AnalysisResult(
        {
            "analyzerId": "prebuilt-layout",
            "apiVersion": "2025-11-01",
            "contents": [
                {
                    "kind": "document",
                    "mimeType": "application/pdf",
                    "markdown": "<table>...</table>\nFooter",
                    "startPageNumber": 1,
                    "endPageNumber": 1,
                    "unit": "inch",
                    "pages": [
                        {
                            "pageNumber": 1,
                            "width": 8.5,
                            "height": 11.0,
                            "unit": "inch",
                        }
                    ],
                    "paragraphs": [
                        {
                            "content": "cell content",
                            "source": "D(1,1,1,2,1,2,2,1,2)",
                            "span": {"offset": 10, "length": 12},
                        },
                        {
                            "content": "93",
                            "role": "pageNumber",
                            "source": "D(1,7,0.2,8,0.2,8,0.5,7,0.5)",
                            "span": {"offset": 101, "length": 2},
                        },
                        {
                            "content": "Page 2 of 28",
                            "role": "pageNumber",
                            "source": "D(1,7,10,8,10,8,10.5,7,10.5)",
                            "span": {"offset": 103, "length": 12},
                        },
                    ],
                    "tables": [
                        {
                            "rowCount": 1,
                            "columnCount": 1,
                            "cells": [
                                {
                                    "rowIndex": 0,
                                    "columnIndex": 0,
                                    "content": "cell content",
                                }
                            ],
                            "source": "D(1,0.5,0.5,6,0.5,6,9.5,0.5,9.5)",
                            "span": {"offset": 0, "length": 100},
                        }
                    ],
                }
            ],
        }
    )

    pages = _build_layout_pages(result.contents or [])

    assert len(pages) == 1
    assert [(item.type, item.value) for item in pages[0].items] == [
        ("table", "cell content"),
        ("text", "cell content"),
        ("text", "93"),
        ("text", "Page 2 of 28"),
    ]
    assert pages[0].items[2].bbox.label == "Page-header"
    assert pages[0].items[3].bbox.label == "Page-footer"


def test_figure_uses_referenced_paragraph_content():
    result = AnalysisResult(
        {
            "analyzerId": "prebuilt-layout",
            "apiVersion": "2025-11-01",
            "contents": [
                {
                    "kind": "document",
                    "mimeType": "application/pdf",
                    "markdown": "",
                    "startPageNumber": 1,
                    "endPageNumber": 1,
                    "unit": "inch",
                    "pages": [
                        {
                            "pageNumber": 1,
                            "width": 8.5,
                            "height": 11.0,
                            "unit": "inch",
                        }
                    ],
                    "paragraphs": [
                        {
                            "content": "37",
                            "source": "D(1,1,1,2,1,2,2,1,2)",
                            "span": {"offset": 10, "length": 2},
                        },
                        {
                            "content": "2023",
                            "source": "D(1,1,3,2,3,2,4,1,4)",
                            "span": {"offset": 13, "length": 4},
                        },
                    ],
                    "figures": [
                        {
                            "id": "1.1",
                            "source": "D(1,0.5,0.5,6,0.5,6,5,0.5,5)",
                            "elements": ["/paragraphs/0", "/paragraphs/1"],
                            "kind": "chart",
                            "content": {
                                "type": "bar",
                                "data": {"labels": ["37", "2023"]},
                            },
                            "span": {"offset": 10, "length": 7},
                        }
                    ],
                }
            ],
        }
    )

    pages = _build_layout_pages(result.contents or [])

    picture = next(item for item in pages[0].items if item.type == "image")
    assert picture.value == "37 2023"
    assert _build_azure_cu_content("Picture", picture.value).text == "37 2023"
    assert _build_azure_cu_content("Picture", "") is None
