import json

from parse_bench.analysis.detailed_report import _build_data_blob
from parse_bench.schemas.evaluation import EvaluationResult, EvaluationSummary


def test_expected_output_prefers_sidecar_then_falls_back_to_jsonl(tmp_path):
    (tmp_path / "sample.test.json").write_text(
        json.dumps({"expected_markdown": "sidecar expected"}),
        encoding="utf-8",
    )
    records = [
        {
            "pdf": "docs/table/sample.pdf",
            "expected_markdown": "jsonl should not replace sidecar",
        },
        {
            "pdf": "docs/table/fallback.pdf",
            "expected_markdown": "jsonl expected",
        },
        {
            "pdf": "docs/layout/checks.pdf",
            "type": "layout",
            "rule": json.dumps({"canonical_class": "Title", "ro_index": 0}),
        },
        {
            "pdf": "docs/layout/checks.pdf",
            "type": "layout",
            "rule": json.dumps({"canonical_class": "Text", "ro_index": 1}),
        },
    ]
    (tmp_path / "table.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )
    summary = EvaluationSummary(
        total_examples=3,
        successful=3,
        failed=0,
        skipped=0,
        per_example_results=[
            EvaluationResult(
                test_id="table/sample",
                example_id="table/sample",
                pipeline_name="test",
                product_type="parse",
                success=True,
            ),
            EvaluationResult(
                test_id="table/fallback",
                example_id="table/fallback",
                pipeline_name="test",
                product_type="parse",
                success=True,
            ),
            EvaluationResult(
                test_id="layout/checks",
                example_id="layout/checks",
                pipeline_name="test",
                product_type="parse",
                success=True,
            ),
        ],
    )

    examples = _build_data_blob(summary, test_cases_dir=tmp_path)["examples"]

    assert examples[0]["expectedOutput"] == "sidecar expected"
    assert examples[1]["expectedOutput"] == "jsonl expected"
    assert examples[2]["expectedOutput"] == ""
    assert examples[2]["expectedChecks"] == [
        {"type": "layout", "canonical_class": "Title", "ro_index": 0},
        {"type": "layout", "canonical_class": "Text", "ro_index": 1},
    ]


def test_expected_output_uses_checks_from_requested_group(tmp_path):
    for group, rule_type in [
        ("text_content", "missing_specific_word"),
        ("text_formatting", "is_bold"),
    ]:
        (tmp_path / f"{group}.jsonl").write_text(
            json.dumps(
                {
                    "pdf": "docs/text/shared.pdf",
                    "type": rule_type,
                    "rule": json.dumps({"text": "expected"}),
                }
            ),
            encoding="utf-8",
        )

    summary = EvaluationSummary(
        total_examples=1,
        successful=1,
        failed=0,
        skipped=0,
        per_example_results=[
            EvaluationResult(
                test_id="text/shared",
                example_id="text/shared",
                pipeline_name="test",
                product_type="parse",
                success=True,
            )
        ],
    )

    examples = _build_data_blob(summary, test_cases_dir=tmp_path, group="text_formatting")["examples"]

    assert examples[0]["expectedChecks"] == [{"type": "is_bold", "text": "expected"}]


def test_expected_output_only_indexes_reported_documents(tmp_path):
    records = [
        {
            "pdf": "docs/text/included.pdf",
            "type": "contains_text",
            "rule": json.dumps({"text": "included"}),
        },
        {
            "pdf": "docs/text/excluded.pdf",
            "type": "contains_text",
            "rule": json.dumps({"text": "excluded"}),
        },
    ]
    (tmp_path / "text_content.jsonl").write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )
    summary = EvaluationSummary(
        total_examples=1,
        successful=1,
        failed=0,
        skipped=0,
        per_example_results=[
            EvaluationResult(
                test_id="text/included",
                example_id="text/included",
                pipeline_name="test",
                product_type="parse",
                success=True,
            )
        ],
    )

    examples = _build_data_blob(summary, test_cases_dir=tmp_path, group="text_content")["examples"]

    assert examples[0]["expectedChecks"] == [{"type": "contains_text", "text": "included"}]
    assert examples[0]["expectedHtml"] == ""


def test_expected_checks_preserve_loader_identity_fields(tmp_path):
    (tmp_path / "layout.jsonl").write_text(
        json.dumps(
            {
                "pdf": "docs/layout/sample.pdf",
                "id": "rule-1",
                "page": 2,
                "type": "layout",
                "rule": json.dumps({"canonical_class": "Title"}),
            }
        ),
        encoding="utf-8",
    )
    summary = EvaluationSummary(
        total_examples=1,
        successful=1,
        failed=0,
        skipped=0,
        per_example_results=[
            EvaluationResult(
                test_id="layout/sample",
                example_id="layout/sample",
                pipeline_name="test",
                product_type="parse",
                success=True,
                tags=["layout"],
            )
        ],
    )

    examples = _build_data_blob(summary, test_cases_dir=tmp_path)["examples"]

    assert examples[0]["expectedChecks"] == [
        {"type": "layout", "canonical_class": "Title", "id": "rule-1", "page": 2}
    ]


def test_expected_checks_do_not_mix_categories_with_shared_filenames(tmp_path):
    for category, rule_type in [("chart", "chart_data_point"), ("layout", "layout")]:
        (tmp_path / f"{category}.jsonl").write_text(
            json.dumps(
                {
                    "pdf": f"docs/{category}/shared.pdf",
                    "type": rule_type,
                    "rule": json.dumps({"value": category}),
                }
            ),
            encoding="utf-8",
        )
    summary = EvaluationSummary(
        total_examples=1,
        successful=1,
        failed=0,
        skipped=0,
        per_example_results=[
            EvaluationResult(
                test_id="layout/shared",
                example_id="layout/shared",
                pipeline_name="test",
                product_type="parse",
                success=True,
                tags=["layout"],
            )
        ],
    )

    examples = _build_data_blob(summary, test_cases_dir=tmp_path)["examples"]

    assert examples[0]["expectedChecks"] == [{"type": "layout", "value": "layout"}]


def test_invalid_jsonl_reports_file_and_line(tmp_path):
    (tmp_path / "layout.jsonl").write_text('{"pdf": "valid.pdf"}\nnot-json\n', encoding="utf-8")
    summary = EvaluationSummary(total_examples=0, successful=0, failed=0, skipped=0)

    try:
        _build_data_blob(summary, test_cases_dir=tmp_path, group="layout")
    except ValueError as exc:
        assert f"{tmp_path / 'layout.jsonl'}:2" in str(exc)
    else:
        raise AssertionError("Expected malformed JSONL to fail report generation")


def test_expected_markdown_pointer_uses_dataset_lookup(tmp_path):
    (tmp_path / "expected_markdown.json").write_text(
        json.dumps({"docs/table/sample.pdf": "# Expected table"}),
        encoding="utf-8",
    )
    (tmp_path / "table.jsonl").write_text(
        "\n"
        + json.dumps(
            {
                "pdf": "docs/table/sample.pdf",
                "type": "expected_markdown",
                "rule": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = EvaluationSummary(
        total_examples=1,
        successful=1,
        failed=0,
        skipped=0,
        per_example_results=[
            EvaluationResult(
                test_id="table/sample",
                example_id="table/sample",
                pipeline_name="test",
                product_type="parse",
                success=True,
                tags=["table"],
            )
        ],
    )

    examples = _build_data_blob(summary, test_cases_dir=tmp_path)["examples"]

    assert examples[0]["expectedOutput"] == "# Expected table"
    assert examples[0]["expectedChecks"] == []
