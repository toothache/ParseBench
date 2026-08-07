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
    assert examples[2]["expectedOutput"] == (
        "Evaluation checks (2 checks)\n\n"
        "```json\n"
        "[\n"
        "  {\n"
        '    "type": "layout",\n'
        '    "rule": {\n'
        '      "canonical_class": "Title",\n'
        '      "ro_index": 0\n'
        "    }\n"
        "  },\n"
        "  {\n"
        '    "type": "layout",\n'
        '    "rule": {\n'
        '      "canonical_class": "Text",\n'
        '      "ro_index": 1\n'
        "    }\n"
        "  }\n"
        "]\n"
        "```"
    )


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

    assert '"type": "is_bold"' in examples[0]["expectedOutput"]
    assert "missing_specific_word" not in examples[0]["expectedOutput"]
