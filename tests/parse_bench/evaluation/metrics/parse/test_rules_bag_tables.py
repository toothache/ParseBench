"""Regression tests for table text serialization in TextContent rules."""

from parse_bench.evaluation.metrics.parse.rules_bag import (
    TooManySentenceOccurenceRule,
    TooManyWordOccurenceRule,
)
from parse_bench.evaluation.metrics.parse.rules_base import _augment_with_table_cell_text
from parse_bench.test_cases.parse_rule_schemas import (
    ParseTooManySentenceOccurrenceRule,
    ParseTooManyWordOccurrenceRule,
)


def test_html_table_cells_are_not_duplicated() -> None:
    markdown = "Before\n<table><tr><td>Alpha</td><td>Beta</td></tr></table>\nAfter"

    serialized = _augment_with_table_cell_text(markdown)

    assert serialized.split().count("Alpha") == 1
    assert serialized.split().count("Beta") == 1
    assert "Before" in serialized
    assert "After" in serialized


def test_markdown_table_cells_are_not_duplicated() -> None:
    markdown = "Before\n| Alpha | Beta |\n| --- | --- |\n| One | Two |\nAfter"

    serialized = _augment_with_table_cell_text(markdown)

    assert serialized.split().count("Alpha") == 1
    assert serialized.split().count("Beta") == 1
    assert serialized.split().count("One") == 1
    assert serialized.split().count("Two") == 1


def test_row_mode_supports_cross_cell_matching_without_duplicate_words() -> None:
    markdown = "<table><tr><td>Alpha</td><td>Beta</td></tr></table>"

    serialized = _augment_with_table_cell_text(markdown, group_rows=True)

    assert serialized == "Alpha Beta"
    assert serialized.split().count("Alpha") == 1
    assert serialized.split().count("Beta") == 1


def test_occurrence_rules_do_not_double_count_table_content() -> None:
    markdown = "<table><tr><td>Alpha</td><td>Beta</td></tr></table>"
    word_rule = TooManyWordOccurenceRule(
        ParseTooManyWordOccurrenceRule(
            type="too_many_word_occurence",
            bag_of_word={"alpha": 1},
        )
    )
    sentence_rule = TooManySentenceOccurenceRule(
        ParseTooManySentenceOccurrenceRule(
            type="too_many_sentence_occurence",
            bag_of_sentence={"Alpha Beta": 1},
        )
    )

    assert word_rule.run(markdown) == (True, "")
    assert sentence_rule.run(markdown) == (True, "")
