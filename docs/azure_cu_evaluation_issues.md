# Azure Content Understanding evaluation issues

## Purpose

This note records the observed problems before deciding on an implementation.
The current investigation found two reproducible defects and one separate,
unconfirmed concern:

1. Azure Content Understanding (CU) page furniture is hidden from
   Content Faithfulness evaluation because CU emits it inside HTML comments.
2. ParseBench's table-to-text serialization duplicates table content during
   occurrence counting.
3. Azure CU sometimes predicts more tables than the ground truth, but the
   evidence does not show that extra table count alone causes lower table
   scores.

These issues should not be fixed as one broad change. The Azure CU
normalization issue is provider-specific. The table serialization issue is a
provider-neutral evaluator issue and should be reviewed independently.

## Issue 1: CU page furniture disappears from Content Faithfulness

### Observed Azure CU output

CU's `prebuilt-layout` analyzer can encode page headers, footers, and printed
page numbers as semantic HTML comments:

```markdown
<!-- PageHeader: 4/22/25, 7:39 PM -->
<!-- PageHeader: New FINRA Initiatives to Support Members, Markets, and the Investors They Serve | FINRA.org -->

First, we will offer additional actionable compliance resources...

<!-- PageFooter: https://www.finra.org/media-center/blog/new-finra-initiatives-support-members-markets-and-investors-they-serve -->
<!-- PageNumber: 3/4 -->
```

The page-furniture text is genuine OCR content. The corresponding ParseBench
rules expect, among other values:

```text
4/22/25, 7:39 PM
New FINRA Initiatives to Support Members, Markets, and the Investors They Serve | FINRA.org
https://www.finra.org/media-center/blog/new-finra-initiatives-support-members-markets-and-investors-they-serve
3/4
```

### Why it fails

Content Faithfulness evaluates `ParseOutput.markdown`. Its sentence and word
normalization removes HTML tags/comments before matching. Because the semantic
payload is inside the comment, the payload is removed with the markup.

Minimal reproduction from the ParseBench repository:

```bash
uv run --extra runners python - <<'PY'
from parse_bench.evaluation.metrics.parse.rules_bag import (
    SentenceBagRule,
    WordBagRule,
)

markdown = "<!-- PageHeader: 4/22/25, 7:39 PM -->\nBody text"
print(repr(SentenceBagRule._normalize_full_text(markdown)))
print(repr(WordBagRule._normalize_full_word_text(markdown)))
PY
```

At the baseline revision `f44d8a2`, the output is:

```text
' body text'
'  body text'
```

The date and time are absent, which produces missing sentence, word, and digit
failures.

### Real-case reproduction

With Azure CU credentials configured:

```bash
uv run parse-bench run azure_cu_layout --group text_content
find output -name 'text_simple__finra.result.json'
```

Inspect the normalized Markdown:

```bash
jq -r '.output.markdown' <path-to-text_simple__finra.result.json> | \
  grep -n -E '<!-- (PageHeader|PageFooter|PageNumber):'
```

Inspect the corresponding evaluation report:

```bash
jq '
  .per_example_results[]
  | select(.example_id == "text/text_simple__finra")
  | .metrics[0].metadata.rule_results[]
  | select(
      .passed == false
      and (
        .type == "missing_specific_sentence"
        or .type == "missing_word_percent"
        or .type == "bag_of_digit_percent"
      )
    )
' <path-to-text_content/_evaluation_report.json>
```

In the investigated run, the hidden header/footer/page-number content caused
missing specific sentence, missing word, and digit failures.

## Issue 2: CU layout roles exist, but dedicated page sections are empty

Azure CU also returns positioned paragraphs with semantic roles:

```text
pageHeader
pageFooter
pageNumber
```

`_build_layout_pages()` maps them to layout item labels such as `Page-header`
and `Page-footer`, so the geometric role information is present. However, at
baseline `f44d8a2`, the generated `ParseLayoutPageIR` does not populate:

```text
page_header_markdown
page_footer_markdown
printed_page_number
```

Header/footer rules prefer these dedicated fields when `layout_pages` exists.
Therefore, merely making the document Markdown visible is not enough to
preserve the structured evaluation contract.

Minimal reproduction:

```bash
uv run --extra runners python - <<'PY'
from azure.ai.contentunderstanding.models import AnalysisResult
from parse_bench.inference.providers.parse.azure_content_understanding import (
    _build_layout_pages,
)

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
print([item.bbox.label for item in page.items])
print(repr(page.page_header_markdown))
print(repr(page.page_footer_markdown))
print(repr(page.printed_page_number))
PY
```

At baseline, the item labels are present but all three dedicated strings are
empty.

## Issue 3: table text is duplicated during TextContent serialization

This issue is not specific to Azure CU.

For missing-content and occurrence rules, ParseBench extracts every table cell,
also creates a concatenated row string, and appends both to the original
Markdown. The original table remains in the input. After HTML tags are removed,
the same words can be counted from:

1. the original table;
2. the individual extracted cell;
3. the concatenated row.

Minimal reproduction at baseline `f44d8a2`:

```bash
uv run python - <<'PY'
from parse_bench.evaluation.metrics.parse.rules_base import (
    _augment_with_table_cell_text,
)
from parse_bench.evaluation.metrics.parse.rules_bag import WordBagRule

markdown = "<table><tr><td>Alpha</td><td>Beta</td></tr></table>"
print(_augment_with_table_cell_text(markdown))
print(WordBagRule._normalize_full_word_text(markdown).split())
PY
```

The augmented representation contains the original table plus:

```text
Alpha
Beta
Alpha Beta
```

After normalization, `Alpha` and `Beta` can each occur three times even though
each appears once in the source table. This can:

- create false `too_many_*_occurrence` failures;
- allow `missing_*` rules to pass with an inflated occurrence count;
- make table-heavy parser output interact differently with TextContent rules.

This should be fixed in a separate evaluator change with provider-neutral HTML
and Markdown table tests.

## Extra predicted tables: observed, but not yet a confirmed metric defect

The investigated Azure CU run contained cases where
`tables_found_actual > tables_found_expected`. That observation alone does not
explain a lower score:

- GriTS pairs predicted and expected tables using Hungarian matching and adds
  zeroes for unmatched expected tables.
- Some observed cases had extra predicted tables and still received a GriTS
  content score of `1.0`.

To inspect this in any completed run:

```bash
jq '
  .per_example_results[]
  | . as $result
  | (
      .metrics[]
      | select(.metric_name == "grits_con")
    ) as $metric
  | select(
      $metric.metadata.tables_found_actual
      > $metric.metadata.tables_found_expected
    )
  | {
      example_id: $result.example_id,
      score: $metric.value,
      expected: $metric.metadata.tables_found_expected,
      actual: $metric.metadata.tables_found_actual
    }
' <path-to-table/_evaluation_report.json>
```

Before changing table matching or penalties, a specific case must demonstrate
that the metric pairs or scores the tables incorrectly. Chart-to-table
conversion is another separate path and should not be changed without such a
fixture.

## Recommended scope split

### Provider-specific Azure CU change

The narrow Azure CU change should:

1. expose only recognized `PageHeader`, `PageFooter`, and `PageNumber` comment
   payloads in `ParseOutput.markdown`;
2. leave unrelated HTML comments unchanged;
3. populate the dedicated structured page fields from CU paragraph roles;
4. preserve layout item labels and geometry;
5. include focused Azure CU tests.

It should not change global Markdown serialization, table metrics, or other
providers.

### Provider-neutral evaluator change

The table serialization change should:

1. replace table markup with one non-duplicated textual representation;
2. support cell-level tokenization and row-level substring matching without
   counting both representations simultaneously;
3. cover HTML and Markdown tables;
4. verify missing and too-many occurrence rules;
5. be reviewed independently of Azure CU normalization.

### Not yet justified

The current evidence does not justify:

- globally exposing all HTML comments for every provider;
- changing GriTS or TableRecordMatch penalties for extra predicted tables;
- changing Azure CU chart-to-table conversion;
- changing the benchmark ground truth.

## Evidence collected

For the FINRA example, replaying the cached Azure CU response after exposing
page furniture and populating structured page fields increased the full
Content Faithfulness rule pass rate from `97.94%` to `99.27%`. Header, footer,
and digit failures cleared. Remaining failures were unrelated sentence
normalization differences, including URL and section-number boundaries.

That result supports the provider-specific page-furniture fix. It does not, by
itself, validate a broader table metric change.
