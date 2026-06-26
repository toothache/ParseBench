"""Provider for Azure AI Content Understanding (PARSE).

Azure Content Understanding is a generative multimodal analysis service (distinct
from the older Azure Document Intelligence). It is exposed as an async REST API:
submit a document to an analyzer, poll an operation URL, then read the result.

This provider:
  * Submits the document bytes as base64 in a JSON envelope, polls the returned
    operation URL until terminal status, and returns the result JSON.
  * Normalizes the result's markdown + paragraph/table/figure polygons into
    ``ParseOutput`` (markdown for text/table/chart eval, ``layout_pages`` for
    layout eval).
  * Optionally renders Chart.js figure data (returned by figure-capable
    analyzers) into markdown tables so the chart metric can score data points.

Three analyzer modes, selected by ``analyzer_id`` + ``enable_figure_analysis``
(see the registered pipelines in ``inference/pipelines/parse.py``):
  * ``prebuilt-layout`` (provider default) — OCR + layout + tables, no LLM
    dependency. Charts are located but not transcribed (chart score ~1%).
  * ``prebuilt-documentSearch`` with ``enable_figure_analysis=True`` — adds
    generative figure analysis; charts come back as Chart.js configs that this
    provider renders to tables (chart score ~45%). Requires the resource's
    backing GPT model deployments.
  * A derived custom analyzer (``create_custom_analyzer=True``, based on
    ``prebuilt-document`` with ``enableFigureAnalysis``) — created/reused on the
    resource. Off by default; the documentSearch path is preferred.

Auth/config:
  * ``AZURE_CONTENT_UNDERSTANDING_KEY`` / ``AZURE_CONTENT_UNDERSTANDING_ENDPOINT``
    env vars (or ``api_key`` / ``endpoint`` in pipeline config).
  * ``api_version`` (default ``2025-11-01``), ``analyzer_id`` (default
    ``prebuilt-layout``), ``enable_figure_analysis`` (default False),
    ``create_custom_analyzer`` (default False), ``table_format`` (default
    ``html``), ``chart_format`` (default ``chartjs``), plus ``poll_interval_seconds``,
    ``timeout``, and ``http_timeout_seconds`` knobs.
"""

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from parse_bench.inference.providers.base import (
    Provider,
    ProviderConfigError,
    ProviderPermanentError,
    ProviderTransientError,
)
from parse_bench.inference.providers.registry import register_provider
from parse_bench.schemas.parse_output import (
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.pipeline_io import (
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType

# Content Understanding paragraph role -> Canonical17 label string.
# Mirrors the Azure DI mapping so layout cross-eval is consistent across the two
# Azure providers.
CU_LABEL_MAP: dict[str, str] = {
    "title": "Title",
    "sectionHeading": "Section-header",
    "pageHeader": "Page-header",
    "pageFooter": "Page-footer",
    "footnote": "Footnote",
    "pageNumber": "Page-footer",
}

_DEFAULT_PARAGRAPH_LABEL = "Text"

# Virtual page dimensions for normalized coordinate conversion. CU polygons are
# normalized to [0,1] via the page's own width/height (in inches), so the actual
# virtual dimension cancels out — kept for parity with the Azure DI provider.
_VIRTUAL_PAGE_DIM = 1000.0

_TRANSIENT_KEYWORDS = (
    "timeout",
    "timed out",
    "network",
    "connection",
    "temporarily",
    "throttl",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
)


@register_provider("azure_content_understanding")
class AzureContentUnderstandingProvider(Provider):
    """Provider for Azure AI Content Understanding PARSE (REST)."""

    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

        self._api_key = self.base_config.get("api_key") or os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY")
        self._endpoint = (
            self.base_config.get("endpoint") or os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT") or ""
        ).rstrip("/")

        if not self._api_key:
            raise ProviderConfigError(
                "Azure Content Understanding API key is required. "
                "Set AZURE_CONTENT_UNDERSTANDING_KEY environment variable "
                "or pass api_key in base_config."
            )
        if not self._endpoint:
            raise ProviderConfigError(
                "Azure Content Understanding endpoint is required. "
                "Set AZURE_CONTENT_UNDERSTANDING_ENDPOINT environment variable "
                "or pass endpoint in base_config."
            )

        self._api_version = self.base_config.get("api_version", "2025-11-01")
        self._base_analyzer_id = self.base_config.get("analyzer_id", "prebuilt-layout")
        self._enable_figure_analysis = bool(self.base_config.get("enable_figure_analysis", False))
        self._table_format = self.base_config.get("table_format", "html")
        self._chart_format = self.base_config.get("chart_format", "chartjs")
        self._poll_interval_s = float(self.base_config.get("poll_interval_seconds", 2.0))
        self._timeout_s = float(self.base_config.get("timeout", 300))
        self._http_timeout_s = float(self.base_config.get("http_timeout_seconds", 120))

        # Two ways to get figure/chart analysis:
        #   1. Point analyzer_id at a figure-capable prebuilt (e.g.
        #      "prebuilt-documentSearch") and set create_custom_analyzer=False —
        #      used directly, no resource mutation. (Requires the resource's
        #      backing GPT model deployments.)
        #   2. Derive a custom analyzer from a base (e.g. "prebuilt-document")
        #      with enableFigureAnalysis — set create_custom_analyzer=True.
        # When enable_figure_analysis is False (e.g. prebuilt-layout / -read),
        # neither applies and the analyzer is used as-is with no LLM dependency.
        self._create_custom_analyzer = bool(
            self.base_config.get("create_custom_analyzer", False) and self._enable_figure_analysis
        )
        if self._create_custom_analyzer:
            # Custom analyzer ids cannot contain "-" or start with the reserved
            # "prebuilt" prefix, so build a safe ParseBench-namespaced id.
            safe_base = re.sub(r"[^A-Za-z0-9_.]", "_", self._base_analyzer_id)
            self._analyzer_id = self.base_config.get("custom_analyzer_id", f"parsebench_{safe_base}_figanalysis")
        else:
            self._analyzer_id = self._base_analyzer_id

        self._headers = {"Ocp-Apim-Subscription-Key": self._api_key}
        self._analyzer_ready = False

    # ------------------------------------------------------------------ helpers

    def _client(self):  # type: ignore[no-untyped-def]
        import httpx

        return httpx.Client(timeout=self._http_timeout_s)

    def _classify_and_raise(self, message: str, status_code: int | None = None, cause: Exception | None = None) -> None:
        """Raise the right ProviderError subtype based on message/status.

        ``cause``, when given, is chained via ``raise ... from cause`` so the
        original traceback is preserved for debugging.
        """
        lowered = message.lower()
        if status_code in (408, 429, 500, 502, 503, 504) or any(k in lowered for k in _TRANSIENT_KEYWORDS):
            raise ProviderTransientError(message) from cause
        raise ProviderPermanentError(message) from cause

    def _ensure_analyzer(self) -> None:
        """Create the derived figure-analysis analyzer if it doesn't exist yet.

        For the plain prebuilt analyzer (figure analysis disabled) this is a
        no-op. The custom analyzer is based on ``prebuilt-document`` with
        ``enableFigureAnalysis`` so figures/charts are described in the markdown.
        """
        if self._analyzer_ready or not self._create_custom_analyzer:
            self._analyzer_ready = True
            return

        import httpx

        base = f"{self._endpoint}/contentunderstanding/analyzers/{self._analyzer_id}"
        params = {"api-version": self._api_version}
        try:
            with self._client() as client:
                # Already exists and ready?
                r = client.get(base, headers=self._headers, params=params)
                if r.status_code == 200 and r.json().get("status") == "ready":
                    self._analyzer_ready = True
                    return

                analyzer_def = {
                    "baseAnalyzerId": self._base_analyzer_id,
                    "description": "ParseBench: prebuilt-document + figure analysis for charts.",
                    "config": {
                        "enableOcr": True,
                        "enableLayout": True,
                        "enableFormula": True,
                        "enableFigureAnalysis": True,
                        "enableFigureDescription": True,
                        "chartFormat": self._chart_format,
                        "tableFormat": self._table_format,
                        "annotationFormat": "markdown",
                        "returnDetails": True,
                    },
                }
                pr = client.put(
                    base,
                    headers={**self._headers, "Content-Type": "application/json"},
                    params=params,
                    json=analyzer_def,
                )
                if pr.status_code not in (200, 201, 202):
                    self._classify_and_raise(
                        f"Failed to create Content Understanding analyzer "
                        f"'{self._analyzer_id}': {pr.status_code} {pr.text[:300]}",
                        pr.status_code,
                    )

                # Poll until the analyzer is ready (creation may be async).
                deadline = time.monotonic() + self._timeout_s
                while time.monotonic() < deadline:
                    g = client.get(base, headers=self._headers, params=params)
                    status = g.json().get("status") if g.status_code == 200 else None
                    if status == "ready":
                        self._analyzer_ready = True
                        return
                    if status == "failed":
                        raise ProviderPermanentError(
                            f"Content Understanding analyzer '{self._analyzer_id}' creation failed: {g.text[:300]}"
                        )
                    time.sleep(self._poll_interval_s)
                raise ProviderTransientError(f"Timed out waiting for analyzer '{self._analyzer_id}' to become ready.")
        except httpx.HTTPError as e:
            self._classify_and_raise(f"HTTP error creating analyzer: {e}", cause=e)

    def _analyze(self, pdf_bytes: bytes) -> dict[str, Any]:
        """Submit a document, poll the operation, and return the result JSON."""
        import base64

        import httpx

        self._ensure_analyzer()
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        analyze_url = f"{self._endpoint}/contentunderstanding/analyzers/{self._analyzer_id}:analyze"
        params = {"api-version": self._api_version}

        try:
            with self._client() as client:
                r = client.post(
                    analyze_url,
                    headers={**self._headers, "Content-Type": "application/json"},
                    params=params,
                    json={"inputs": [{"data": b64}]},
                )
                if r.status_code not in (200, 202):
                    self._classify_and_raise(
                        f"Content Understanding analyze failed: {r.status_code} {r.text[:300]}",
                        r.status_code,
                    )

                # 200 may carry an inline result; 202 returns an Operation-Location.
                op_location = r.headers.get("Operation-Location") or r.headers.get("operation-location")
                if r.status_code == 200 and not op_location:
                    return r.json()
                if not op_location:
                    raise ProviderTransientError("No Operation-Location returned by analyze call.")

                deadline = time.monotonic() + self._timeout_s
                while time.monotonic() < deadline:
                    g = client.get(op_location, headers=self._headers)
                    if g.status_code not in (200, 202):
                        self._classify_and_raise(
                            f"Polling Content Understanding result failed: {g.status_code} {g.text[:300]}",
                            g.status_code,
                        )
                    payload = g.json()
                    status = str(payload.get("status", "")).lower()
                    if status in ("succeeded", "completed"):
                        return payload
                    if status == "failed":
                        raise ProviderPermanentError(f"Content Understanding analysis failed: {str(payload)[:300]}")
                    time.sleep(self._poll_interval_s)
                raise ProviderTransientError("Timed out waiting for Content Understanding analysis result.")
        except httpx.HTTPError as e:
            self._classify_and_raise(f"HTTP error during analyze: {e}", cause=e)
        raise ProviderPermanentError("Unreachable: analyze returned without result.")

    # ------------------------------------------------------------------ provider API

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"AzureContentUnderstandingProvider only supports PARSE product type, got {request.product_type}"
            )

        started_at = datetime.now()
        pdf_path = Path(request.source_file_path)
        if not pdf_path.exists():
            raise ProviderPermanentError(f"PDF file not found: {pdf_path}")

        try:
            pdf_bytes = pdf_path.read_bytes()
            raw_output = self._analyze(pdf_bytes)
            raw_output["_config"] = {
                "analyzer_id": self._analyzer_id,
                "base_analyzer_id": self._base_analyzer_id,
                "api_version": self._api_version,
                "enable_figure_analysis": self._enable_figure_analysis,
            }
        except (ProviderPermanentError, ProviderTransientError, ProviderConfigError):
            raise
        except Exception as e:  # noqa: BLE001 - classify unexpected errors
            self._classify_and_raise(f"Unexpected error during inference: {e}", cause=e)

        completed_at = datetime.now()
        latency_ms = int((completed_at - started_at).total_seconds() * 1000)
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output=raw_output,
            started_at=started_at,
            completed_at=completed_at,
            latency_in_ms=latency_ms,
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        if raw_result.product_type != ProductType.PARSE:
            raise ProviderPermanentError(
                f"AzureContentUnderstandingProvider only supports PARSE product type, got {raw_result.product_type}"
            )

        contents = _get_contents(raw_result.raw_output)

        # Full-document markdown = concatenation of per-content markdown blocks.
        markdown = "\n\n".join(c.get("markdown", "") for c in contents if c.get("markdown")).strip()

        # Content Understanding figure analysis (prebuilt-documentSearch) returns
        # structured chart data in figures[].content as a Chart.js config, but the
        # inline markdown only carries flat image alt-text. The ParseBench chart
        # metric scores data points that appear as TABLES in the markdown, so we
        # render each Chart.js figure as a markdown table and append it. This is a
        # faithful transcription of the data the service already extracted — no
        # values are invented.
        chart_tables = _render_chart_tables(contents)
        if chart_tables:
            markdown = (markdown + "\n\n" + chart_tables).strip()

        # Build coarse per-page IR (markdown is document-level; page_index from content range).
        pages: list[PageIR] = []
        for c in contents:
            start = int(c.get("startPageNumber", 1) or 1)
            pages.append(PageIR(page_index=max(start - 1, 0), markdown=c.get("markdown", "")))

        layout_pages = _build_layout_pages(contents)

        output = ParseOutput(
            task_type="parse",
            example_id=raw_result.request.example_id,
            pipeline_name=raw_result.pipeline_name,
            pages=pages,
            layout_pages=layout_pages,
            markdown=markdown,
        )

        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=raw_result.raw_output,
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _get_contents(raw_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the ``contents`` list from a CU result, tolerant of nesting."""
    result = raw_output.get("result", raw_output)
    contents = result.get("contents")
    if isinstance(contents, list):
        return contents
    return []


def _chartjs_to_markdown_table(content: dict[str, Any], caption: str = "") -> str:
    """Render a Chart.js config (figures[].content) as a markdown table.

    Layout: one row per category label, one column per dataset series. This puts
    every (category, series) -> value association into a markdown table so the
    ParseBench chart metric can match data points. The figure caption is emitted
    as a heading above the table so the metric's context-aware label matching can
    pick up title labels (e.g. a year) that are not series names.
    """
    data = content.get("data")
    if not isinstance(data, dict):
        return ""
    labels = data.get("labels")
    datasets = data.get("datasets")
    if not isinstance(labels, list) or not isinstance(datasets, list) or not datasets:
        return ""

    series_names: list[str] = []
    series_values: list[list[Any]] = []
    for i, ds in enumerate(datasets):
        if not isinstance(ds, dict):
            continue
        name = str(ds.get("label") or f"Series {i + 1}").strip() or f"Series {i + 1}"
        vals = ds.get("data")
        if not isinstance(vals, list):
            continue
        series_names.append(name)
        series_values.append(vals)
    if not series_names:
        return ""

    def _esc(s: str) -> str:
        # A literal "|" or newline in a cell breaks markdown table structure
        # (extra columns / split rows), which would corrupt downstream parsing.
        return s.replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()

    def _cell(v: Any) -> str:
        if v is None:
            return ""
        # Chart.js scatter/bubble points are dicts; flatten to "x, y".
        if isinstance(v, dict):
            xy = [str(v[k]) for k in ("x", "y") if k in v]
            return _esc(", ".join(xy) if xy else str(v))
        return _esc(str(v))

    lines: list[str] = []
    if caption:
        lines.append(f"### {_esc(str(caption))}")
        lines.append("")
    header = ["Category", *[_esc(n) for n in series_names]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r, label in enumerate(labels):
        row = [_esc(str(label))]
        for vals in series_values:
            row.append(_cell(vals[r]) if r < len(vals) else "")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_chart_tables(contents: list[dict[str, Any]]) -> str:
    """Render all Chart.js figures across contents as appended markdown tables."""
    blocks: list[str] = []
    for content in contents:
        for fig in content.get("figures", []) or []:
            if fig.get("kind") != "chart":
                continue
            chart = fig.get("content")
            if not isinstance(chart, dict):
                continue
            caption = ""
            cap = fig.get("caption")
            if isinstance(cap, dict):
                caption = cap.get("content", "") or ""
            elif isinstance(cap, str):
                caption = cap
            table = _chartjs_to_markdown_table(chart, caption=caption)
            if table:
                blocks.append(table)
    return "\n\n".join(blocks)


def _parse_source_polygon(source: str) -> tuple[int, list[float]] | None:
    """Parse a CU source string ``D(page, x1,y1, x2,y2, x3,y3, x4,y4)``.

    Returns (page_number, [x1,y1,...,x4,y4]) in page units (inches), or None.
    """
    if not source:
        return None
    m = re.match(r"\s*[A-Za-z]\(([^)]*)\)", source)
    if not m:
        return None
    try:
        nums = [float(x.strip()) for x in m.group(1).split(",") if x.strip() != ""]
    except ValueError:
        return None
    if len(nums) < 9:
        return None
    page = int(nums[0])
    return page, nums[1:9]


def _polygon_to_normalized_bbox(
    polygon: list[float],
    page_width: float,
    page_height: float,
) -> tuple[float, float, float, float]:
    """Convert an 8-float corner polygon (page units) to normalized [0,1] xywh."""
    xs = [polygon[i] for i in range(0, len(polygon), 2)]
    ys = [polygon[i] for i in range(1, len(polygon), 2)]
    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    nx = x_min / page_width if page_width > 0 else 0.0
    ny = y_min / page_height if page_height > 0 else 0.0
    nw = (x_max - x_min) / page_width if page_width > 0 else 0.0
    nh = (y_max - y_min) / page_height if page_height > 0 else 0.0
    return (nx, ny, nw, nh)


def _build_layout_pages(contents: list[dict[str, Any]]) -> list[ParseLayoutPageIR]:
    """Build layout_pages from CU paragraphs/tables/figures for layout cross-eval.

    Groups elements by page using the source-polygon page index and converts CU
    polygon coordinates (page units) into normalized [0,1] LayoutSegmentIR entries.
    """
    from collections import defaultdict

    # page dimensions (inches) keyed by page number.
    page_dims: dict[int, tuple[float, float]] = {}
    for content in contents:
        for page in content.get("pages", []) or []:
            pnum = int(page.get("pageNumber", 1) or 1)
            width = float(page.get("width", 1.0) or 1.0)
            height = float(page.get("height", 1.0) or 1.0)
            page_dims[pnum] = (width, height)

    # (label, nx, ny, nw, nh, content, confidence) grouped by page.
    pages_items: dict[int, list[tuple[str, float, float, float, float, str, float]]] = defaultdict(list)

    def _add(label: str, source: str, text: str) -> None:
        parsed = _parse_source_polygon(source)
        if not parsed:
            return
        page_num, polygon = parsed
        pw, ph = page_dims.get(page_num, (1.0, 1.0))
        nx, ny, nw, nh = _polygon_to_normalized_bbox(polygon, pw, ph)
        pages_items[page_num].append((label, nx, ny, nw, nh, text, 1.0))

    for content in contents:
        # Paragraphs -> text / heading / header / footer elements.
        for para in content.get("paragraphs", []) or []:
            role = para.get("role")
            label = CU_LABEL_MAP.get(role, _DEFAULT_PARAGRAPH_LABEL) if role else _DEFAULT_PARAGRAPH_LABEL
            _add(label, para.get("source", ""), para.get("content", ""))

        # Tables -> Table elements (CU table objects carry their own source).
        for table in content.get("tables", []) or []:
            cells = table.get("cells", []) or []
            text = " ".join(c.get("content", "") for c in cells if c.get("content"))
            source = table.get("source", "")
            if not source:
                regions = table.get("boundingRegions") or []
                source = regions[0].get("source", "") if regions else ""
            _add("Table", source, text)

        # Figures (charts/images) -> Picture elements.
        for fig in content.get("figures", []) or []:
            caption = ""
            cap = fig.get("caption")
            if isinstance(cap, dict):
                caption = cap.get("content", "")
            elif isinstance(cap, str):
                caption = cap
            _add("Picture", fig.get("source", ""), caption)

    layout_pages: list[ParseLayoutPageIR] = []
    for page_num in sorted(pages_items.keys()):
        items: list[LayoutItemIR] = []
        for label, nx, ny, nw, nh, text, confidence in pages_items[page_num]:
            seg = LayoutSegmentIR(x=nx, y=ny, w=nw, h=nh, confidence=confidence, label=label)
            norm = label.strip().lower()
            if norm == "table":
                item_type = "table"
            elif norm == "picture":
                item_type = "image"
            else:
                item_type = "text"
            items.append(LayoutItemIR(type=item_type, value=text, bbox=seg, layout_segments=[seg]))
        layout_pages.append(
            ParseLayoutPageIR(
                page_number=page_num,
                width=_VIRTUAL_PAGE_DIM,
                height=_VIRTUAL_PAGE_DIM,
                items=items,
            )
        )

    return layout_pages
