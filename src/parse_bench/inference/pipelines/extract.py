"""Extract pipelines - structured data extraction from documents."""

from typing import Any

from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.product import ProductType

GRANULAR_BBOX_OUTPUT_OPTIONS = {
    "granular_bboxes": ["word"],
}
LLAMAEXTRACT_V2_AGENTIC_HOSTED_GRANULAR_PARSE_CONFIG = {
    "tier": "agentic",
    "version": "latest",
    "disable_cache": True,
    "output_options": GRANULAR_BBOX_OUTPUT_OPTIONS,
}


def _pipeline_spec(
    *,
    pipeline_name: str,
    provider_name: str,
    config: dict[str, Any],
) -> PipelineSpec:
    return PipelineSpec(
        pipeline_name=pipeline_name,
        provider_name=provider_name,
        product_type=ProductType.EXTRACT,
        config=config,
    )


def register_extract_pipelines(register_fn) -> None:  # type: ignore[no-untyped-def]
    """Register the implementation-target extract pipelines."""

    register_fn(
        _pipeline_spec(
            pipeline_name="llamaextract_v2_cost_effective_parse_agentic_granular_bboxes_staging",
            provider_name="llamaextract_v2",
            config={
                "tier": "cost_effective",
                "parse_tier": "agentic",
                "use_staging": True,
                "timeout": 3000,
                "cite_sources": True,
                "parse_config": LLAMAEXTRACT_V2_AGENTIC_HOSTED_GRANULAR_PARSE_CONFIG,
            },
        )
    )

    register_fn(
        _pipeline_spec(
            pipeline_name="extend_extract",
            provider_name="extend",
            config={
                "baseProcessor": "extraction_performance",
                "baseVersion": "4.1.1",
                "advancedOptions": {
                    "citationsEnabled": True,
                    "advancedFigureParsingEnabled": True,
                },
            },
        )
    )
