"""
Pipeline Core - VulnPipeline orchestration and composition

Provides:
- VulnPipeline: Main ETL orchestrator with composition support
- PipelineResult: Execution results with metadata and error tracking

Layer: Core ETL orchestration
Dependencies: dataset, extractors, transforms, loaders
Used by: User code, integration tests
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset


@dataclass
class PipelineResult:
    """Result of pipeline execution with metadata."""

    dataset: VulnDataset | None
    loader_outputs: dict[str, Any] = field(default_factory=dict)
    errors: list[Exception] = field(default_factory=list)
    successful_stages: list[str] = field(default_factory=list)
    failed_stages: list[str] = field(default_factory=list)
    duration: float = 0.0


@dataclass
class VulnPipeline:
    """ETL pipeline orchestrator with composition support."""

    extractors: list[Any] = field(default_factory=list)
    transformers: list[Any] = field(default_factory=list)
    loaders: list[Any] = field(default_factory=list)

    def run(
        self, dataset: VulnDataset | None = None, strict: bool = False
    ) -> PipelineResult:
        """Execute pipeline stages: extract → transform → load.

        Args:
            dataset: Optional pre-existing dataset (skips extraction)
            strict: If True, raise on transformer errors. If False, collect errors.

        Returns:
            PipelineResult with dataset, metadata, and errors
        """
        import time

        from vulnflow.dataset import VulnDataset

        start_time = time.time()
        errors: list[Exception] = []
        successful_stages: list[str] = []
        failed_stages: list[str] = []

        # Stage 1: Extract
        if dataset is None and self.extractors:
            records = []
            for extractor in self.extractors:
                try:
                    records.extend(extractor.extract())
                    successful_stages.append(extractor.__class__.__name__)
                except Exception as e:
                    failed_stages.append(extractor.__class__.__name__)
                    errors.append(e)
                    raise  # Extractor failures always raise

            dataset = VulnDataset._from_records(records)

        # Stage 2: Transform (placeholder for Phase 4+)
        # Stage 3: Load (placeholder for Phase 1+)

        duration = time.time() - start_time

        return PipelineResult(
            dataset=dataset,
            loader_outputs={},
            errors=errors,
            successful_stages=successful_stages,
            failed_stages=failed_stages,
            duration=duration,
        )

    def then(self, other: VulnPipeline) -> VulnPipeline:
        """Compose pipelines immutably.

        Args:
            other: Pipeline to append after this one

        Returns:
            New VulnPipeline with combined stages
        """
        return VulnPipeline(
            extractors=self.extractors + other.extractors,
            transformers=self.transformers + other.transformers,
            loaders=self.loaders + other.loaders,
        )

    @classmethod
    def chain(cls, pipelines: list[VulnPipeline]) -> VulnPipeline:
        """Combine multiple pipelines into one.

        Args:
            pipelines: List of pipelines to combine

        Returns:
            New VulnPipeline with all stages combined
        """
        extractors: list[Any] = []
        transformers: list[Any] = []
        loaders: list[Any] = []

        for pipeline in pipelines:
            extractors.extend(pipeline.extractors)
            transformers.extend(pipeline.transformers)
            loaders.extend(pipeline.loaders)

        return cls(extractors=extractors, transformers=transformers, loaders=loaders)
