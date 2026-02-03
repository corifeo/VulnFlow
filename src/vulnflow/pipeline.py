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

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset
    from vulnflow.extractors.base import Extractor
    from vulnflow.loaders.base import Loader
    from vulnflow.transforms.base import BaseTransform

logger = logging.getLogger(__name__)


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

    if TYPE_CHECKING:
        extractors: list[Extractor] = field(default_factory=list)
        transformers: list[BaseTransform] = field(default_factory=list)
        loaders: list[Loader] = field(default_factory=list)
    else:
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


        start_time = time.time()
        errors: list[Exception] = []
        successful_stages: list[str] = []
        failed_stages: list[str] = []

        # Stage 1: Extract
        if dataset is None and self.extractors:
            datasets = []
            for extractor in self.extractors:
                try:
                    ds = extractor.extract()  # Returns VulnDataset
                    datasets.append(ds)
                    successful_stages.append(extractor.__class__.__name__)
                except Exception as e:
                    failed_stages.append(extractor.__class__.__name__)
                    errors.append(e)
                    raise  # Extractor failures always raise

            # Merge datasets if multiple extractors
            from vulnflow.dataset import VulnDataset

            dataset = VulnDataset.merge(datasets)

        # Stage 2: Transform (enrichers + transforms)
        if self.transformers and dataset is not None:
            import asyncio

            from vulnflow.utils.transforms import partition_transforms

            # Separate async transforms (enrichers) from sync transforms
            async_transforms, sync_transforms = partition_transforms(self.transformers)

            # Run async transforms concurrently with asyncio.gather()
            if async_transforms:

                async def run_async_concurrent() -> list[BaseException | None]:
                    if strict:
                        # In strict mode, any failure raises immediately
                        await asyncio.gather(
                            *[t.transform(dataset) for t in async_transforms]
                        )
                        return []
                    else:
                        # In lenient mode, collect exceptions
                        results = await asyncio.gather(
                            *[t.transform(dataset) for t in async_transforms],
                            return_exceptions=True,
                        )
                        return results

                try:
                    results = asyncio.run(run_async_concurrent())
                    if strict:
                        # All succeeded if we got here
                        for t in async_transforms:
                            successful_stages.append(t.__class__.__name__)
                    else:
                        # Check for exceptions in results
                        for t, result in zip(async_transforms, results, strict=True):
                            t_name = t.__class__.__name__
                            if isinstance(result, Exception):
                                logger.warning(
                                    "Transform %s failed: %s",
                                    getattr(t, "name", t_name),
                                    result,
                                )
                                errors.append(result)
                                failed_stages.append(t_name)
                            else:
                                successful_stages.append(t_name)
                except Exception as e:
                    for t in async_transforms:
                        failed_stages.append(t.__class__.__name__)
                    errors.append(e)
                    raise

            # Run sync transforms sequentially (may have dependencies)
            for transformer in sync_transforms:
                transformer_name = transformer.__class__.__name__
                try:
                    transformer.transform(dataset)
                    successful_stages.append(transformer_name)
                except Exception as e:
                    logger.warning(
                        "Transform %s failed: %s",
                        getattr(transformer, "name", transformer_name),
                        e,
                    )
                    failed_stages.append(transformer_name)
                    errors.append(e)
                    if strict:
                        raise

        # Stage 3: Load
        loader_outputs: dict[str, Any] = {}
        if self.loaders and dataset is not None:
            for loader in self.loaders:
                try:
                    output = loader.load(dataset)
                    loader_name = loader.__class__.__name__
                    loader_outputs[loader_name] = output
                    successful_stages.append(loader_name)
                except Exception as e:
                    failed_stages.append(loader.__class__.__name__)
                    errors.append(e)
                    raise  # Loader failures always raise

        duration = time.time() - start_time

        return PipelineResult(
            dataset=dataset,
            loader_outputs=loader_outputs,
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
