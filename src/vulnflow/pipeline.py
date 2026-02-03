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
from typing import TYPE_CHECKING, Any, cast

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

    def _get_stage_name(self, obj: Any) -> str:
        """Get stage name from object (prefer name attribute, fallback to class name)."""
        return cast(str, getattr(obj, "name", obj.__class__.__name__))

    async def _run_async_transforms(
        self,
        dataset: VulnDataset,
        transforms: list[Any],
        strict: bool,
    ) -> tuple[list[str], list[str], list[Exception]]:
        """Execute async transforms concurrently with error handling.

        Args:
            dataset: Dataset to transform
            transforms: Async transform instances
            strict: If True, raise on errors. If False, collect errors.

        Returns:
            Tuple of (successful_stages, failed_stages, errors)
        """
        import asyncio

        if not transforms:
            return [], [], []

        successful: list[str] = []
        failed: list[str] = []
        errors: list[Exception] = []

        async def run_all() -> list[BaseException | None]:
            """Run all transforms concurrently with error handling."""
            if strict:
                await asyncio.gather(*[t.transform(dataset) for t in transforms])
                return []
            else:
                return cast(
                    list[BaseException | None],
                    await asyncio.gather(
                        *[t.transform(dataset) for t in transforms],
                        return_exceptions=True,
                    ),
                )

        try:
            results = await run_all()

            if strict:
                # All succeeded
                successful = [self._get_stage_name(t) for t in transforms]
            else:
                # Check each result
                for t, result in zip(transforms, results, strict=True):
                    t_name = self._get_stage_name(t)
                    if isinstance(result, Exception):
                        logger.warning("Transform %s failed: %s", t_name, result)
                        errors.append(result)
                        failed.append(t_name)
                    else:
                        successful.append(t_name)
        except Exception as e:
            failed = [self._get_stage_name(t) for t in transforms]
            errors.append(e)
            raise

        return successful, failed, errors

    def _run_sync_transforms(
        self,
        dataset: VulnDataset,
        transforms: list[Any],
        strict: bool,
    ) -> tuple[list[str], list[str], list[Exception]]:
        """Execute sync transforms sequentially with error handling.

        Args:
            dataset: Dataset to transform
            transforms: Sync transform instances
            strict: If True, raise on errors. If False, collect errors.

        Returns:
            Tuple of (successful_stages, failed_stages, errors)
        """
        successful: list[str] = []
        failed: list[str] = []
        errors: list[Exception] = []

        for t in transforms:
            t_name = self._get_stage_name(t)
            try:
                t.transform(dataset)
                successful.append(t_name)
            except Exception as e:
                logger.warning("Transform %s failed: %s", t_name, e)
                failed.append(t_name)
                errors.append(e)
                if strict:
                    raise

        return successful, failed, errors

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
                    successful_stages.append(self._get_stage_name(extractor))
                except Exception as e:
                    failed_stages.append(self._get_stage_name(extractor))
                    errors.append(e)
                    raise  # Extractor failures always raise

            # Merge datasets if multiple extractors
            from vulnflow.dataset import VulnDataset

            dataset = VulnDataset.merge(datasets)

        # Stage 2: Transform (enrichers + transforms)
        if self.transformers and dataset is not None:
            import asyncio

            from vulnflow.utils.transforms import partition_transforms

            async_transforms, sync_transforms = partition_transforms(self.transformers)

            # Run async concurrently
            if async_transforms:
                async_succ, async_fail, async_err = asyncio.run(
                    self._run_async_transforms(dataset, async_transforms, strict)
                )
                successful_stages.extend(async_succ)
                failed_stages.extend(async_fail)
                errors.extend(async_err)

            # Run sync sequentially
            if sync_transforms:
                sync_succ, sync_fail, sync_err = self._run_sync_transforms(
                    dataset, sync_transforms, strict
                )
                successful_stages.extend(sync_succ)
                failed_stages.extend(sync_fail)
                errors.extend(sync_err)

        # Stage 3: Load
        loader_outputs: dict[str, Any] = {}
        if self.loaders and dataset is not None:
            for loader in self.loaders:
                try:
                    output = loader.load(dataset)
                    loader_name = self._get_stage_name(loader)
                    loader_outputs[loader_name] = output
                    successful_stages.append(loader_name)
                except Exception as e:
                    failed_stages.append(self._get_stage_name(loader))
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
