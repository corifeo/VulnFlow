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

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from vulnflow.dataset import VulnDataset
from vulnflow.executor import PipelineExecutor
from vulnflow.extractors.base import BaseExtractor
from vulnflow.loaders.base import BaseLoader
from vulnflow.registry import Registry
from vulnflow.transforms.base import BaseTransform
from vulnflow.utils.transforms import partition_transforms


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

    extractors: list[BaseExtractor] = field(default_factory=list)
    transformers: list[BaseTransform] = field(default_factory=list)
    loaders: list[BaseLoader] = field(default_factory=list)

    def run(
        self,
        dataset: VulnDataset | None = None,
        strict: bool = False,
        parallel: bool = False,
        workers: int = 4,
    ) -> PipelineResult:
        """Execute pipeline stages: extract → transform → load.

        Args:
            dataset: Optional pre-existing dataset (skips extraction)
            strict: If True, raise on transformer errors. If False, collect errors.
            parallel: If True, run extract→load pairs in parallel (Pattern 1).
                      Requires len(extractors) == len(loaders) and no transforms.
            workers: Max parallel workers when parallel=True (default 4)

        Returns:
            PipelineResult with dataset, metadata, and errors
        """
        if parallel:
            return self._run_parallel(workers=workers)

        start_time = time.time()
        errors: list[Exception] = []
        successful_stages: list[str] = []
        failed_stages: list[str] = []

        # Stage 1: Extract
        if dataset is None and self.extractors:
            dataset, ext_succ, ext_fail, ext_err = PipelineExecutor.run_extractors(
                self.extractors
            )
            successful_stages.extend(ext_succ)
            failed_stages.extend(ext_fail)
            errors.extend(ext_err)

        # Stage 2: Transform (enrichers + transforms)
        if self.transformers and dataset is not None:
            async_transforms, sync_transforms = partition_transforms(self.transformers)

            # Run async concurrently
            if async_transforms:
                async_succ, async_fail, async_err = asyncio.run(
                    PipelineExecutor.run_async_transforms(
                        dataset, async_transforms, strict
                    )
                )
                successful_stages.extend(async_succ)
                failed_stages.extend(async_fail)
                errors.extend(async_err)

            # Run sync sequentially
            if sync_transforms:
                sync_succ, sync_fail, sync_err = PipelineExecutor.run_sync_transforms(
                    dataset, sync_transforms, strict
                )
                successful_stages.extend(sync_succ)
                failed_stages.extend(sync_fail)
                errors.extend(sync_err)

        # Stage 3: Load
        loader_outputs: dict[str, Any] = {}
        if self.loaders and dataset is not None:
            loader_outputs, load_succ, load_fail, load_err = (
                PipelineExecutor.run_loaders(dataset, self.loaders)
            )
            successful_stages.extend(load_succ)
            failed_stages.extend(load_fail)
            errors.extend(load_err)

        duration = time.time() - start_time

        return PipelineResult(
            dataset=dataset,
            loader_outputs=loader_outputs,
            errors=errors,
            successful_stages=successful_stages,
            failed_stages=failed_stages,
            duration=duration,
        )

    def _run_parallel(self, workers: int = 4) -> PipelineResult:
        """Execute Pattern 1: parallel Extract→Load mini-pipelines.

        Validates pipeline structure for parallel execution:
        - At least one extractor must exist
        - Number of extractors must equal number of loaders (1:1 pairing)
        - No transforms allowed (apply after merging)

        Args:
            workers: Max parallel workers (default 4)

        Returns:
            PipelineResult with merged dataset and metadata

        Raises:
            PipelineError: If pipeline structure is invalid for parallel execution
        """
        from vulnflow.exceptions import PipelineError

        if not self.extractors:
            raise PipelineError(
                "Parallel execution requires at least one extractor"
            )
        if len(self.extractors) != len(self.loaders):
            raise PipelineError(
                f"Parallel execution requires equal extractors and loaders: "
                f"got {len(self.extractors)} extractors, {len(self.loaders)} loaders"
            )
        if self.transformers:
            raise PipelineError(
                "Parallel execution requires no transforms. "
                "Apply transforms after merging: dataset.transform([...])"
            )

        # TODO: delegate to PipelineExecutor.run_parallel() in Task 3
        raise NotImplementedError("Parallel execution not yet implemented")

    def merge(self, other: VulnPipeline) -> VulnPipeline:
        """Merge pipeline stages horizontally by type (immutable).

        Combines extractors, transformers, and loaders from both pipelines
        into a single pipeline. During execution, all extractors run first
        (sequentially), then all transformers, then all loaders.

        This is horizontal stage merging, NOT sequential composition where
        the output of one pipeline feeds into another. Instead, it merges
        stage lists by type:
        - extractors: self.extractors + other.extractors
        - transformers: self.transformers + other.transformers
        - loaders: self.loaders + other.loaders

        Args:
            other: Pipeline whose stages will be merged with this one

        Returns:
            New VulnPipeline with all stages concatenated by type.
            Original pipelines are unchanged (immutable).

        Example:
            >>> p1 = VulnPipeline(extractors=[NVDExtractor("data.json")])
            >>> p2 = VulnPipeline(transformers=[KEVEnricher()])
            >>> merged = p1.merge(p2)
            >>> # merged has 1 extractor and 1 transformer
            >>> # When run(), extractor runs first, then transformer
        """
        return VulnPipeline(
            extractors=self.extractors + other.extractors,
            transformers=self.transformers + other.transformers,
            loaders=self.loaders + other.loaders,
        )

    def _validate_transform_only(self, pipeline: VulnPipeline, context: str) -> None:
        """Validate that a pipeline contains only transformers.

        Args:
            pipeline: Pipeline to validate
            context: Context string for error messages (e.g., "self", "other")

        Raises:
            ValueError: If pipeline contains extractors or loaders
        """
        if pipeline.extractors:
            raise ValueError(
                f"then() requires transform-only pipelines but {context} has extractors. "
                "Use merge() for general stage combining."
            )
        if pipeline.loaders:
            raise ValueError(
                f"then() requires transform-only pipelines but {context} has loaders. "
                "Use merge() for general stage combining."
            )

    def then(self, other: VulnPipeline) -> VulnPipeline:
        """Chain transform-only pipelines (sklearn-like behavior).

        Validates that both pipelines contain ONLY transformers (no extractors
        or loaders), then concatenates their transformer lists. This is for
        building reusable transform chains.

        Args:
            other: Pipeline whose transformers will be chained after this one

        Returns:
            New VulnPipeline with concatenated transformers only.
            Original pipelines are unchanged (immutable).

        Raises:
            ValueError: If either pipeline contains extractors or loaders

        Example:
            >>> from vulnflow.transforms.computed import CVDStateTransform
            >>> from vulnflow.transforms.enrichers import KEVEnricher, EPSSEnricher
            >>> cvd = VulnPipeline(transformers=[CVDStateTransform()])
            >>> enrich = VulnPipeline(transformers=[KEVEnricher(), EPSSEnricher()])
            >>> transform_chain = cvd.then(enrich)
            >>> # transform_chain has 3 transformers, no extractors/loaders
        """
        # Validate both pipelines are transform-only
        self._validate_transform_only(self, "self")
        self._validate_transform_only(other, "other")

        # Chain transformers
        return VulnPipeline(transformers=self.transformers + other.transformers)

    @classmethod
    def chain(cls, pipelines: list[VulnPipeline]) -> VulnPipeline:
        """Merge multiple pipelines into one (horizontal stage concatenation).

        Horizontally concatenates extractors, transformers, and loaders from
        all pipelines. Equivalent to calling .merge() repeatedly.

        During execution:
        1. All extractors run sequentially (merge outputs)
        2. All transformers run (async concurrently, sync sequentially)
        3. All loaders run sequentially

        This is NOT sequential pipeline execution. It merges stage lists.

        Args:
            pipelines: List of pipelines whose stages will be merged

        Returns:
            New VulnPipeline with all stages concatenated by type

        Example:
            >>> extract = VulnPipeline(extractors=[NVDExtractor("data.json")])
            >>> transform = VulnPipeline(transformers=[KEVEnricher()])
            >>> load = VulnPipeline(loaders=[ParquetLoader("out.parquet")])
            >>> full = VulnPipeline.chain([extract, transform, load])
            >>> # Equivalent to: extract.merge(transform).merge(load)
        """
        if not pipelines:
            return cls()

        result = pipelines[0]
        for next_pipeline in pipelines[1:]:
            result = result.merge(next_pipeline)

        return result

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> VulnPipeline:
        """Build pipeline from configuration dict.

        Enables config-driven pipeline construction for production use.
        Components are looked up via Registry and instantiated with params.

        Args:
            config: Dict with 'extractors', 'transformers', 'loaders' keys.
                    Each value is a list of component specs:
                    - String spec: "kev" (uses default constructor)
                    - Dict spec: {"type": "parquet", "path": "out.parquet"}

        Returns:
            Configured VulnPipeline instance

        Raises:
            KeyError: If component type not found in Registry

        Example:
            >>> config = {
            ...     "extractors": [{"type": "nvd_json", "path": "data.json"}],
            ...     "transformers": ["KEVEnricher"],
            ...     "loaders": [{"type": "parquet", "path": "out.parquet"}]
            ... }
            >>> pipeline = VulnPipeline.from_config(config)
        """
        extractors: list[BaseExtractor] = []
        transformers: list[BaseTransform] = []
        loaders: list[BaseLoader] = []

        # Build extractors
        for spec in config.get("extractors", []):
            if isinstance(spec, str):
                # String spec: use default constructor
                extractor_cls = Registry.get_extractor(spec)
                extractors.append(extractor_cls())
            else:
                # Dict spec: extract type and params
                component_type = spec["type"]
                params = {k: v for k, v in spec.items() if k != "type"}
                extractor_cls = Registry.get_extractor(component_type)
                extractors.append(extractor_cls(**params))

        # Build transformers (enrichers + transforms)
        for spec in config.get("transformers", []):
            if isinstance(spec, str):
                # String spec: use default constructor
                transform_cls = Registry.get_transform(spec)
                transformers.append(transform_cls())
            else:
                # Dict spec: extract type and params
                component_type = spec["type"]
                params = {k: v for k, v in spec.items() if k != "type"}
                transform_cls = Registry.get_transform(component_type)
                transformers.append(transform_cls(**params))

        # Build loaders
        for spec in config.get("loaders", []):
            if isinstance(spec, str):
                # String spec: use default constructor
                loader_cls = Registry.get_loader(spec)
                loaders.append(loader_cls())
            else:
                # Dict spec: extract type and params
                component_type = spec["type"]
                params = {k: v for k, v in spec.items() if k != "type"}
                loader_cls = Registry.get_loader(component_type)
                loaders.append(loader_cls(**params))

        return cls(
            extractors=extractors,
            transformers=transformers,
            loaders=loaders,
        )
