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
from typing import Any, cast

from vulnflow.dataset import VulnDataset
from vulnflow.extractors.base import Extractor
from vulnflow.loaders.base import Loader
from vulnflow.registry import Registry
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

    extractors: list[Extractor] = field(default_factory=list)
    transformers: list[BaseTransform] = field(default_factory=list)
    loaders: list[Loader] = field(default_factory=list)

    def _get_stage_name(self, obj: Any) -> str:
        """Get stage name from object (prefer name attribute, fallback to class name)."""
        return cast(str, getattr(obj, "name", obj.__class__.__name__))

    def _run_extractors(
        self,
        extractors: list[Extractor],
    ) -> tuple[VulnDataset, list[str], list[str], list[Exception]]:
        """Execute extractors sequentially and merge results.

        Extractors always run in strict mode (failures always raise).

        Args:
            extractors: List of extractor instances

        Returns:
            Tuple of (merged_dataset, successful_stages, failed_stages, errors)

        Raises:
            Exception: If any extractor fails (strict mode only)
        """
        if not extractors:
            return VulnDataset(), [], [], []

        datasets: list[VulnDataset] = []
        successful: list[str] = []
        failed: list[str] = []
        errors: list[Exception] = []

        for extractor in extractors:
            ext_name = self._get_stage_name(extractor)
            try:
                ds = extractor.extract()
                datasets.append(ds)
                successful.append(ext_name)
            except Exception as e:
                failed.append(ext_name)
                errors.append(e)
                raise  # Extractor failures always raise

        # Merge all extracted datasets
        merged = VulnDataset.merge(datasets)
        return merged, successful, failed, errors

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

    def _run_loaders(
        self,
        dataset: VulnDataset,
        loaders: list[Loader],
    ) -> tuple[dict[str, Any], list[str], list[str], list[Exception]]:
        """Execute loaders sequentially and collect outputs.

        Loaders always run in strict mode (failures always raise).

        Args:
            dataset: Dataset to load
            loaders: List of loader instances

        Returns:
            Tuple of (loader_outputs, successful_stages, failed_stages, errors)

        Raises:
            Exception: If any loader fails (strict mode only)
        """
        if not loaders:
            return {}, [], [], []

        loader_outputs: dict[str, Any] = {}
        successful: list[str] = []
        failed: list[str] = []
        errors: list[Exception] = []

        for loader in loaders:
            loader_name = self._get_stage_name(loader)
            try:
                output = loader.load(dataset)
                loader_outputs[loader_name] = output
                successful.append(loader_name)
            except Exception as e:
                failed.append(loader_name)
                errors.append(e)
                raise  # Loader failures always raise

        return loader_outputs, successful, failed, errors

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
            dataset, ext_succ, ext_fail, ext_err = self._run_extractors(self.extractors)
            successful_stages.extend(ext_succ)
            failed_stages.extend(ext_fail)
            errors.extend(ext_err)

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
            loader_outputs, load_succ, load_fail, load_err = self._run_loaders(
                dataset, self.loaders
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
            >>> normalize = VulnPipeline(transformers=[NormalizeTransform()])
            >>> enrich = VulnPipeline(transformers=[KEVEnricher(), EPSSEnricher()])
            >>> transform_chain = normalize.then(enrich)
            >>> # transform_chain has 3 transformers, no extractors/loaders
        """
        # Validate self has no extractors or loaders
        if self.extractors:
            raise ValueError(
                "then() requires transform-only pipelines (no extractors). "
                "Use merge() for general stage combining."
            )
        if self.loaders:
            raise ValueError(
                "then() requires transform-only pipelines (no loaders). "
                "Use merge() for general stage combining."
            )

        # Validate other has no extractors or loaders
        if other.extractors:
            raise ValueError(
                "then() requires transform-only pipelines (no extractors). "
                "Use merge() for general stage combining."
            )
        if other.loaders:
            raise ValueError(
                "then() requires transform-only pipelines (no loaders). "
                "Use merge() for general stage combining."
            )

        # Chain transformers
        return VulnPipeline(transformers=self.transformers + other.transformers)

    def chain_with(self, other: VulnPipeline) -> VulnPipeline:
        """Chain pipelines with format validation (load → extract).

        Validates format compatibility between this pipeline's loaders and the
        other pipeline's extractors. This is useful for multi-stage file-based
        workflows where output format must match input format.

        Validation checks (in order):
        1. First pipeline must have loaders
        2. Second pipeline must have extractors
        3. Counts must match: len(self.loaders) == len(other.extractors)
        4. Formats must match by position: loader[i].format == extractor[i].format

        If all validations pass, merges stages horizontally (like merge()).

        Args:
            other: Pipeline whose extractors must be compatible with our loaders

        Returns:
            New VulnPipeline with all stages merged horizontally.
            Original pipelines are unchanged (immutable).

        Raises:
            ValueError: If any validation check fails

        Example:
            >>> # Pipeline 1: extract → load to parquet
            >>> p1 = VulnPipeline(
            ...     extractors=[NVDExtractor("data.json")],
            ...     loaders=[ParquetLoader("out.parquet")]
            ... )
            >>> # Pipeline 2: extract from parquet → load to json
            >>> p2 = VulnPipeline(
            ...     extractors=[ParquetExtractor("out.parquet")],
            ...     loaders=[JSONLoader("out.json")]
            ... )
            >>> # Chain them (validates parquet format compatibility)
            >>> chained = p1.chain_with(p2)
        """
        # Validation 1: First pipeline must have loaders
        if not self.loaders:
            raise ValueError(
                "First pipeline must have loaders to chain with another pipeline. "
                "Use merge() for general stage combining."
            )

        # Validation 2: Second pipeline must have extractors
        if not other.extractors:
            raise ValueError(
                "Second pipeline must have extractors to chain with another pipeline. "
                "Use merge() for general stage combining."
            )

        # Validation 3: Counts must match
        if len(self.loaders) != len(other.extractors):
            raise ValueError(
                f"Count mismatch: first pipeline has {len(self.loaders)} loaders "
                f"but second pipeline has {len(other.extractors)} extractors. "
                "Loader and extractor counts must match for chaining."
            )

        # Validation 4: Formats must match by position
        for i, (loader, extractor) in enumerate(
            zip(self.loaders, other.extractors, strict=True)
        ):
            loader_format = cast(str, getattr(loader, "format", None))
            extractor_format = cast(str, getattr(extractor, "format", None))

            if loader_format != extractor_format:
                raise ValueError(
                    f"Format mismatch at position {i}: "
                    f"loader format '{loader_format}' != extractor format '{extractor_format}'. "
                    "Loader and extractor formats must match by position for chaining."
                )

        # All validations passed - merge horizontally
        return self.merge(other)

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
        extractors: list[Extractor] = []
        transformers: list[BaseTransform] = []
        loaders: list[Loader] = []

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
                # Try enricher first, then transform
                transform_cls = Registry.get_enricher(spec)
                if transform_cls is None:
                    transform_cls = Registry.get_transform(spec)
                if transform_cls is None:
                    raise KeyError(f"Transform not found: {spec}")
                transformers.append(transform_cls())
            else:
                # Dict spec: extract type and params
                component_type = spec["type"]
                params = {k: v for k, v in spec.items() if k != "type"}
                # Try enricher first, then transform
                transform_cls = Registry.get_enricher(component_type)
                if transform_cls is None:
                    transform_cls = Registry.get_transform(component_type)
                if transform_cls is None:
                    raise KeyError(f"Transform not found: {component_type}")
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
