"""
Pipeline Executor - Stage execution logic

Provides:
- PipelineExecutor: Stateless executor for extractors, transforms, and loaders
- get_stage_name: Utility to extract stage name from object

Layer: Core ETL execution
Dependencies: dataset, extractors, transforms, loaders
Used by: pipeline
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset
    from vulnflow.extractors.base import BaseExtractor
    from vulnflow.loaders.base import BaseLoader

__all__ = ["PipelineExecutor", "get_stage_name"]

logger = logging.getLogger(__name__)


def get_stage_name(obj: Any) -> str:
    """Get stage name from object (prefer name attribute, fallback to class name)."""
    return cast(str, getattr(obj, "name", obj.__class__.__name__))


class PipelineExecutor:
    """Stateless executor for pipeline stages."""

    @staticmethod
    def run_extractors(
        extractors: list[BaseExtractor],
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
        from vulnflow.dataset import VulnDataset

        if not extractors:
            return VulnDataset(), [], [], []

        datasets: list[VulnDataset] = []
        successful: list[str] = []
        failed: list[str] = []
        errors: list[Exception] = []

        for extractor in extractors:
            ext_name = get_stage_name(extractor)
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

    @staticmethod
    async def run_async_transforms(
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
                successful = [get_stage_name(t) for t in transforms]
            else:
                # Check each result
                for t, result in zip(transforms, results, strict=True):
                    t_name = get_stage_name(t)
                    if isinstance(result, Exception):
                        logger.warning("Transform %s failed: %s", t_name, result)
                        errors.append(result)
                        failed.append(t_name)
                    else:
                        successful.append(t_name)
        except Exception as e:
            failed = [get_stage_name(t) for t in transforms]
            errors.append(e)
            raise

        return successful, failed, errors

    @staticmethod
    def run_sync_transforms(
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
            t_name = get_stage_name(t)
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

    @staticmethod
    def run_loaders(
        dataset: VulnDataset,
        loaders: list[BaseLoader],
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
            loader_name = get_stage_name(loader)
            try:
                output = loader.load(dataset)
                loader_outputs[loader_name] = output
                successful.append(loader_name)
            except Exception as e:
                failed.append(loader_name)
                errors.append(e)
                raise  # Loader failures always raise

        return loader_outputs, successful, failed, errors
