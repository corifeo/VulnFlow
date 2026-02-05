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
import time
from typing import TYPE_CHECKING, Any, cast

from vulnflow.extractors.base import BaseExtractor
from vulnflow.loaders.base import BaseLoader

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset
    from vulnflow.pipeline import PipelineResult

__all__ = ["PipelineExecutor", "get_stage_name"]

logger = logging.getLogger(__name__)


def get_stage_name(obj: Any) -> str:
    """Get stage name from object (prefer name attribute, fallback to class name)."""
    return cast(str, getattr(obj, "name", obj.__class__.__name__))


def _run_mini_pipeline(
    extractor: BaseExtractor,
    loader: BaseLoader,
) -> dict[str, Any]:
    """Worker function: Extract → Load.

    Must be module-level (not a method) for pickling with ProcessPoolExecutor.

    Args:
        extractor: Extractor instance to run
        loader: Loader instance to run

    Returns:
        Loader output metadata dict
    """
    dataset = extractor.extract()
    return loader.load(dataset)


class PipelineExecutor:
    """Stateless executor for pipeline stages."""

    @staticmethod
    def run_extractors(
        extractors: list[BaseExtractor],
    ) -> tuple[VulnDataset, list[str], list[str], list[Exception]]:
        """Execute extractors sequentially and merge results.

        Extractors always run in strict mode (failures always raise).
        Shows progress bar when multiple extractors present.

        Args:
            extractors: List of extractor instances

        Returns:
            Tuple of (merged_dataset, successful_stages, failed_stages, errors)

        Raises:
            Exception: If any extractor fails (strict mode only)
        """
        from vulnflow.dataset import VulnDataset
        from vulnflow.utils.progress import progress_bar

        if not extractors:
            return VulnDataset(), [], [], []

        datasets: list[VulnDataset] = []
        successful: list[str] = []
        failed: list[str] = []
        errors: list[Exception] = []

        # Show progress bar when multiple extractors
        with progress_bar(
            total=len(extractors),
            desc="Extracting data",
            unit="extractors",
        ) as pbar:
            for extractor in extractors:
                ext_name = get_stage_name(extractor)
                pbar.set_postfix_str(f"[{ext_name}]")
                try:
                    ds = extractor.extract()
                    datasets.append(ds)
                    successful.append(ext_name)
                except Exception as e:
                    failed.append(ext_name)
                    errors.append(e)
                    raise  # Extractor failures always raise
                finally:
                    pbar.update(1)

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
        Shows progress bar when multiple loaders present.

        Args:
            dataset: Dataset to load
            loaders: List of loader instances

        Returns:
            Tuple of (loader_outputs, successful_stages, failed_stages, errors)

        Raises:
            Exception: If any loader fails (strict mode only)
        """
        from vulnflow.utils.progress import progress_bar

        if not loaders:
            return {}, [], [], []

        loader_outputs: dict[str, Any] = {}
        successful: list[str] = []
        failed: list[str] = []
        errors: list[Exception] = []

        # Show progress bar when multiple loaders
        with progress_bar(
            total=len(loaders),
            desc="Loading data",
            unit="loaders",
        ) as pbar:
            for loader in loaders:
                loader_name = get_stage_name(loader)
                pbar.set_postfix_str(f"[{loader_name}]")
                try:
                    output = loader.load(dataset)
                    loader_outputs[loader_name] = output
                    successful.append(loader_name)
                except Exception as e:
                    failed.append(loader_name)
                    errors.append(e)
                    raise  # Loader failures always raise
                finally:
                    pbar.update(1)

        return loader_outputs, successful, failed, errors

    @staticmethod
    def run_parallel(
        extractors: list[BaseExtractor],
        loaders: list[BaseLoader],
        workers: int = 4,
    ) -> PipelineResult:
        """Run parallel mini-pipelines: extractor[i] → loader[i].

        Pattern 1: Each worker runs one Extract→Load pair independently.

        Args:
            extractors: List of extractor instances (one per source)
            loaders: List of loader instances (must match extractor count)
            workers: Number of parallel worker processes

        Returns:
            PipelineResult with loader_outputs, no dataset (dataset=None)

        Raises:
            PipelineError: If any worker fails
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from vulnflow.exceptions import PipelineError
        from vulnflow.pipeline import PipelineResult

        start_time = time.time()
        tasks = list(zip(extractors, loaders, strict=True))
        loader_outputs: dict[str, Any] = {}
        successful: list[str] = []
        failed: list[str] = []
        errors: list[Exception] = []

        # Build unique output keys: disambiguate duplicate loader names
        raw_names = [get_stage_name(loader) for loader in loaders]
        name_counts: dict[str, int] = {}
        for n in raw_names:
            name_counts[n] = name_counts.get(n, 0) + 1
        name_counters: dict[str, int] = {}
        output_keys: list[str] = []
        for n in raw_names:
            if name_counts[n] > 1:
                counter = name_counters.get(n, 0)
                output_keys.append(f"{n}_{counter}")
                name_counters[n] = counter + 1
            else:
                output_keys.append(n)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_mini_pipeline, ext, loader): idx
                for idx, (ext, loader) in enumerate(tasks)
            }

            for future in as_completed(futures):
                idx = futures[future]
                ext_name = get_stage_name(extractors[idx])
                loader_key = output_keys[idx]
                try:
                    result = future.result()
                    loader_outputs[loader_key] = result
                    successful.extend([ext_name, loader_key])
                except Exception as exc:
                    logger.warning("Parallel worker %d failed: %s", idx, exc)
                    failed.extend([ext_name, loader_key])
                    errors.append(exc)

        duration = time.time() - start_time

        if errors:
            raise PipelineError(
                f"Parallel execution failed: "
                f"{len(errors)} of {len(tasks)} workers failed"
            ) from errors[0]

        return PipelineResult(
            dataset=None,
            loader_outputs=loader_outputs,
            errors=errors,
            successful_stages=successful,
            failed_stages=failed,
            duration=duration,
        )
