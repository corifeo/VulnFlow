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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, cast

from vulnflow.extractors.base import BaseExtractor
from vulnflow.loaders.base import BaseLoader
from vulnflow.utils.graph import build_phases
from vulnflow.utils.transforms import is_async_transform

if TYPE_CHECKING:
    from vulnflow.dataset import VulnDataset
    from vulnflow.utils.progress import ProgressReporter

__all__ = ["PipelineExecutor", "get_stage_name"]

logger = logging.getLogger(__name__)


def get_stage_name(obj: Any) -> str:
    """Get stage name from object.

    Priority: label > name > class name.
    label provides short, unique display names (e.g., nvdcve-2.0-2023).
    """
    label = getattr(obj, "label", None)
    if label is not None:
        return cast(str, label)
    return cast(str, getattr(obj, "name", obj.__class__.__name__))


class PipelineExecutor:
    """Stateless executor for pipeline stages."""

    @staticmethod
    def run_extractors(
        extractors: list[BaseExtractor],
    ) -> tuple[VulnDataset, list[str], list[str], list[Exception]]:
        """Execute extractors concurrently and merge results.

        Uses ThreadPoolExecutor for multiple extractors (orjson releases
        GIL, so threads provide real parallelism for JSON parsing).
        Falls back to sequential for single extractor.

        Extractors always run in strict mode (failures always raise).

        Args:
            extractors: List of extractor instances

        Returns:
            Tuple of (merged_dataset, successful_stages, failed_stages, errors)

        Raises:
            Exception: If any extractor fails
        """
        from vulnflow.dataset import VulnDataset
        from vulnflow.utils.progress import ProgressReporter

        if not extractors:
            return VulnDataset(), [], [], []

        datasets: list[VulnDataset] = []
        successful: list[str] = []
        failed: list[str] = []
        errors: list[Exception] = []
        total_records = 0

        with ProgressReporter(
            "Extracting data", total=len(extractors), unit="extractors"
        ) as rpt:
            if len(extractors) == 1:
                # Single extractor: run directly (no thread overhead)
                ext = extractors[0]
                ext_name = get_stage_name(ext)
                rpt.stage_started(ext_name)
                try:
                    ds = ext.extract()
                    count = len(ds)
                    total_records += count
                    datasets.append(ds)
                    successful.append(ext_name)
                    rpt.stage_completed(ext_name, detail=f"{count:,} CVEs")
                except Exception as e:
                    failed.append(ext_name)
                    errors.append(e)
                    rpt.stage_failed(ext_name, error=str(e))
                    raise
            else:
                # Multiple extractors: run concurrently via threads
                for ext in extractors:
                    rpt.stage_started(get_stage_name(ext))

                with ThreadPoolExecutor(
                    max_workers=min(len(extractors), 8)
                ) as pool:
                    futures = {
                        pool.submit(ext.extract): ext for ext in extractors
                    }
                    for future in as_completed(futures):
                        ext = futures[future]
                        ext_name = get_stage_name(ext)
                        try:
                            ds = future.result()
                            count = len(ds)
                            total_records += count
                            datasets.append(ds)
                            successful.append(ext_name)
                            rpt.stage_completed(
                                ext_name, detail=f"{count:,} CVEs"
                            )
                        except Exception as e:
                            failed.append(ext_name)
                            errors.append(e)
                            rpt.stage_failed(ext_name, error=str(e))
                            raise

            if total_records > 0:
                rpt.summary(f"{total_records:,} CVEs total")

        merged = VulnDataset.merge(datasets)
        return merged, successful, failed, errors

    @staticmethod
    async def run_transforms(
        dataset: VulnDataset,
        transforms: list[Any],
        strict: bool,
        reporter: ProgressReporter | None = None,
    ) -> tuple[list[str], list[str], list[Exception]]:
        """Execute transforms in dependency-aware phases.

        Groups transforms into phases using build_phases(). Within each
        phase, async transforms run via gather() and sync transforms run
        via to_thread(). A barrier separates each phase.

        Args:
            dataset: Dataset to transform
            transforms: Transform instances (async and/or sync)
            strict: If True, raise on errors. If False, collect errors.
            reporter: Optional ProgressReporter for stage tracking.

        Returns:
            Tuple of (successful_stages, failed_stages, errors)
        """
        if not transforms:
            return [], [], []

        successful: list[str] = []
        failed: list[str] = []
        errors: list[Exception] = []

        phases = build_phases(transforms)

        for phase in phases:
            phase_tasks: list[tuple[str, Any]] = []

            for t in phase:
                t_name = get_stage_name(t)
                if reporter is not None:
                    reporter.stage_started(t_name)

                if is_async_transform(t):
                    phase_tasks.append((t_name, t.transform(dataset)))
                else:
                    phase_tasks.append(
                        (t_name, asyncio.to_thread(t.transform, dataset))
                    )

            if strict:
                try:
                    await asyncio.gather(*[task for _, task in phase_tasks])
                    for t_name, _ in phase_tasks:
                        successful.append(t_name)
                        if reporter is not None:
                            reporter.stage_completed(t_name)
                except Exception as e:
                    for t_name, _ in phase_tasks:
                        failed.append(t_name)
                        if reporter is not None:
                            reporter.stage_failed(t_name, error=str(e))
                    raise
            else:
                results = await asyncio.gather(
                    *[task for _, task in phase_tasks],
                    return_exceptions=True,
                )
                for (t_name, _), result in zip(
                    phase_tasks, results, strict=True
                ):
                    if isinstance(result, Exception):
                        logger.warning(
                            "Transform %s failed: %s", t_name, result
                        )
                        failed.append(t_name)
                        errors.append(result)
                        if reporter is not None:
                            reporter.stage_failed(t_name, error=str(result))
                    else:
                        successful.append(t_name)
                        if reporter is not None:
                            reporter.stage_completed(t_name)

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
            Exception: If any loader fails
        """
        from vulnflow.utils.progress import ProgressReporter

        if not loaders:
            return {}, [], [], []

        loader_outputs: dict[str, Any] = {}
        successful: list[str] = []
        failed: list[str] = []
        errors: list[Exception] = []
        record_count = len(dataset)

        with ProgressReporter(
            "Loading data", total=len(loaders), unit="loaders"
        ) as rpt:
            for loader in loaders:
                loader_name = get_stage_name(loader)
                rpt.stage_started(loader_name)
                try:
                    output = loader.load(dataset)
                    loader_outputs[loader_name] = output
                    successful.append(loader_name)
                    rpt.stage_completed(
                        loader_name, detail=f"{record_count:,} records"
                    )
                except Exception as e:
                    failed.append(loader_name)
                    errors.append(e)
                    rpt.stage_failed(loader_name, error=str(e))
                    raise

        return loader_outputs, successful, failed, errors
