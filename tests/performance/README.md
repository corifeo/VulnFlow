# Performance Tests

Performance benchmarks using real-world data from the `data/` directory.

## Requirements

Place these files in the `data/` directory:
- `nvdcve-2.0-2025.json` - NVD CVE feed (~42k records)
- `known_exploited_vulnerabilities.csv` - CISA KEV catalog
- `epss_scores-2026-01-25.csv` - FIRST EPSS scores

## Running Tests

```bash
# Run all performance tests
uv run pytest tests/performance/ -v

# Run with performance output
uv run pytest tests/performance/ -v -s

# Run directly for detailed profiling
uv run python tests/performance/test_large_dataset_performance.py
```

## Performance Targets

| Test | Target | Measures |
|------|--------|----------|
| NVD Extraction | <5s | Parse 42k CVEs from JSON |
| KEV Enrichment | <2s | Vectorized join on 42k records |
| EPSS Enrichment | <2s | Vectorized join on 42k records |
| Full Pipeline | <10s | Extract + Transform end-to-end |
| Memory Usage | <500MB | Peak memory for 42k records |
| DataFrame Export | <3s | Convert to pandas/polars |
| Parquet Round-trip | <5s | Save + Load 42k records |

## Optimization Guidelines

If tests fail:

1. **NVD Extraction Slow** (>5s)
   - Check JSON parsing (use `orjson` if available)
   - Profile dataclass creation

2. **Enrichment Slow** (>2s)
   - Ensure vectorized operations (no Python for loops)
   - Use `np.isin()` for matching
   - Check dict lookup efficiency

3. **High Memory** (>500MB)
   - Verify columnar storage (not object arrays)
   - Check for duplicate data
   - Profile with `tracemalloc`

4. **DataFrame Export Slow** (>3s)
   - Ensure direct numpy → DataFrame (no iteration)
   - Check JSON deserialization performance

## Profiling

```bash
# CPU profiling
uv run python -m cProfile -o profile.stats tests/performance/test_large_dataset_performance.py

# View results
uv run python -m pstats profile.stats
> sort cumulative
> stats 20

# Memory profiling
uv run python -m memory_profiler tests/performance/test_large_dataset_performance.py
```

## Expected Output

```
============================================================
Full Pipeline Performance Report
============================================================
Records processed:        42,047
Extraction time:            3.45s
Transform time:             2.12s
Total time:                 5.57s

KEV matches:                1,234 (2.9%)
EPSS matches:              41,523 (98.8%)

Overall throughput:         7,548 records/sec
============================================================
```
