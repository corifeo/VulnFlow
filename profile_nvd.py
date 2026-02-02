"""Profile NVD extraction performance."""
import cProfile
import pstats
from vulnflow.dataset import VulnDataset

def profile_nvd():
    dataset = VulnDataset.from_nvd("data/nvdcve-2.0-2025.json")
    print(f"Loaded {len(dataset)} records")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    profile_nvd()
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(30)
