import os
import json
from pathlib import Path
from collections import defaultdict

# Set input path
input_root = Path("/data2/ac2220/macenko_pipeline_output")

# Store runtimes
total_runtimes = []
ti_runtimes = defaultdict(list)

# Loop through results
for slide_dir in input_root.iterdir():
    if not slide_dir.is_dir():
        continue

    slide_name = slide_dir.name
    results_path = slide_dir / f"{slide_name}_wbc_results.json"

    if not results_path.exists():
        print(f"⚠️ Missing: {results_path}")
        continue

    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Error loading {results_path}: {e}")
        continue

    runtime = results.get("runtime_min")
    ti_score = results.get("TI")

    if runtime is None:
        print(f"⚠️ No runtime in {results_path}")
        continue

    total_runtimes.append(runtime)
    if ti_score is not None:
        ti_runtimes[ti_score].append(runtime)

# Calculate averages
overall_avg = sum(total_runtimes) / len(total_runtimes) if total_runtimes else 0

print(f"\n📊 Overall average runtime: {overall_avg:.2f} minutes")

for ti in sorted(ti_runtimes.keys()):
    runtimes = ti_runtimes[ti]
    avg_ti = sum(runtimes) / len(runtimes)
    print(f"  TI = {ti}: {avg_ti:.2f} minutes over {len(runtimes)} slides")
