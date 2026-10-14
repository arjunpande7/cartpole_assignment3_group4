import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

CONFIGS = ["baseline", "adaptive", "stability", "combined"]


def run_one(job):
    config, seed = job
    name = f"{config}_seed{seed}"
    command = ["python", "-m", "src.train", "--config", f"config/{config}.yaml",
               "--seed", str(seed), "--out_dir", "runs", "--weights_dir", "weights"]
    with open(f"runs/{name}.log", "w") as log_file:
        subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"finished {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="e.g. --seeds 0 1")
    parser.add_argument("--parallel", type=int, default=4, help="how many runs at the same time")
    args = parser.parse_args()

    jobs = []
    for seed in args.seeds:
        for config in CONFIGS:
            jobs.append((config, seed))

    print(f"{len(jobs)} runs, {args.parallel} at a time")
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        pool.map(run_one, jobs)
    print("all done")