import argparse
from real_workers.plotting import generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scientific figures from measured worker CSVs")
    parser.add_argument("--results", help="Defaults to full results if present, otherwise quick results")
    parser.add_argument("--output", default="plots/real_workers")
    args = parser.parse_args()
    generate(args.results, args.output)
