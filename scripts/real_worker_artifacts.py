"""Archive, restore, or independently verify measured real-worker artifacts."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real_workers.artifacts import archive_events, restore_events, verify

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/real_workers")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--restore-events", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.archive:
        result = archive_events(args.results)
        print(f"Archived {result['rows']} measured events in {len(result['parts'])} lossless parts")
    if args.restore_events:
        print(restore_events(args.results))
    if args.verify or not (args.archive or args.restore_events):
        print(verify(args.results))
