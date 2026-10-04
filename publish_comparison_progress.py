import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def publish_progress(
    output_dir: Path,
    epoch: int,
    generation: str,
    expected_rows: int,
    rows_per_chunk: int,
    poll_seconds: float,
) -> None:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    temporary_path = output_dir / "manifest_progress.tmp"
    columns = ["Referencia", "Pose", "IA cruda", "IA remapeada", "Ground truth"]
    last_chunk_count = 0

    while True:
        chunks = sorted(
            output_dir.glob(f"comparison_epoch_{epoch:03d}_{generation}_part_*.png")
        )
        chunk_count = len(chunks)
        row_count = min(expected_rows, chunk_count * rows_per_chunk)
        complete = row_count >= expected_rows

        if chunk_count and chunk_count != last_chunk_count:
            try:
                current: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = {}
            if current.get("generation") == generation and current.get("complete") is True:
                return

            manifest: dict[str, Any] = {
                "epoch": epoch,
                "generation": generation,
                "row_count": row_count,
                "complete": complete,
                "columns": columns,
                "chunks": [path.name for path in chunks],
            }
            temporary_path.write_text(json.dumps(manifest), encoding="utf-8")
            os.replace(temporary_path, manifest_path)
            last_chunk_count = chunk_count
            if chunk_count % 5 == 0 or complete:
                print(f"Published {row_count}/{expected_rows} comparison rows", flush=True)
            if complete:
                return

        time.sleep(poll_seconds)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--rows-per-chunk", type=int, default=32)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "training_samples" / "live_comparison",
    )
    args = parser.parse_args()
    publish_progress(
        args.output_dir,
        args.epoch,
        args.generation,
        args.expected_rows,
        args.rows_per_chunk,
        args.poll_seconds,
    )


if __name__ == "__main__":
    main()