"""Command-line entry point for transcript extraction."""

import argparse
from pathlib import Path

from app.processing import process_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract grounded items from a transcript.")
    parser.add_argument("input", nargs="?", type=Path, default=Path("input_data/input2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model", help="Override GEMINI_MODEL_NAME from .env")
    parser.add_argument("--max-chars", type=int, default=12_000)
    args = parser.parse_args()

    output_path = process_file(
        args.input,
        args.output_dir,
        model=args.model,
        max_chars=args.max_chars,
    )
    print(f"Wrote extraction to {output_path}")


if __name__ == "__main__":
    main()
