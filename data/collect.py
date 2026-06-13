"""
Collect Python source files for Phase 1 training.

Primary source: Python standard library (always available in any Python install).
The stdlib provides ~500 well-structured Python files spanning many structural
patterns: class definitions, decorators, nested scopes, comprehensions, etc.

Usage:
    python data/collect.py --output data/corpus.txt
    python data/collect.py --list-only
"""

import sys
import os
import argparse
import sysconfig
from pathlib import Path
from typing import List


def find_stdlib_files(min_lines: int = 20, max_lines: int = 2000) -> List[Path]:
    """
    Find all .py files in the Python standard library.
    Filters by line count to exclude trivially short or extremely long files.
    """
    stdlib_path = Path(sysconfig.get_paths()["stdlib"])
    candidates = list(stdlib_path.rglob("*.py"))

    # Also check the 'lib' directory in the prefix for platforms where stdlib
    # lives elsewhere (e.g. Linux distributions)
    prefix_lib = Path(sys.prefix) / "lib"
    if prefix_lib != stdlib_path and prefix_lib.exists():
        candidates += list(prefix_lib.rglob("*.py"))

    filtered = []
    for p in sorted(set(candidates)):
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            n = len(lines)
            if min_lines <= n <= max_lines:
                filtered.append(p)
        except (OSError, PermissionError):
            pass

    return filtered


def collect_corpus(output_path: str, min_lines: int = 20, max_lines: int = 2000) -> int:
    """
    Write all collected Python file paths to a text file, one path per line.
    Returns the number of files collected.
    """
    files = find_stdlib_files(min_lines, max_lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(f) for f in files), encoding="utf-8")
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Python stdlib files for QuINN training")
    parser.add_argument("--output", default="data/corpus.txt", help="Output file path list")
    parser.add_argument("--min-lines", type=int, default=20)
    parser.add_argument("--max-lines", type=int, default=2000)
    parser.add_argument("--list-only", action="store_true", help="Just print file count")
    args = parser.parse_args()

    if args.list_only:
        files = find_stdlib_files(args.min_lines, args.max_lines)
        print(f"Found {len(files)} Python files in stdlib")
        for f in files[:10]:
            print(f"  {f}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")
    else:
        n = collect_corpus(args.output, args.min_lines, args.max_lines)
        print(f"Collected {n} files → {args.output}")
