"""Execute each learning notebook in a fresh kernel using this Python interpreter.

Run from the repository root:
  .venv-llm/bin/python mini_llm/learning_path/verify_lessons.py
Requires nbclient and nbformat for verification, not for studying saved outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    root = Path(__file__).resolve().parent
    available = {path.name: path for path in sorted(root.glob("[0-9][0-9]_*.ipynb"))}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Optional notebook filenames to rerun; defaults to all six.")
    args = parser.parse_args()
    unknown = set(args.notebooks) - set(available)
    if unknown:
        parser.error(f"Unknown notebooks: {sorted(unknown)}")
    paths = [available[name] for name in args.notebooks] if args.notebooks else list(available.values())
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="sino-lesson-kernel-") as temporary:
            temp = Path(temporary)
            kernel = temp / "kernels" / "sino-verify"
            kernel.mkdir(parents=True)
            (kernel / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "sino lesson verification", "language": "python"}))
            manager = KernelManager(
                kernel_name="sino-verify", connection_file=str(temp / "connection.json"),
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
            client = NotebookClient(notebook, km=manager, timeout=180,
                                    resources={"metadata": {"path": str(root)}})
            environment = dict(os.environ, MPLCONFIGDIR=str(temp / "matplotlib"))
            try:
                client.execute(env=environment)
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
        codes = sum(cell.cell_type == "code" for cell in notebook.cells)
        print(f"PASS {path.name}: {codes} code cells, {time.perf_counter()-started:.1f}s", flush=True)
    print(f"All {len(paths)} selected notebooks executed and saved successfully.")


if __name__ == "__main__":
    main()
