"""Execute notebook.ipynb in place, saving the outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1] if len(sys.argv) > 1 else "notebook.ipynb")

notebook = nbformat.read(str(path), as_version=4)
client = NotebookClient(
    notebook,
    timeout=1800,
    kernel_name="python3",
    resources={"metadata": {"path": str(path.parent.resolve())}},
)
client.execute()

nbformat.write(notebook, str(path))
print(f"Executed {path}: {len(notebook.cells)} cells")
