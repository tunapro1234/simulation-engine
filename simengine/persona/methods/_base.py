"""PersonaMethod protocol, a function-backed implementation and data-file resolution for data-backed methods."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

ENV_DATA = "SIMENGINE_DATA"
DATA_SUBDIR = "persona_methods"


class PersonaDataNotFound(FileNotFoundError):
    """A data-backed persona method could not find its file under data_root / $SIMENGINE_DATA."""


@runtime_checkable
class PersonaMethod(Protocol):
    """A persona injection method: `sample(n, seed)` returns n persona texts ("" = no persona)."""
    name: str
    needs_data: bool

    def sample(self, n: int, seed: Any = None, data_root: str | None = None) -> list[str]: ...


@dataclass
class FunctionMethod:
    """PersonaMethod backed by `draw(n, seed, data_root) -> list[(text, params)]`.

    `params` are the per-persona parameters (type, dials, value, row id, ...) stored next to every simulated
    answer. `source` and `paper` are provenance strings shown by `list_methods()`."""
    name: str
    draw_fn: Callable[..., list[tuple[str, dict]]]
    needs_data: bool = False
    description: str = ""
    paper: str = ""
    source: str = ""
    data_file: str | None = None
    info: dict = field(default_factory=dict)

    def draw(self, n: int, seed: Any = None, data_root: str | None = None) -> list[tuple[str, dict]]:
        if n < 0:
            raise ValueError("n must be >= 0")
        return self.draw_fn(n, seed, data_root)

    def sample(self, n: int, seed: Any = None, data_root: str | None = None) -> list[str]:
        return [t for t, _ in self.draw(n, seed, data_root)]

    def describe(self) -> dict:
        return {"name": self.name, "needs_data": self.needs_data, "description": self.description,
                "paper": self.paper, "source": self.source,
                "data_file": f"<root>/{DATA_SUBDIR}/{self.data_file}" if self.data_file else None}


def data_path(filename: str, data_root: str | os.PathLike | None = None) -> str:
    """Locate `<root>/persona_methods/<filename>` (or `<root>/<filename>` when data_root is the persona_methods
    folder itself). `root` is `data_root`, else $SIMENGINE_DATA. Raises PersonaDataNotFound with the expected
    layout."""
    root = data_root if data_root is not None else os.environ.get(ENV_DATA)
    layout = f"expected layout: <root>/{DATA_SUBDIR}/{filename}"
    if not root:
        raise PersonaDataNotFound(f"no data_root given and ${ENV_DATA} is not set ({layout})")
    root = os.fspath(root)
    for cand in (os.path.join(root, DATA_SUBDIR, filename), os.path.join(root, filename)):
        if os.path.isfile(cand):
            return cand
    raise PersonaDataNotFound(f"{filename} not found under {root} ({layout})")


_CACHE: dict[tuple[str, float], Any] = {}


def cached_load(path: str, loader: Callable[[str], Any]) -> Any:
    """Load a data file once per (path, mtime) per process."""
    key = (os.path.abspath(path), os.path.getmtime(path))
    if key not in _CACHE:
        _CACHE[key] = loader(path)
    return _CACHE[key]
