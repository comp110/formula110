"""Keep file-loaded submissions' controllers packages in separate namespaces."""

from __future__ import annotations

import builtins
import sys
from collections.abc import Sequence
from importlib import util
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, SourceFileLoader
from pathlib import Path
from types import ModuleType
from typing import Any


class _ControllerSourceLoader(SourceFileLoader):
    def __init__(self, name: str, path: str, namespace: str) -> None:
        super().__init__(name, path)
        self.namespace = namespace

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        module = ModuleType(spec.name)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": self._import}
        return module

    def _import(
        self,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        # Function-local imports use these same module builtins on later ticks.
        if level == 0 and (name == "controllers" or name.startswith("controllers.")):
            name = self.namespace + name.removeprefix("controllers")
        elif level == 0:
            # Preserve bare sibling imports supported by the old file loader.
            sibling = Path(self.path).parent / name.split(".")[0]
            if sibling.with_suffix(".py").is_file() or sibling.is_dir():
                package = self.name if Path(self.path).name == "__init__.py" else self.name.rpartition(".")[0]
                imported = builtins.__import__(f"{package}.{name}", globals, locals, fromlist, level)
                return imported if fromlist else sys.modules[f"{package}.{name.split('.')[0]}"]
        return builtins.__import__(name, globals, locals, fromlist, level)


class _ControllerPackageFinder(MetaPathFinder):
    def __init__(self) -> None:
        self.roots: dict[str, Path] = {}

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        namespace, *parts = fullname.split(".")
        root = self.roots.get(namespace)
        if root is None:
            return None
        candidate = root.joinpath(*parts)
        if candidate.is_dir():
            source = candidate / "__init__.py"
            if not source.is_file():
                spec = ModuleSpec(fullname, loader=None, is_package=True)
                spec.submodule_search_locations = [str(candidate)]
                return spec
        else:
            source = candidate.with_suffix(".py")
        if not source.is_file():
            return None
        return util.spec_from_file_location(
            fullname, source, loader=_ControllerSourceLoader(fullname, str(source), namespace)
        )


_FINDER = _ControllerPackageFinder()


def controller_module_spec(module_path: Path, namespace: str) -> ModuleSpec | None:
    """Resolve controllers/foo.py (also under src/) without changing global imports.

    Absolute controllers.* imports and relative imports share the private package,
    including imports made by helpers, factories, and running controller calls.
    Other imports, including racing, retain their ordinary Python behavior.
    """
    package_root = next((parent for parent in module_path.parents if parent.name == "controllers"), None)
    if package_root is None:
        return None
    _FINDER.roots[namespace] = package_root
    if _FINDER not in sys.meta_path:
        sys.meta_path.insert(0, _FINDER)
    parts = module_path.relative_to(package_root).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    # find_spec imports parent packages, preserving their __init__.py behavior.
    return util.find_spec(".".join((namespace, *parts)))
