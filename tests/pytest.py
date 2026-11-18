"""Minimal pytest-compatible shim (the sandbox has no pytest/pip).

Implements only what this test suite uses:

* ``pytest.mark.parametrize`` collected from function attributes;
* ``pytest.raises`` as a context manager asserting the expected exception;
* plain ``assert`` statements plus ``test_*`` discovery and reporting.

If real pytest is installed it will simply take precedence on sys.path.
"""

from __future__ import annotations

import inspect
import sys
import traceback
from contextlib import contextmanager


class _Mark:
    def parametrize(self, argnames, argvalues):
        names = [a.strip() for a in argnames.split(",")]

        def decorate(fn):
            cases = []
            for values in argvalues:
                if not isinstance(values, tuple):
                    values = (values,)
                cases.append((names, values))
            fn._parametrize_cases = cases
            return fn

        return decorate


mark = _Mark()


@contextmanager
def raises(expected):
    try:
        yield
    except expected:
        return
    except Exception as exc:  # pragma: no cover - test failure path
        raise AssertionError(
            f"expected {expected.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{expected.__name__} was not raised")


def _iter_cases(module):
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        cases = getattr(fn, "_parametrize_cases", None)
        if not cases:
            yield f"{name}()", fn, {}
            continue
        for names, values in cases:
            label = ",".join(f"{n}={v!r}" for n, v in zip(names, values))
            yield f"{name}[{label}]", fn, dict(zip(names, values))


def main(path: str) -> int:
    import os
    test_dir = os.path.dirname(os.path.abspath(path))
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("module_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    passed = failed = 0
    failures = []
    for label, fn, kwargs in _iter_cases(module):
        try:
            fn(**kwargs)
        except Exception:
            failed += 1
            failures.append((label, traceback.format_exc()))
            print(f"FAIL {label}")
        else:
            passed += 1
            print(f"ok   {label}")
    print("-" * 64)
    print(f"{passed} passed, {failed} failed, {passed + failed} total")
    for label, tb in failures:
        print(f"\n=== {label} ===")
        print(tb)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
