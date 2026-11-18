"""diskfm: external-memory substring index (disk FM-index).

Public surface:

* :class:`~diskfm.builder.IndexBuilder` -- external-memory construction;
* :class:`~diskfm.query.DiskFMIndex` -- reopen, count, paged locate;
* :class:`~diskfm.format.IndexCorruptError`,
  :class:`~diskfm.format.CursorError`;
* :class:`~diskfm.query.Hit`.
"""

from .builder import BuildStats, IndexBuilder
from .format import CursorError, IndexCorruptError
from .query import DiskFMIndex, Hit

__all__ = [
    "IndexBuilder",
    "BuildStats",
    "DiskFMIndex",
    "Hit",
    "IndexCorruptError",
    "CursorError",
]
