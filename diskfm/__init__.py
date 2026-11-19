"""diskfm: external-memory substring index (disk FM-index).

Public surface:

* :class:`~diskfm.builder.IndexBuilder` -- external-memory construction;
* :class:`~diskfm.query.DiskFMIndex` -- reopen, count, paged locate;
* :class:`~diskfm.simbuilder.SimilarityBuilder` -- document-level
  near-duplicate similarity catalog construction (qgram postings);
* :class:`~diskfm.simquery.SimilarityCatalog` -- reopen, paged
  near-duplicate query, shared-shingle explain, deep validation;
* :class:`~diskfm.format.IndexCorruptError`,
  :class:`~diskfm.format.CursorError`;
* :class:`~diskfm.query.Hit`.
"""

from .builder import BuildStats, IndexBuilder
from .cooccur import (CooccurAnalysis, CooccurError, CooccurRequest, Pattern,
                      analyse, build_request)
from .format import CursorError, IndexCorruptError
from .pubbuilder import PublishStats, ReleaseBuilder
from .pubformat import PublishRejected, ReleaseCorruptError
from .pubquery import ReleasePackage
from .query import DiskFMIndex, Hit
from .simbuilder import SimilarityBuilder, SimBuildStats
from .simquery import SimilarityCatalog

__all__ = [
    "IndexBuilder",
    "BuildStats",
    "DiskFMIndex",
    "Hit",
    "IndexCorruptError",
    "CursorError",
    "SimilarityBuilder",
    "SimBuildStats",
    "SimilarityCatalog",
    "CooccurAnalysis",
    "CooccurError",
    "CooccurRequest",
    "Pattern",
    "analyse",
    "build_request",
    "ReleaseBuilder",
    "ReleasePackage",
    "PublishStats",
    "PublishRejected",
    "ReleaseCorruptError",
]
