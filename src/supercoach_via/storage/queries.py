"""Read-only DuckDB access to a snapshot.

Tables are registered explicitly from the manifest's fragment list (never by globbing a
directory), optionally restricted to named partitions so season-scoped queries read only
the fragments they need. Connections are disposable in-memory query state.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

from supercoach_via.domain.schemas import SnapshotManifest
from supercoach_via.storage.snapshots import contained_path

if TYPE_CHECKING:  # pragma: no cover
    import duckdb
    import pandas as pd
    import pyarrow as pa


def _quote_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe table name {name!r}")
    return f'"{name}"'


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class SnapshotQuery:
    def __init__(
        self,
        data_root: Path,
        manifest: SnapshotManifest,
        *,
        tables: set[str] | None = None,
        partitions: dict[str, set[str]] | None = None,
        threads: int = 4,
    ):
        self.data_root = data_root
        self.manifest = manifest
        self.tables = tables
        self.partitions = partitions or {}
        self.threads = threads
        self.fragments_registered: dict[str, int] = {}
        self._con: duckdb.DuckDBPyConnection | None = None

    def __enter__(self) -> SnapshotQuery:
        import duckdb

        con = duckdb.connect(database=":memory:")
        con.execute(f"SET threads TO {int(self.threads)}")
        for name, entry in sorted(self.manifest.tables.items()):
            if self.tables is not None and name not in self.tables:
                continue
            wanted = self.partitions.get(name)
            frags = [f for f in entry.fragments if wanted is None or f.partition in wanted]
            paths = [str(contained_path(self.data_root / "fragments", f.path)) for f in frags]
            self.fragments_registered[name] = len(paths)
            ident = _quote_ident(name)
            if paths:
                files = "[" + ",".join(_sql_str(p) for p in paths) + "]"
                con.execute(f"CREATE VIEW {ident} AS SELECT * FROM read_parquet({files}, union_by_name=true)")  # noqa: S608 - identifiers validated by _quote_ident, paths contained and quoted
            else:
                # empty selection: keep the table name resolvable with the first fragment's schema
                if entry.fragments:
                    first = _sql_str(str(contained_path(self.data_root / "fragments", entry.fragments[0].path)))
                    con.execute(f"CREATE VIEW {ident} AS SELECT * FROM read_parquet({first}) LIMIT 0")  # noqa: S608 - identifiers validated by _quote_ident, paths contained and quoted
        self._con = con
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            raise RuntimeError("SnapshotQuery used outside its context")
        return self._con

    def scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        row = self.con.execute(sql, params or []).fetchone()
        return None if row is None else row[0]

    def rows(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        return self.con.execute(sql, params or []).fetchall()

    def df(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def arrow(self, sql: str, params: list[Any] | None = None) -> pa.Table:
        return self.con.execute(sql, params or []).fetch_arrow_table()
