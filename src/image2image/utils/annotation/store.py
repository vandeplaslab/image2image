"""Disk-backed annotation objects, spatial queries, and persistent edit history."""

from __future__ import annotations

import json
import sqlite3
import zlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Self, cast
from uuid import uuid4

from shapely import from_wkb, to_wkb
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

from image2image.models.annotation import (
    AnnotationClass,
    AnnotationObject,
    AnnotationProject,
    AnnotationSlide,
    EditCommand,
)

SCHEMA_VERSION = 1
OBJECT_COLUMNS = "id,set_id,object_id,class_id,geometry,properties,provenance,review,training"


def validate_geometry(geometry: BaseGeometry) -> None:
    """Reject empty, invalid, non-finite, or non-polygonal geometry without repairs."""
    import numpy as np
    from shapely import get_coordinates

    if geometry.geom_type not in ("Polygon", "MultiPolygon") or geometry.is_empty or geometry.area <= 0:
        raise ValueError("An object must contain a nonempty Polygon or MultiPolygon.")
    if geometry.has_z or not np.isfinite(get_coordinates(geometry)).all():
        raise ValueError("Annotations require finite two-dimensional XY coordinates.")
    if not geometry.is_valid:
        raise ValueError(f"Invalid geometry: {explain_validity(geometry)}")


def _snapshot(row: sqlite3.Row | None) -> bytes | None:
    if row is None:
        return None
    values = list(row)
    values[4] = bytes(values[4]).hex()
    return zlib.compress(json.dumps(values, ensure_ascii=False).encode())


def _object(row: sqlite3.Row) -> AnnotationObject:
    return AnnotationObject(
        key=row["id"],
        set_id=row["set_id"],
        object_id=json.loads(row["object_id"]),
        class_id=row["class_id"],
        geometry=from_wkb(row["geometry"]),
        properties=json.loads(row["properties"]),
        provenance=json.loads(row["provenance"]),
        review=row["review"],
        training=bool(row["training"]),
    )


class AnnotationStore:
    """One connection per thread/process, with atomic commands and indexed geometry."""

    def __init__(self, path: str | Path, *, create: bool = False) -> None:
        self.path = Path(path).expanduser().resolve()
        if create:
            if self.path.exists() and any(self.path.iterdir()):
                raise FileExistsError(f"Project already exists: {self.path}")
            self.path.mkdir(parents=True, exist_ok=True)
            for name in ("models", "cache", "staging"):
                (self.path / name).mkdir(exist_ok=True)
            (self.path / "manifest.json").write_text(AnnotationProject().model_dump_json())
        self.manifest = AnnotationProject.model_validate_json((self.path / "manifest.json").read_text())
        if not create and not (self.path / "objects.sqlite").is_file():
            raise FileNotFoundError("The project object database is missing.")
        self.db = sqlite3.connect(self.path / "objects.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA cache_size=-16384")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            INSERT OR IGNORE INTO metadata VALUES('revision','0');
            CREATE TABLE IF NOT EXISTS slides(id TEXT PRIMARY KEY,data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS classes(id TEXT PRIMARY KEY,data TEXT NOT NULL,staging_set_id TEXT);
            CREATE TABLE IF NOT EXISTS annotation_sets(
                id TEXT PRIMARY KEY, slide_id TEXT NOT NULL REFERENCES slides(id), name TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'ready', metadata TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE IF NOT EXISTS id_counters(
                set_id TEXT PRIMARY KEY REFERENCES annotation_sets(id) ON DELETE CASCADE,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS objects(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                set_id TEXT NOT NULL REFERENCES annotation_sets(id) ON DELETE CASCADE,
                object_id TEXT NOT NULL, class_id TEXT REFERENCES classes(id), geometry BLOB NOT NULL,
                properties TEXT NOT NULL, provenance TEXT NOT NULL, review TEXT NOT NULL, training INTEGER NOT NULL,
                UNIQUE(set_id,object_id));
            CREATE INDEX IF NOT EXISTS objects_set_review ON objects(set_id,review,id);
            CREATE INDEX IF NOT EXISTS objects_class ON objects(class_id);
            CREATE INDEX IF NOT EXISTS objects_training ON objects(training) WHERE training=1;
            CREATE VIRTUAL TABLE IF NOT EXISTS bounds USING rtree(id,minx,maxx,miny,maxy);
            CREATE TRIGGER IF NOT EXISTS remove_bounds AFTER DELETE ON objects BEGIN
                DELETE FROM bounds WHERE id=old.id;
            END;
            CREATE TABLE IF NOT EXISTS commands(id INTEGER PRIMARY KEY AUTOINCREMENT,label TEXT NOT NULL,undone INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS changes(
                command_id INTEGER REFERENCES commands(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, object_key INTEGER NOT NULL, before BLOB, after BLOB,
                PRIMARY KEY(command_id,ordinal));
            CREATE TABLE IF NOT EXISTS catalog_changes(
                command_id INTEGER REFERENCES commands(id) ON DELETE CASCADE,
                kind TEXT NOT NULL, record_id TEXT NOT NULL, before TEXT, after TEXT);
            """
        )

    def __enter__(self) -> Self:
        """Keep a project connection open for a bounded operation."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the connection at context exit."""
        self.close()

    def close(self) -> None:
        """Release this connection; committed edits remain recoverable."""
        self.db.close()

    @property
    def revision(self) -> int:
        """Return the revision of accepted objects and their training inputs."""
        return int(self.db.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0])

    def _bump(self) -> None:
        self.db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit a complete operation or roll it back, including its spatial index."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def _command(self, label: str) -> int:
        self.db.execute("DELETE FROM commands WHERE undone=1")
        return cast(int, self.db.execute("INSERT INTO commands(label) VALUES(?)", (label,)).lastrowid)

    @contextmanager
    def read_snapshot(self) -> Iterator[None]:
        """Hold a consistent export view while other connections continue editing."""
        self.db.execute("BEGIN")
        try:
            yield
        finally:
            self.db.rollback()

    def _remember_ids(self, highest: dict[str, int]) -> None:
        for set_id, value in highest.items():
            row = self.db.execute("SELECT value FROM id_counters WHERE set_id=?", (set_id,)).fetchone()
            if row is None or value > int(row[0]):
                self.db.execute("INSERT OR REPLACE INTO id_counters VALUES(?,?)", (set_id, str(value)))

    def save_slide(self, slide: AnnotationSlide) -> None:
        """Save calibration or relocate a referenced slide as an undoable edit."""
        self._save_catalog("slides", slide.id, slide.model_dump_json(), f"Update slide {slide.name}")

    def save_class(self, annotation_class: AnnotationClass) -> None:
        """Save a shared class as an undoable edit."""
        self._save_catalog("classes", annotation_class.id, annotation_class.model_dump_json(), "Update class")

    def _save_catalog(self, kind: str, record_id: str, data: str, label: str) -> None:
        if kind not in ("slides", "classes"):
            raise ValueError("Unknown catalog.")
        with self.transaction():
            row = self.db.execute(f"SELECT data FROM {kind} WHERE id=?", (record_id,)).fetchone()  # noqa: S608
            # Adding catalog entries is permanent because objects/sets may reference them.
            # Changes to existing entries are reversible; object creation has its own history.
            self.db.execute(f"INSERT OR REPLACE INTO {kind}(id,data) VALUES(?,?)", (record_id, data))  # noqa: S608
            if row:
                command = self._command(label)
                self.db.execute(
                    "INSERT INTO catalog_changes VALUES(?,?,?,?,?)", (command, kind, record_id, row[0], data)
                )
            else:
                self.db.execute("DELETE FROM commands WHERE undone=1")
            self._bump()

    def slides(self) -> list[AnnotationSlide]:
        """Return the small slide catalog in insertion order."""
        return [
            AnnotationSlide.model_validate_json(row[0])
            for row in self.db.execute("SELECT data FROM slides ORDER BY rowid")
        ]

    def slide(self, slide_id: str) -> AnnotationSlide:
        """Read a slide reference by stable identifier."""
        row = self.db.execute("SELECT data FROM slides WHERE id=?", (slide_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown slide: {slide_id}")
        return AnnotationSlide.model_validate_json(row[0])

    def classes(self) -> list[AnnotationClass]:
        """Return shared project classes."""
        return [
            AnnotationClass.model_validate_json(row[0])
            for row in self.db.execute("SELECT data FROM classes WHERE staging_set_id IS NULL ORDER BY rowid")
        ]

    def stage_class(self, set_id: str, annotation_class: AnnotationClass) -> None:
        """Keep newly imported classes hidden until their complete import is published."""
        with self.transaction():
            self.db.execute(
                "INSERT INTO classes(id,data,staging_set_id) VALUES(?,?,?)",
                (annotation_class.id, annotation_class.model_dump_json(), set_id),
            )

    def resolve_path(self, slide: AnnotationSlide) -> Path:
        """Resolve a relative image reference against the project directory."""
        path = Path(slide.path)
        return path if path.is_absolute() else self.path / path

    def create_set(self, slide_id: str, name: str, *, staging: bool = False, metadata: dict | None = None) -> str:
        """Create a namespace for imported IDs, initially hidden during background work."""
        key = uuid4().hex
        with self.transaction():
            self.db.execute(
                "INSERT INTO annotation_sets VALUES(?,?,?,?,?)",
                (key, slide_id, name, "staging" if staging else "ready", json.dumps(metadata or {})),
            )
        return key

    def sets(self, slide_id: str) -> list[dict]:
        """List published annotation sets for a slide."""
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM annotation_sets WHERE slide_id=? AND state='ready' ORDER BY rowid", (slide_id,)
            )
        ]

    def set_info(self, set_id: str) -> dict:
        """Return namespace metadata, including prediction provenance."""
        row = self.db.execute("SELECT * FROM annotation_sets WHERE id=?", (set_id,)).fetchone()
        if row is None:
            raise KeyError(set_id)
        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])
        return result

    def discard_set(self, set_id: str) -> None:
        """Discard only an unpublished staging set after a failed job."""
        with self.transaction():
            self.db.execute("DELETE FROM annotation_sets WHERE id=? AND state='staging'", (set_id,))
            self.db.execute("DELETE FROM classes WHERE staging_set_id=?", (set_id,))

    def recover(self) -> int:
        """Remove unpublished sets left by interrupted jobs; call before starting workers."""
        with self.transaction():
            count = self.db.execute("SELECT count(*) FROM annotation_sets WHERE state='staging'").fetchone()[0]
            self.db.execute("DELETE FROM annotation_sets WHERE state='staging'")
            self.db.execute("DELETE FROM classes WHERE staging_set_id IS NOT NULL")
        return int(count)

    def _row(self, key: int) -> sqlite3.Row | None:
        return cast(
            sqlite3.Row | None,
            self.db.execute(f"SELECT {OBJECT_COLUMNS} FROM objects WHERE id=?", (key,)).fetchone(),  # noqa: S608 - fixed internal SQL identifiers
        )  # noqa: S608 - fixed columns

    def get(self, key: int) -> AnnotationObject:
        """Fetch complete geometry for a single object."""
        row = self._row(key)
        if row is None:
            raise KeyError(f"Object {key} no longer exists.")
        return _object(row)

    def _write(self, obj: AnnotationObject, key: int | None = None) -> int:
        validate_geometry(obj.geometry)
        if isinstance(obj.object_id, bool) or not isinstance(obj.object_id, (str, int)):
            raise TypeError("Object IDs must be integers or strings.")
        cursor = self.db.execute(
            "INSERT INTO objects(id,set_id,object_id,class_id,geometry,properties,provenance,review,training) "
            "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET set_id=excluded.set_id,"
            "object_id=excluded.object_id,class_id=excluded.class_id,geometry=excluded.geometry,"
            "properties=excluded.properties,provenance=excluded.provenance,review=excluded.review,training=excluded.training",
            (
                key,
                obj.set_id,
                json.dumps(obj.object_id, ensure_ascii=False),
                obj.class_id,
                to_wkb(obj.geometry),
                json.dumps(obj.properties, ensure_ascii=False, allow_nan=False),
                json.dumps(obj.provenance, ensure_ascii=False, allow_nan=False),
                obj.review,
                int(obj.training),
            ),
        )
        key = key if key is not None else cast(int, cursor.lastrowid)
        x0, y0, x1, y1 = obj.geometry.bounds
        self.db.execute("INSERT OR REPLACE INTO bounds VALUES(?,?,?,?,?)", (key, x0, x1, y0, y1))
        return key

    def _require_unlocked(self, obj: AnnotationObject) -> None:
        if obj.properties.get("isLocked") or obj.properties.get("locked"):
            raise ValueError("The selected object is locked.")
        if obj.class_id:
            row = self.db.execute("SELECT data FROM classes WHERE id=?", (obj.class_id,)).fetchone()
            if row and json.loads(row[0]).get("locked"):
                raise ValueError("The selected object's class is locked.")

    def apply(
        self, command: EditCommand, *, expected_revision: int | None = None, allow_unlock: bool = False
    ) -> list[int]:
        """Commit an edit and its inverse atomically, checking prediction revisions if supplied."""
        return self.apply_stream(
            command.label, command.changes, expected_revision=expected_revision, allow_unlock=allow_unlock
        )

    def apply_stream(
        self,
        label: str,
        changes: Iterable[tuple[int | None, AnnotationObject | None]],
        *,
        expected_revision: int | None = None,
        collect_keys: bool = True,
        allow_unlock: bool = False,
    ) -> list[int]:
        """Apply streamed changes without loading the whole project or command into RAM."""
        keys = []
        with self.transaction():
            initial_revision = self.revision
            highest: dict[str, int] = {}
            affected_sets: set[str] = set()
            input_change = False
            if expected_revision is not None and self.revision != expected_revision:
                raise ValueError(
                    "Results are stale because annotation or training inputs changed. Run prediction again."
                )
            command = self._command(label)
            for ordinal, (key, obj) in enumerate(changes):
                before = self._row(key) if key is not None else None
                if key is not None and before is None:
                    raise KeyError(f"Object {key} no longer exists.")
                if before is not None:
                    previous = _object(before)
                    affected_sets.add(previous.set_id)
                    input_change |= previous.review == "accepted"
                    if allow_unlock and obj is not None:
                        a, b = dict(previous.properties), dict(obj.properties)
                        for field in ("locked", "isLocked"):
                            a.pop(field, None)
                            b.pop(field, None)
                        if (
                            a != b
                            or previous.geometry.wkb != obj.geometry.wkb
                            or previous.object_id != obj.object_id
                            or previous.class_id != obj.class_id
                            or previous.set_id != obj.set_id
                            or previous.review != obj.review
                            or previous.training != obj.training
                            or previous.provenance != obj.provenance
                        ):
                            raise ValueError("Unlock commands may change only object lock flags.")
                        self._require_unlocked(replace(previous, properties={}))
                    else:
                        self._require_unlocked(previous)
                if obj is None:
                    if key is None:
                        raise ValueError("Deletion requires an object key.")
                    self.db.execute("DELETE FROM objects WHERE id=?", (key,))
                else:
                    affected_sets.add(obj.set_id)
                    input_change |= obj.review == "accepted"
                    self._require_unlocked(replace(obj, properties={})) if before is None else None
                    key = self._write(obj, key)
                    if isinstance(obj.object_id, int):
                        highest[obj.set_id] = max(highest.get(obj.set_id, 0), obj.object_id)
                self.db.execute(
                    "INSERT INTO changes VALUES(?,?,?,?,?)",
                    (
                        command,
                        ordinal,
                        key,
                        _snapshot(before),
                        _snapshot(self._row(key)),
                    ),
                )
                if collect_keys and obj is not None:
                    keys.append(key)
            self._remember_ids(highest)
            if input_change:
                self._bump()
            self._advance_batch_revisions(affected_sets, initial_revision)
        return keys

    def _advance_batch_revisions(self, set_ids: set[str], previous_revision: int) -> None:
        for set_id in set_ids:
            metadata = self.set_info(set_id)["metadata"]
            request = metadata.get("request")
            if request and metadata.get("review_revision", request["revision"]) == previous_revision:
                metadata["review_revision"] = self.revision
                self.db.execute("UPDATE annotation_sets SET metadata=? WHERE id=?", (json.dumps(metadata), set_id))

    def append_staged(self, set_id: str, objects: Iterable[AnnotationObject]) -> int:
        """Append a bounded batch to a hidden import/prediction set."""
        count = 0
        highest = 0
        with self.transaction():
            if self.set_info(set_id)["state"] != "staging":
                raise ValueError("Only staging sets accept unjournaled objects.")
            for obj in objects:
                if obj.set_id != set_id:
                    raise ValueError("Object belongs to a different annotation set.")
                self._write(obj)
                if isinstance(obj.object_id, int):
                    highest = max(highest, obj.object_id)
                count += 1
            self._remember_ids({set_id: highest})
        return count

    def publish_set(self, set_id: str, *, prediction: bool = False, expected_revision: int | None = None) -> None:
        """Publish a complete staged import or review batch in a single transaction."""
        with self.transaction():
            if self.set_info(set_id)["state"] != "staging":
                raise ValueError("Set has already been published.")
            if expected_revision is not None and self.revision != expected_revision:
                raise ValueError("Inputs changed while this job was running; results were discarded.")
            if not prediction:
                command = self._command("Import annotations")
                for ordinal, row in enumerate(
                    self.db.execute(f"SELECT {OBJECT_COLUMNS} FROM objects WHERE set_id=?", (set_id,))  # noqa: S608 - identifiers are fixed internal SQL; all data is bound
                ):
                    self.db.execute(
                        "INSERT INTO changes VALUES(?,?,?,?,?)", (command, ordinal, row[0], None, _snapshot(row))
                    )
                self._bump()
            self.db.execute("UPDATE annotation_sets SET state='ready' WHERE id=?", (set_id,))
            self.db.execute("UPDATE classes SET staging_set_id=NULL WHERE staging_set_id=?", (set_id,))

    def iter_objects(
        self,
        *,
        slide_id: str | None = None,
        set_id: str | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        review: str | None = "accepted",
        training: bool = False,
        class_id: str | None = None,
        visible_only: bool = False,
        limit: int | None = None,
        after: int = 0,
    ) -> Iterator[AnnotationObject]:
        """Stream a spatially indexed or keyset-paged subset of complete objects."""
        columns = ",".join(f"o.{name}" for name in OBJECT_COLUMNS.split(","))
        source = "bounds b CROSS JOIN objects o ON o.id=b.id" if bounds is not None else "objects o"
        sql = f"SELECT {columns} FROM {source} JOIN annotation_sets s ON s.id=o.set_id"  # noqa: S608
        args: list[Any] = []
        clauses = ["s.state='ready'", "o.id>?"]
        args.append(after)
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            clauses.extend(["b.minx<=?", "b.maxx>=?", "b.miny<=?", "b.maxy>=?"])
            args.extend((x1, x0, y1, y0))
        for column, value in (
            ("s.slide_id", slide_id),
            ("o.set_id", set_id),
            ("o.review", review),
            ("o.class_id", class_id),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                args.append(value)
        if training:
            clauses.append("o.training=1")
        if visible_only:
            clauses.append(
                "(o.class_id IS NULL OR o.class_id NOT IN (SELECT id FROM classes WHERE json_extract(data,'$.visible')=0))"
            )
        sql += " WHERE " + " AND ".join(clauses) + " ORDER BY o.id"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        for row in self.db.execute(sql, args):
            yield _object(row)

    def training_summary(self) -> list[dict[str, Any]]:
        """Count accepted training candidates by slide and class without decoding geometry."""
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT json_extract(sl.data,'$.name') slide_name, "
                "c.id class_id, COALESCE(json_extract(c.data,'$.name'),'Unclassified') class_name, "
                "COALESCE(json_extract(c.data,'$.is_ignore'),0) is_ignore, "
                "COUNT(*) total, SUM(CASE WHEN o.training=1 AND c.id IS NOT NULL "
                "AND COALESCE(json_extract(c.data,'$.is_ignore'),0)=0 THEN 1 ELSE 0 END) included "
                "FROM objects o JOIN annotation_sets s ON s.id=o.set_id "
                "JOIN slides sl ON sl.id=s.slide_id LEFT JOIN classes c ON c.id=o.class_id "
                "WHERE s.state='ready' AND o.review='accepted' "
                "GROUP BY sl.id,o.class_id ORDER BY slide_name,class_name"
            )
        ]

    def count(self, set_id: str, review: str | None = "accepted") -> int:
        """Count one annotation set without materializing its objects."""
        sql, args = "SELECT count(*) FROM objects WHERE set_id=?", [set_id]
        if review is not None:
            sql += " AND review=?"
            args.append(review)
        return int(self.db.execute(sql, args).fetchone()[0])

    def find_id(self, set_id: str, object_id: str | int) -> AnnotationObject | None:
        """Look up an exact typed public ID in its namespace."""
        row = self.db.execute(
            f"SELECT {OBJECT_COLUMNS} FROM objects WHERE set_id=? AND object_id=?",  # noqa: S608 - identifiers are fixed internal SQL; all data is bound
            (
                set_id,
                json.dumps(object_id, ensure_ascii=False),
            ),
        ).fetchone()
        return _object(row) if row else None

    def hit_test(self, slide_id: str, xy: tuple[float, float], *, set_id: str | None = None) -> AnnotationObject | None:
        """Select the smallest visible accepted polygon containing the pixel location."""
        point = Point(xy)
        winner = None
        for obj in self.iter_objects(slide_id=slide_id, set_id=set_id, bounds=(*xy, *xy), visible_only=True):
            if obj.geometry.covers(point) and (winner is None or obj.geometry.area < winner.geometry.area):
                winner = obj
        return winner

    def next_id(self, set_id: str, *, integer: bool = False) -> str | int:
        """Allocate a fresh ID; original IDs are never renumbered."""
        if not integer:
            return uuid4().hex
        # Counters survive deletes and undo, so a retired public ID cannot be reused.
        # Decimal text preserves values beyond SQLite's signed integer range.
        with self.transaction():
            row = self.db.execute("SELECT value FROM id_counters WHERE set_id=?", (set_id,)).fetchone()
            value = int(row[0]) + 1 if row else 1
            self._remember_ids({set_id: value})
        return value

    def undo(self) -> str | None:
        """Undo the latest committed command, including ID and geometry restoration."""
        return self._replay(undo=True)

    def redo(self) -> str | None:
        """Reapply the next undone command."""
        return self._replay(undo=False)

    def history_size(self, *, undo: bool) -> int:
        """Count affected objects to dispatch large history operations off the UI thread."""
        sql = (
            "SELECT id FROM commands WHERE undone=0 ORDER BY id DESC LIMIT 1"
            if undo
            else "SELECT id FROM commands WHERE undone=1 ORDER BY id LIMIT 1"
        )
        row = self.db.execute(sql).fetchone()
        return (
            int(self.db.execute("SELECT count(*) FROM changes WHERE command_id=?", (row[0],)).fetchone()[0])
            if row
            else 0
        )

    def _replay(self, *, undo: bool) -> str | None:
        with self.transaction():
            previous_revision = self.revision
            affected_sets: set[str] = set()
            order, state = ("DESC", 0) if undo else ("ASC", 1)
            row = self.db.execute(
                f"SELECT id,label FROM commands WHERE undone=? ORDER BY id {order} LIMIT 1",  # noqa: S608 - fixed internal SQL identifiers
                (state,),  # noqa: S608 - identifiers are fixed internal SQL; all data is bound
            ).fetchone()  # noqa: S608
            if row is None:
                return None
            command, label = row
            column = "before" if undo else "after"
            for change in self.db.execute(
                f"SELECT object_key,{column} FROM changes WHERE command_id=? ORDER BY ordinal {order}",  # noqa: S608 - fixed internal SQL identifiers
                (command,),  # noqa: S608 - identifiers are fixed internal SQL; all data is bound
            ):  # noqa: S608
                key, snapshot = change
                previous = self._row(key)
                if previous:
                    affected_sets.add(previous["set_id"])
                self.db.execute("DELETE FROM objects WHERE id=?", (key,))
                if snapshot is not None:
                    values = json.loads(zlib.decompress(snapshot))
                    values[4] = bytes.fromhex(values[4])
                    affected_sets.add(values[1])
                    self.db.execute(f"INSERT INTO objects({OBJECT_COLUMNS}) VALUES(?,?,?,?,?,?,?,?,?)", values)
                    x0, y0, x1, y1 = from_wkb(values[4]).bounds
                    self.db.execute("INSERT INTO bounds VALUES(?,?,?,?,?)", (key, x0, x1, y0, y1))
            for item in self.db.execute(
                f"SELECT kind,record_id,{column} FROM catalog_changes WHERE command_id=?",  # noqa: S608 - fixed internal SQL identifiers
                (command,),  # noqa: S608 - identifiers are fixed internal SQL; all data is bound
            ):  # noqa: S608
                kind, record_id, data = item
                if kind not in ("slides", "classes"):
                    raise ValueError("Invalid history catalog.")
                self.db.execute(f"UPDATE {kind} SET data=? WHERE id=?", (data, record_id))  # noqa: S608
            self.db.execute("UPDATE commands SET undone=? WHERE id=?", (int(undo), command))
            self._bump()
            self._advance_batch_revisions(affected_sets, previous_revision)
            return str(label)

    def review(self, set_id: str, *, accept: bool, keys: list[int] | None = None, cancel: Any = None) -> None:
        """Accept/reject selected predictions or an entire batch as one undoable command."""
        info = self.set_info(set_id)
        metadata = info["metadata"]
        if accept and metadata.get("source_fingerprint"):
            from image2image.utils.annotation.images import fingerprint

            if fingerprint(self.resolve_path(self.slide(info["slide_id"]))) != metadata["source_fingerprint"]:
                raise ValueError("Results are stale because the source image changed. Run prediction again.")
            model = metadata.get("model", {})
            if model.get("fingerprint") and fingerprint(Path(model["checkpoint"])) != model["fingerprint"]:
                raise ValueError("Results are stale because the model checkpoint changed. Run prediction again.")
            decoder = model.get("decoder", {})
            if decoder.get("fingerprint") and fingerprint(Path(decoder["checkpoint"])) != decoder["fingerprint"]:
                raise ValueError("Results are stale because the SAM decoder changed. Run prediction again.")
        revision = metadata.get("review_revision", metadata.get("request", {}).get("revision"))
        objects = (
            (self.get(key) for key in keys) if keys is not None else self.iter_objects(set_id=set_id, review="proposed")
        )

        def changes() -> Iterator[tuple[int, AnnotationObject]]:
            for obj in objects:
                if cancel is not None and cancel():
                    raise InterruptedError("Review cancelled; no changes were published.")
                if obj.set_id != set_id or obj.review != "proposed":
                    raise ValueError("Only proposed objects from this batch can be reviewed.")
                yield obj.key, replace(obj, review="accepted" if accept else "rejected")

        self.apply_stream(
            "Accept predictions" if accept else "Reject predictions",
            changes(),
            expected_revision=revision if accept else None,
            collect_keys=False,
        )

    def selection_in(self, slide_id: str, geometry: BaseGeometry, *, set_id: str | None = None) -> Iterator[int]:
        """Stream selected object keys from a drawn selection region."""
        for obj in self.iter_objects(slide_id=slide_id, set_id=set_id, bounds=geometry.bounds, visible_only=True):
            if geometry.intersects(obj.geometry):
                yield obj.key

    def overlap_exists(self, set_id: str) -> bool:
        """Detect positive-area overlaps without holding a complete set in memory."""
        for obj in self.iter_objects(set_id=set_id):
            for other in self.iter_objects(set_id=set_id, bounds=obj.geometry.bounds, after=obj.key):
                if obj.geometry.intersection(other.geometry).area > 1e-9:
                    return True
        return False
