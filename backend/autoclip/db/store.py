"""CRUD operations over the AutoClip database.

Every function opens its own transaction via :func:`autoclip.db.connection`,
so callers don't manage commits. Functions are synchronous; async callers should
wrap them in ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
from typing import Any

from . import connection
from .models import (
    Clip,
    ClipEdit,
    ClipStatus,
    Export,
    Job,
    JobStatus,
    PerformanceSnapshot,
    Posting,
    Source,
    Transcript,
    utcnow,
)

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def create_source(source: Source) -> Source:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO sources (id, type, url, filename, path, title, channel,
                                 duration_s, width, height, fps, has_audio,
                                 has_video, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source.id,
                source.type,
                source.url,
                source.filename,
                source.path,
                source.title,
                source.channel,
                source.duration_s,
                source.width,
                source.height,
                source.fps,
                int(source.has_audio),
                int(source.has_video),
                source.created_at,
            ),
        )
    return source


def get_source(source_id: str) -> Source | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
    return Source.from_row(row) if row else None


def list_sources(limit: int = 50) -> list[Source]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM sources ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [Source.from_row(r) for r in rows]


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


def create_job(job: Job) -> Job:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, source_id, status, current_stage, progress, error,
                              provider, settings_json, created_at, updated_at,
                              started_at, finished_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.source_id,
                job.status,
                job.current_stage,
                job.progress,
                job.error,
                job.provider,
                json.dumps(job.settings),
                job.created_at,
                job.updated_at,
                job.started_at,
                job.finished_at,
            ),
        )
    return job


def get_job(job_id: str) -> Job | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return Job.from_row(row) if row else None


def list_jobs(limit: int = 50, status: JobStatus | None = None) -> list[Job]:
    query = "SELECT * FROM jobs"
    params: list[Any] = []
    if status:
        query += " WHERE status = ?"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [Job.from_row(r) for r in rows]


class _Unset:
    """Sentinel distinguishing "leave this alone" from "set this to NULL".

    Without it, nullable columns can only ever be written, never cleared — a
    retried job would keep displaying the error message from its failed run.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNSET"


UNSET = _Unset()


def update_job(
    job_id: str,
    *,
    status: JobStatus | None = None,
    current_stage: str | None = None,
    progress: float | None = None,
    error: str | None | _Unset = UNSET,
    started_at: str | None | _Unset = UNSET,
    finished_at: str | None | _Unset = UNSET,
) -> None:
    """Patch the supplied fields. Omitted fields are left untouched.

    Nullable fields take :data:`UNSET` as their default, so passing ``None``
    explicitly clears them.

    ``updated_at`` is always refreshed so the UI can detect staleness.
    """
    fields: dict[str, Any] = {"updated_at": utcnow()}
    for name, value in (
        ("status", status),
        ("current_stage", current_stage),
        ("progress", progress),
    ):
        if value is not None:
            fields[name] = value

    for name, nullable in (
        ("error", error),
        ("started_at", started_at),
        ("finished_at", finished_at),
    ):
        if not isinstance(nullable, _Unset):
            fields[name] = nullable

    assignments = ", ".join(f"{name} = ?" for name in fields)
    with connection() as conn:
        conn.execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?",
            (*fields.values(), job_id),
        )


def next_queued_job() -> Job | None:
    """Return the oldest queued job — the FIFO the worker pulls from."""
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
    return Job.from_row(row) if row else None


# --------------------------------------------------------------------------
# Transcripts
# --------------------------------------------------------------------------


def upsert_transcript(transcript: Transcript) -> Transcript:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO transcripts (job_id, json_path, language, model,
                                     has_diarization, word_count, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                json_path = excluded.json_path,
                language = excluded.language,
                model = excluded.model,
                has_diarization = excluded.has_diarization,
                word_count = excluded.word_count,
                source = excluded.source
            """,
            (
                transcript.job_id,
                transcript.json_path,
                transcript.language,
                transcript.model,
                int(transcript.has_diarization),
                transcript.word_count,
                transcript.source,
                transcript.created_at,
            ),
        )
    return transcript


def get_transcript(job_id: str) -> Transcript | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM transcripts WHERE job_id = ?", (job_id,)).fetchone()
    return Transcript.from_row(row) if row else None


# --------------------------------------------------------------------------
# Clips
# --------------------------------------------------------------------------


def replace_clips(job_id: str, clips: list[Clip]) -> list[Clip]:
    """Swap in a fresh set of clips for a job.

    Used when highlight detection reruns — a retry should not leave the previous
    run's candidates behind.
    """
    with connection() as conn:
        conn.execute("DELETE FROM clips WHERE job_id = ?", (job_id,))
        conn.executemany(
            """
            INSERT INTO clips (id, job_id, rank, start_s, end_s, start_word, end_word,
                               title, hook, score, reason, status, user_trimmed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.id,
                    c.job_id,
                    c.rank,
                    c.start_s,
                    c.end_s,
                    c.start_word,
                    c.end_word,
                    c.title,
                    c.hook,
                    c.score,
                    c.reason,
                    c.status,
                    int(c.user_trimmed),
                    c.created_at,
                )
                for c in clips
            ],
        )
    return clips


def get_clip(clip_id: str) -> Clip | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    return Clip.from_row(row) if row else None


def list_clips(job_id: str) -> list[Clip]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM clips WHERE job_id = ? ORDER BY rank ASC", (job_id,)
        ).fetchall()
    return [Clip.from_row(r) for r in rows]


def update_clip(
    clip_id: str,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    start_word: int | None = None,
    end_word: int | None = None,
    title: str | None = None,
    status: ClipStatus | None = None,
    user_trimmed: bool | None = None,
) -> None:
    fields: dict[str, Any] = {}
    for name, value in (
        ("start_s", start_s),
        ("end_s", end_s),
        ("start_word", start_word),
        ("end_word", end_word),
        ("title", title),
        ("status", status),
    ):
        if value is not None:
            fields[name] = value
    if user_trimmed is not None:
        fields["user_trimmed"] = int(user_trimmed)

    if not fields:
        return

    assignments = ", ".join(f"{name} = ?" for name in fields)
    with connection() as conn:
        conn.execute(
            f"UPDATE clips SET {assignments} WHERE id = ?",
            (*fields.values(), clip_id),
        )


# --------------------------------------------------------------------------
# Clip edits
# --------------------------------------------------------------------------


def upsert_clip_edit(edit: ClipEdit) -> ClipEdit:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO clip_edits
                (clip_id, edited_words_json, caption_style, ratio,
                 caption_position, color_grade, caption_color, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_id) DO UPDATE SET
                edited_words_json = excluded.edited_words_json,
                caption_style = excluded.caption_style,
                ratio = excluded.ratio,
                caption_position = excluded.caption_position,
                color_grade = excluded.color_grade,
                caption_color = excluded.caption_color,
                updated_at = excluded.updated_at
            """,
            (
                edit.clip_id,
                json.dumps(edit.edited_words) if edit.edited_words is not None else None,
                edit.caption_style,
                edit.ratio,
                edit.caption_position,
                edit.color_grade,
                edit.caption_color,
                utcnow(),
            ),
        )
    return edit


def get_clip_edit(clip_id: str) -> ClipEdit | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM clip_edits WHERE clip_id = ?", (clip_id,)).fetchone()
    return ClipEdit.from_row(row) if row else None


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------


def create_export(export: Export) -> Export:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO exports (id, clip_id, path, ratio, style, size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                export.id,
                export.clip_id,
                export.path,
                export.ratio,
                export.style,
                export.size_bytes,
                export.created_at,
            ),
        )
    return export


def get_export(export_id: str) -> Export | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM exports WHERE id = ?", (export_id,)).fetchone()
    return Export.from_row(row) if row else None


def list_exports(clip_id: str) -> list[Export]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM exports WHERE clip_id = ? ORDER BY created_at DESC", (clip_id,)
        ).fetchall()
    return [Export.from_row(r) for r in rows]


# --------------------------------------------------------------------------
# Postings — where a clip was published
# --------------------------------------------------------------------------


def create_posting(posting: Posting) -> Posting:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO postings (id, clip_id, platform, url, caption_used, notes,
                                  posted_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                posting.id,
                posting.clip_id,
                posting.platform,
                posting.url,
                posting.caption_used,
                posting.notes,
                posting.posted_at,
                posting.created_at,
            ),
        )
    return posting


def get_posting(posting_id: str) -> Posting | None:
    with connection() as conn:
        row = conn.execute("SELECT * FROM postings WHERE id = ?", (posting_id,)).fetchone()
    return Posting.from_row(row) if row else None


def list_postings(clip_id: str) -> list[Posting]:
    """Postings for one clip, oldest first — the order they were logged."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM postings WHERE clip_id = ? ORDER BY created_at ASC", (clip_id,)
        ).fetchall()
    return [Posting.from_row(r) for r in rows]


def list_all_postings() -> list[Posting]:
    """Every posting, newest first — the tracking report's scan order."""
    with connection() as conn:
        rows = conn.execute("SELECT * FROM postings ORDER BY created_at DESC").fetchall()
    return [Posting.from_row(r) for r in rows]


def update_posting(
    posting_id: str,
    *,
    url: str | None = None,
    caption_used: str | None = None,
    notes: str | None = None,
    posted_at: str | None = None,
) -> None:
    """Patch the editable fields of a posting. Omitted fields are left alone."""
    fields: dict[str, Any] = {}
    for name, value in (
        ("url", url),
        ("caption_used", caption_used),
        ("notes", notes),
        ("posted_at", posted_at),
    ):
        if value is not None:
            fields[name] = value

    if not fields:
        return

    assignments = ", ".join(f"{name} = ?" for name in fields)
    with connection() as conn:
        conn.execute(
            f"UPDATE postings SET {assignments} WHERE id = ?",
            (*fields.values(), posting_id),
        )


def delete_posting(posting_id: str) -> None:
    """Remove a posting and, via the foreign key, its snapshots."""
    with connection() as conn:
        conn.execute("DELETE FROM postings WHERE id = ?", (posting_id,))


# --------------------------------------------------------------------------
# Performance snapshots — observed metrics for a posting
# --------------------------------------------------------------------------


def add_snapshot(snapshot: PerformanceSnapshot) -> PerformanceSnapshot:
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO performance_snapshots
                (id, posting_id, captured_at, views, likes, comments, shares, saves,
                 avg_watch_seconds, retention_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.id,
                snapshot.posting_id,
                snapshot.captured_at,
                snapshot.views,
                snapshot.likes,
                snapshot.comments,
                snapshot.shares,
                snapshot.saves,
                snapshot.avg_watch_seconds,
                snapshot.retention_pct,
            ),
        )
    return snapshot


def get_snapshot(snapshot_id: str) -> PerformanceSnapshot | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT * FROM performance_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
    return PerformanceSnapshot.from_row(row) if row else None


def list_snapshots(posting_id: str) -> list[PerformanceSnapshot]:
    """Snapshots for one posting, oldest first, so growth curves read naturally."""
    with connection() as conn:
        rows = conn.execute(
            "SELECT * FROM performance_snapshots WHERE posting_id = ? ORDER BY captured_at ASC",
            (posting_id,),
        ).fetchall()
    return [PerformanceSnapshot.from_row(r) for r in rows]
