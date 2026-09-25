"""Performance tracking: log postings, capture metrics, and report outcomes.

Everything here is user-supplied data — manual entry now, an official platform
API later. Nothing is scraped. The derived numbers (engagement, checkpoint
views, outperformance vs the account's own baseline) are computed by
:mod:`autoclip.pipeline.outcomes`, which is also what the ranking loop reads,
so the API and the learning can never disagree about what a number means.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from ..db import store
from ..db.models import PerformanceSnapshot, Posting, new_id, utcnow
from ..pipeline.outcomes import (
    collect_outcomes,
    summary_for_posting,
)
from .schemas import (
    BaselineOut,
    PostingIn,
    PostingOut,
    PostingPatchIn,
    PostingSummaryOut,
    ReportOut,
    SnapshotIn,
    SnapshotOut,
)

router = APIRouter(prefix="/api", tags=["tracking"])


async def _get_clip_or_404(clip_id: str):
    clip = await asyncio.to_thread(store.get_clip, clip_id)
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found.")
    return clip


async def _get_posting_or_404(posting_id: str) -> Posting:
    posting = await asyncio.to_thread(store.get_posting, posting_id)
    if posting is None:
        raise HTTPException(status_code=404, detail="Posting not found.")
    return posting


@router.post("/clips/{clip_id}/postings", response_model=PostingOut, status_code=201)
async def create_posting(clip_id: str, payload: PostingIn) -> PostingOut:
    """Record that this clip was posted to a platform."""
    await _get_clip_or_404(clip_id)

    posting = Posting(
        id=new_id(),
        clip_id=clip_id,
        platform=payload.platform,
        url=payload.url,
        caption_used=payload.caption_used,
        notes=payload.notes,
        posted_at=payload.posted_at,
    )
    await asyncio.to_thread(store.create_posting, posting)
    return PostingOut.of(posting)


@router.get("/clips/{clip_id}/postings", response_model=list[PostingOut])
async def list_postings(clip_id: str) -> list[PostingOut]:
    await _get_clip_or_404(clip_id)
    postings = await asyncio.to_thread(store.list_postings, clip_id)
    return [PostingOut.of(p) for p in postings]


@router.patch("/postings/{posting_id}", response_model=PostingOut)
async def patch_posting(posting_id: str, payload: PostingPatchIn) -> PostingOut:
    await _get_posting_or_404(posting_id)
    await asyncio.to_thread(
        store.update_posting,
        posting_id,
        url=payload.url,
        caption_used=payload.caption_used,
        notes=payload.notes,
        posted_at=payload.posted_at,
    )
    updated = await asyncio.to_thread(store.get_posting, posting_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Posting not found.")
    return PostingOut.of(updated)


@router.delete("/postings/{posting_id}", status_code=204)
async def delete_posting(posting_id: str) -> None:
    await _get_posting_or_404(posting_id)
    await asyncio.to_thread(store.delete_posting, posting_id)


@router.post("/postings/{posting_id}/snapshots", response_model=SnapshotOut, status_code=201)
async def add_snapshot(posting_id: str, payload: SnapshotIn) -> SnapshotOut:
    """Log one observation of this posting's metrics."""
    await _get_posting_or_404(posting_id)

    snapshot = PerformanceSnapshot(
        id=new_id(),
        posting_id=posting_id,
        views=payload.views,
        likes=payload.likes,
        comments=payload.comments,
        shares=payload.shares,
        saves=payload.saves,
        avg_watch_seconds=payload.avg_watch_seconds,
        retention_pct=payload.retention_pct,
        captured_at=payload.captured_at or utcnow(),
    )
    await asyncio.to_thread(store.add_snapshot, snapshot)
    return SnapshotOut.of(snapshot)


@router.get("/postings/{posting_id}/snapshots", response_model=list[SnapshotOut])
async def list_snapshots(posting_id: str) -> list[SnapshotOut]:
    await _get_posting_or_404(posting_id)
    snapshots = await asyncio.to_thread(store.list_snapshots, posting_id)
    return [SnapshotOut.of(s) for s in snapshots]


@router.get("/clips/{clip_id}/performance", response_model=list[PostingSummaryOut])
async def clip_performance(clip_id: str) -> list[PostingSummaryOut]:
    """Every posting of this clip with its derived metrics."""
    await _get_clip_or_404(clip_id)
    postings = await asyncio.to_thread(store.list_postings, clip_id)
    _, baselines = await asyncio.to_thread(collect_outcomes)

    summaries: list[PostingSummaryOut] = []
    for posting in postings:
        snapshots = await asyncio.to_thread(store.list_snapshots, posting.id)
        summary = await asyncio.to_thread(
            summary_for_posting, posting, snapshots, baselines=baselines
        )
        summaries.append(PostingSummaryOut.of_summary(posting, summary))
    return summaries


@router.get("/tracking/report", response_model=ReportOut)
async def tracking_report() -> ReportOut:
    """All postings, the per-platform baselines, and whether there is signal.

    This is the same view the ranking model consumes, so the dashboard shows
    exactly what the learning sees.
    """
    outcomes, baselines = await asyncio.to_thread(collect_outcomes)

    postings_out: list[PostingSummaryOut] = []
    for outcome in outcomes:
        posting = outcome.posting
        snapshots = await asyncio.to_thread(store.list_snapshots, posting.id)
        summary = await asyncio.to_thread(
            summary_for_posting, posting, snapshots, baselines=baselines
        )
        postings_out.append(PostingSummaryOut.of_summary(posting, summary))

    baselines_out = [
        BaselineOut(
            platform=platform,
            by_checkpoint={
                # JSON object keys are strings; hours stay sortable numerically
                # client-side because every value is a whole number.
                str(int(hours) if float(hours).is_integer() else hours): views
                for hours, views in per_checkpoint.items()
            },
        )
        for platform, per_checkpoint in baselines.items()
    ]

    return ReportOut(
        postings=postings_out,
        baselines=baselines_out,
        has_signal=bool(baselines_out) and bool(postings_out),
    )
