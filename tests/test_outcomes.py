"""Outcome learning: checkpoint math, baselines, shrinkage ranking, few-shots.

Every number the ranking model consumes flows through this module, so the
tests pin the arithmetic exactly — a silent change to the shrinkage or the
checkpoint interpolation would reorder users' clips without anyone noticing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from autoclip.db import store
from autoclip.db.models import Clip, Job, PerformanceSnapshot, Posting, Source, new_id
from autoclip.pipeline import outcomes
from autoclip.pipeline.outcomes import (
    CHECKPOINT_HOURS,
    FewShotExample,
    RankingModel,
    build_few_shot_examples,
    build_posting_outcome,
    build_ranking_model,
    checkpoint_views,
    compute_baselines,
    duration_band,
    outperformance,
    parse_timestamp,
    rerank_clips,
)

NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)


def iso(dt: datetime) -> str:
    return dt.isoformat()


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def job(initialised_db: int) -> Job:
    return fresh_job()


def fresh_job() -> Job:
    """A new source + job, so helpers never share one job's clip set.

    ``store.replace_clips`` swaps the whole clip set for a job, so two clips
    created against the same job would cascade-delete each other's postings.
    Every helper therefore works against its own job.
    """
    source = store.create_source(
        Source(id=new_id(), type="upload", path="C:/media/v.mp4", title="T")
    )
    return store.create_job(Job(id=new_id(), source_id=source.id))


def make_clip(job: Job, *, score: int = 80, style: str | None = None) -> Clip:
    clip = Clip(
        id=new_id(),
        job_id=job.id,
        rank=1,
        start_s=0.0,
        end_s=30.0,
        start_word=0,
        end_word=60,
        title="t",
        hook="h",
        score=score,
    )
    store.replace_clips(job.id, [clip])
    if style is not None:
        from autoclip.db.models import ClipEdit

        store.upsert_clip_edit(ClipEdit(clip_id=clip.id, caption_style=style))
    return clip


def make_posting(
    clip: Clip,
    *,
    platform: str = "tiktok",
    posted_at: datetime | None = NOW,
) -> Posting:
    posting = Posting(
        id=new_id(),
        clip_id=clip.id,
        platform=platform,  # type: ignore[arg-type]
        posted_at=iso(posted_at) if posted_at else None,
    )
    return store.create_posting(posting)


def posting_for(
    *,
    platform: str = "tiktok",
    posted_at: datetime | None = NOW,
    style: str | None = None,
) -> Posting:
    """A posting on a fresh clip (optionally with a caption style) in its own job."""
    return make_posting(make_clip(fresh_job(), style=style), platform=platform, posted_at=posted_at)


def snapshot(
    posting: Posting,
    views: int,
    *,
    at: datetime,
    likes: int = 0,
    retention_pct: float | None = None,
) -> PerformanceSnapshot:
    return store.add_snapshot(
        PerformanceSnapshot(
            id=new_id(),
            posting_id=posting.id,
            views=views,
            likes=likes,
            captured_at=iso(at),
            retention_pct=retention_pct,
        )
    )


# --------------------------------------------------------------------------
# Timestamp parsing
# --------------------------------------------------------------------------


class TestParseTimestamp:
    def test_iso_with_offset(self) -> None:
        parsed = parse_timestamp("2026-09-20T12:00:00+00:00")

        assert parsed.utcoffset() == timedelta(0)
        assert parsed.year == 2026

    def test_trailing_z(self) -> None:
        parsed = parse_timestamp("2026-09-20T12:00:00Z")

        assert parsed.utcoffset() == timedelta(0)

    def test_naive_is_read_as_utc(self) -> None:
        parsed = parse_timestamp("2026-09-20 12:00:00")

        assert parsed.utcoffset() == timedelta(0)


# --------------------------------------------------------------------------
# Checkpoint interpolation
# --------------------------------------------------------------------------


class TestCheckpointViews:
    def test_exact_snapshot_lands_on_its_value(self, job: Job) -> None:
        posting = posting_for()
        # A snapshot taken exactly 24h after posting.
        snapshot(posting, 1000, at=NOW + timedelta(hours=24))

        by_checkpoint = checkpoint_views(store.list_snapshots(posting.id), NOW)

        assert by_checkpoint[24.0] == pytest.approx(1000.0)

    def test_between_snapshots_interpolates_linearly(self, job: Job) -> None:
        posting = posting_for()
        snapshot(posting, 500, at=NOW + timedelta(hours=2))
        snapshot(posting, 900, at=NOW + timedelta(hours=30))

        by_checkpoint = checkpoint_views(store.list_snapshots(posting.id), NOW)

        # 24h sits between the 2h and 30h observations: linear interpolation.
        expected = 500 + (24 - 2) / (30 - 2) * (900 - 500)
        assert by_checkpoint[24.0] == pytest.approx(expected)
        # 72h and 168h are past the last observation: unobserved, not zero.
        assert by_checkpoint[72.0] is None
        assert by_checkpoint[168.0] is None

    def test_checkpoint_before_the_first_snapshot_is_none(self, job: Job) -> None:
        posting = posting_for()
        snapshot(posting, 500, at=NOW + timedelta(hours=30))

        by_checkpoint = checkpoint_views(store.list_snapshots(posting.id), NOW)

        # A single observation at 30h brackets no standard checkpoint: 24h is
        # before it (unknown — inventing a 0 would poison baselines) and 72h
        # is after it (unknowable).
        assert by_checkpoint[24.0] is None
        assert by_checkpoint[72.0] is None

    def test_checkpoint_covered_by_a_single_exact_observation(self, job: Job) -> None:
        posting = posting_for()
        snapshot(posting, 500, at=NOW + timedelta(hours=72))

        by_checkpoint = checkpoint_views(store.list_snapshots(posting.id), NOW)

        # An observation exactly at a checkpoint covers it — and only it.
        assert by_checkpoint[24.0] is None
        assert by_checkpoint[72.0] == pytest.approx(500.0)
        assert by_checkpoint[168.0] is None

    def test_no_posted_time_is_none_everywhere(self, job: Job) -> None:
        posting = posting_for(posted_at=None)
        snapshot(posting, 777, at=NOW + timedelta(hours=30))

        by_checkpoint = checkpoint_views(store.list_snapshots(posting.id), None)

        assert by_checkpoint == dict.fromkeys(CHECKPOINT_HOURS)

    def test_no_snapshots_is_none_everywhere(self, job: Job) -> None:
        by_checkpoint = checkpoint_views([], NOW)

        assert by_checkpoint == dict.fromkeys(CHECKPOINT_HOURS)


class TestBuildPostingOutcome:
    def test_checkpoint_is_the_newest_covered(self, job: Job) -> None:
        posting = posting_for()
        snapshot(posting, 100, at=NOW + timedelta(hours=30))
        snapshot(posting, 400, at=NOW + timedelta(hours=80))

        outcome = build_posting_outcome(posting, store.list_snapshots(posting.id))

        # The 72h checkpoint is covered by the 80h observation (interpolated
        # backwards); 168h is not covered yet.
        assert outcome.checkpoint_hours == 72.0
        assert outcome.views_at_checkpoint == pytest.approx(
            100 + (72 - 30) / (80 - 30) * (400 - 100)
        )

    def test_fresh_posting_uses_its_observed_age(self, job: Job) -> None:
        posting = posting_for()
        snapshot(posting, 50, at=NOW + timedelta(hours=5))

        outcome = build_posting_outcome(posting, store.list_snapshots(posting.id))

        # Too new for the 24h checkpoint: compared at 5h, which no baseline
        # will match, so outperformance stays None until it matures.
        assert outcome.checkpoint_hours == 5.0
        assert outcome.views_at_checkpoint == 50.0

    def test_engagement_needs_a_minimum_audience(self, job: Job) -> None:
        posting = posting_for()
        snapshot(posting, 10, at=NOW + timedelta(hours=30), likes=5)

        outcome = build_posting_outcome(posting, store.list_snapshots(posting.id))

        # 10 views is far below the engagement floor — the rate is noise.
        assert outcome.engagement_rate is None

    def test_engagement_is_the_median_of_usable_snapshots(self, job: Job) -> None:
        posting = posting_for()
        snapshot(posting, 1000, at=NOW + timedelta(hours=30), likes=100)
        snapshot(posting, 2000, at=NOW + timedelta(hours=80), likes=40)

        outcome = build_posting_outcome(posting, store.list_snapshots(posting.id))

        # Rates are 0.1 and 0.02 -> median 0.06.
        assert outcome.engagement_rate == pytest.approx(0.06)


# --------------------------------------------------------------------------
# Baselines and outperformance
# --------------------------------------------------------------------------


class TestBaselines:
    def test_median_per_platform(self, job: Job) -> None:
        posting_a = posting_for()
        posting_b = posting_for()
        # Observations at exactly 72h cover that checkpoint on both postings.
        snapshot(posting_a, 1000, at=NOW + timedelta(hours=72))
        snapshot(posting_b, 3000, at=NOW + timedelta(hours=72))

        posting_c = posting_for(platform="youtube")
        snapshot(posting_c, 9000, at=NOW + timedelta(hours=72))

        outcomes_list = [
            build_posting_outcome(p, store.list_snapshots(p.id))
            for p in (posting_a, posting_b, posting_c)
        ]
        baselines = compute_baselines(outcomes_list)

        assert baselines["tiktok"] == {72.0: 2000.0}
        assert baselines["youtube"] == {72.0: 9000.0}

    def test_median_uses_only_covered_checkpoints(self, job: Job) -> None:
        posting_a = posting_for()
        posting_b = posting_for()
        # Observations at exactly 24h and 168h cover every checkpoint: 24h
        # exactly, 72h by interpolation, 168h exactly.
        snapshot(posting_a, 1000, at=NOW + timedelta(hours=24))
        snapshot(posting_a, 1000, at=NOW + timedelta(hours=168))
        snapshot(posting_b, 3000, at=NOW + timedelta(hours=24))
        snapshot(posting_b, 3000, at=NOW + timedelta(hours=168))

        outcomes_list = [
            build_posting_outcome(p, store.list_snapshots(p.id)) for p in (posting_a, posting_b)
        ]
        baselines = compute_baselines(outcomes_list)

        assert baselines["tiktok"][24.0] == 2000.0
        assert baselines["tiktok"][72.0] == 2000.0
        assert baselines["tiktok"][168.0] == 2000.0

    def test_outperformance_against_baseline(self, job: Job) -> None:
        posting = posting_for()
        snapshot(posting, 1000, at=NOW + timedelta(hours=30))
        outcome = build_posting_outcome(posting, store.list_snapshots(posting.id))

        # The posting's newest observed age is 30h, so its comparison point is
        # 30h — outperformance is None against any standard-checkpoint baseline
        # (it is not comparable yet), and works only when the baseline matches.
        fresh_baseline = {posting.platform: {30.0: 500.0}}
        ratio = outperformance(outcome, fresh_baseline[posting.platform][30.0])
        assert ratio == pytest.approx(2.0)
        assert outperformance(outcome, 0.0) is None
        assert outperformance(outcome, None) is None


def test_duration_band_boundaries() -> None:
    assert duration_band(10.0) == "short"
    assert duration_band(30.0) == "medium"
    assert duration_band(60.0) == "medium"
    assert duration_band(61.0) == "long"


# --------------------------------------------------------------------------
# The ranking model
# --------------------------------------------------------------------------


class TestRankingModel:
    def test_no_data_is_trivial(self) -> None:
        model = build_ranking_model([], {})

        assert model.is_trivial
        assert model.multiplier_for(Clip(id="x", job_id="j", start_s=0, end_s=30)) == 1.0

    def test_single_style_learns_a_shrunk_multiplier(self, job: Job) -> None:
        posting = posting_for(style="bold_pop")
        # 2x the (unknown-to-him) baseline of 1000, observed at the 72h
        # checkpoint so the comparison point exists.
        snapshot(posting, 2000, at=NOW + timedelta(hours=72))

        baseline = {posting.platform: {72.0: 1000.0}}
        model = build_ranking_model(
            [build_posting_outcome(posting, store.list_snapshots(posting.id))], baseline
        )

        assert model.sample_count == 1
        assert not model.is_trivial
        # Shrinkage: 1 + (2-1) * 1/(1+5) = 1.1667.
        assert model.style_multipliers["bold_pop"] == pytest.approx(1 + 1 / 6)

    def test_multiplier_clamps_to_bounds(self, job: Job) -> None:
        posting = posting_for(style="boxed")
        snapshot(posting, 1_000_000, at=NOW + timedelta(hours=72))

        baseline = {posting.platform: {72.0: 100.0}}
        model = build_ranking_model(
            [build_posting_outcome(posting, store.list_snapshots(posting.id))], baseline
        )

        # Even a 10000x outlier cannot exceed the clamp.
        assert model.style_multipliers["boxed"] == outcomes.MULTIPLIER_MAX

    def test_below_minimum_snapshots_contributes_nothing(self, job: Job) -> None:
        posting = posting_for(style="bold_pop")
        snapshot(posting, 2000, at=NOW + timedelta(hours=72))

        baseline = {posting.platform: {72.0: 1000.0}}
        model = build_ranking_model(
            [build_posting_outcome(posting, store.list_snapshots(posting.id))],
            baseline,
            min_snapshots=2,
        )

        assert model.is_trivial

    def test_missing_baseline_is_skipped(self, job: Job) -> None:
        posting = posting_for()
        snapshot(posting, 5000, at=NOW + timedelta(hours=30))

        model = build_ranking_model(
            [build_posting_outcome(posting, store.list_snapshots(posting.id))], {}
        )

        assert model.is_trivial

    def test_learned_style_reorders_candidates(self, job: Job) -> None:
        low_score_bold = make_clip(job, score=70, style="bold_pop")
        high_score_clean = make_clip(job, score=90, style="clean_lower")

        model = RankingModel(style_multipliers={"bold_pop": 1.5, "clean_lower": 0.9})

        reranked = rerank_clips([high_score_clean, low_score_bold], model)

        # 70 * 1.5 = 105 beats 90 * 0.9 = 81.
        assert [c.id for c in reranked] == [low_score_bold.id, high_score_clean.id]
        assert reranked[0].rank == 1
        assert reranked[1].rank == 2

    def test_trivial_model_keeps_order_exactly(self, job: Job) -> None:
        a = make_clip(job, score=50)
        b = make_clip(job, score=90)

        reranked = rerank_clips([a, b], RankingModel())

        assert reranked == [a, b]
        assert a.rank == 1  # untouched — no reassignment on the no-op path


# --------------------------------------------------------------------------
# Few-shot examples
# --------------------------------------------------------------------------


class TestFewShotExamples:
    def test_best_and_worst_are_selected(self, job: Job) -> None:
        winner = make_clip(fresh_job(), style="bold_pop")
        winner_posting = make_posting(winner)
        snapshot(winner_posting, 3000, at=NOW + timedelta(hours=72))

        loser = make_clip(fresh_job(), style="clean_lower")
        loser_posting = make_posting(loser)
        snapshot(loser_posting, 200, at=NOW + timedelta(hours=72))

        baseline = {winner_posting.platform: {72.0: 1000.0}}
        outcomes_list = [
            build_posting_outcome(p, store.list_snapshots(p.id))
            for p in (winner_posting, loser_posting)
        ]

        examples = build_few_shot_examples(outcomes_list, baseline)

        hooks = {e.hook for e in examples}
        assert winner.hook in hooks
        assert loser.hook in hooks
        verdicts = {e.outcome_line for e in examples}
        assert any("outperformed" in v for v in verdicts)
        assert any("underperformed" in v for v in verdicts)

    def test_middle_performers_make_no_example(self, job: Job) -> None:
        posting = posting_for()
        snapshot(posting, 1000, at=NOW + timedelta(hours=72))  # exactly baseline

        examples = build_few_shot_examples(
            [build_posting_outcome(posting, store.list_snapshots(posting.id))],
            {posting.platform: {72.0: 1000.0}},
        )

        assert examples == []

    def test_no_examples_without_baseline(self, job: Job) -> None:
        posting = posting_for()
        snapshot(posting, 5000, at=NOW + timedelta(hours=30))

        examples = build_few_shot_examples(
            [build_posting_outcome(posting, store.list_snapshots(posting.id))], {}
        )

        assert examples == []


def test_few_shot_block_renders_examples() -> None:
    from autoclip.pipeline.outcomes import render_few_shot_block

    block = render_few_shot_block(
        [FewShotExample(hook="Nobody tells you this", title="t", outcome_line="3.0x baseline")]
    )

    assert 'Opening: "Nobody tells you this"' in block
    assert "3.0x baseline" in block


def test_few_shot_block_empty_without_examples() -> None:
    from autoclip.pipeline.outcomes import render_few_shot_block

    assert render_few_shot_block([]) == ""
