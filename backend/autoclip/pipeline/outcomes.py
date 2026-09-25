"""Learning from posted-clip outcomes.

Everything here consumes data the user logged about clips they actually
published — one row in ``postings`` per upload, snapshots of the metrics over
time — and derives three things:

**Metrics** (engagement rate, view velocity) that normalise away account size,
so a 3k-view clip on a 5k-follower account can outrank a 30k-view clip on a
500k-follower account.

**A baseline** per platform: the median views a posting on that platform
achieves by its checkpoint. "Outperformance" is measured against this, not
against zero, because raw views measure the account as much as the clip.

**A ranking model** applied to new clips: empirical multipliers per feature
(caption style, duration band, platform) computed with Bayesian shrinkage so a
hot streak from three postings cannot dominate the ranking. With few samples the
multiplier stays near 1.0 — evidence moves it, never noise alone.

The model is deliberately transparent: every multiplier is a readable number
derived from the user's own data, not a fitted black box. The ranking never
*removes* a clip the model liked; it only reorders the ones that qualified.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..db import store
from ..db.models import Clip, PerformanceSnapshot, Posting

#: Interpolated checkpoints. Posting times differ, so raw "views so far" is not
#: comparable between clips; each posting's growth curve is read at these ages
#: instead, and the freshest checkpoint both clips reached is the comparison
#: point.
CHECKPOINT_HOURS: tuple[float, ...] = (24.0, 72.0, 168.0)

#: Engagement rate floor. A clip with 2 views and 2 likes has a perfect rate on
#: a meaningless denominator; treating it as signal would be noise.
MIN_VIEWS_FOR_ENGAGEMENT = 50

#: Shrinkage strength for empirical multipliers. Higher values pull estimates
#: harder toward 1.0. Each observation counts as 1/k of a direct measurement.
SHRINKAGE_K = 5.0

#: Postings with fewer logged snapshots than this contribute no signal — their
#: outperformance would be pure noise. The tracking settings value, when set
#: above the floor, overrides this at the call sites in this module.
MIN_SNAPSHOTS_FOR_SIGNAL = 1

#: Bounds on any learned multiplier. Whatever the data says, a feature can at
#: most double or halve a candidate's effective score.
MULTIPLIER_MIN = 0.5
MULTIPLIER_MAX = 2.0


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a missing timezone.

    Naive timestamps are read as UTC: everything in the database is written
    with :func:`autoclip.db.models.utcnow`, which is UTC, and manually entered
    values without an offset mean local wall-clock time — assuming UTC is the
    only defensible reading for data that must compare against other rows.
    """
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


# --------------------------------------------------------------------------
# Snapshot arithmetic
# --------------------------------------------------------------------------


def cumulative_metrics(snapshot: PerformanceSnapshot) -> dict[str, float | None]:
    """Engagement rate and retention for one snapshot, each None when undefined.

    Engagement uses every interaction signal the platforms expose in common.
    Rates on tiny view counts are None — see :data:`MIN_VIEWS_FOR_ENGAGEMENT`.
    """
    engagement = (
        (snapshot.likes + snapshot.comments + snapshot.shares + snapshot.saves) / snapshot.views
        if snapshot.views >= MIN_VIEWS_FOR_ENGAGEMENT
        else None
    )
    return {
        "engagement_rate": engagement,
        "retention_pct": snapshot.retention_pct,
    }


def _snapshot_age_hours(snapshot: PerformanceSnapshot, posted_at: datetime) -> float:
    captured = parse_timestamp(snapshot.captured_at)
    return (captured - posted_at).total_seconds() / 3600.0


def _interpolate_at_hours(
    ordered: list[PerformanceSnapshot],
    posted_at: datetime,
    target_hours: float,
) -> float | None:
    """Views read off the posting's growth curve at ``target_hours`` age.

    ``ordered`` must be sorted by ``captured_at``. Between two observations the
    curve is linear. A checkpoint outside the observed window returns None: an
    observation at 30h says nothing honest about hour 24, and inventing a 0
    there would poison both baselines and ranking with fake dead postings.
    """
    if not ordered:
        return None

    ages = [_snapshot_age_hours(s, posted_at) for s in ordered]
    views = [float(s.views) for s in ordered]

    if target_hours < ages[0] or target_hours > ages[-1]:
        return None
    if target_hours == ages[-1]:
        return views[-1]

    for i in range(1, len(ages)):
        if target_hours <= ages[i]:
            span = ages[i] - ages[i - 1]
            fraction = (target_hours - ages[i - 1]) / span
            return views[i - 1] + fraction * (views[i] - views[i - 1])

    return views[-1]  # pragma: no cover - the range checks above cover this


def checkpoint_views(
    snapshots: list[PerformanceSnapshot], posted_at: datetime | None
) -> dict[float, float | None]:
    """Views at each of :data:`CHECKPOINT_HOURS`; None where unobserved.

    A checkpoint is filled only when the posting's observation window covers
    that age — the newest observation must be at or past it, and it must not
    predate the first one. A posting without a recorded time cannot be placed
    on a timeline at all, so every checkpoint is None and it shows up in
    reports with its raw latest views only.
    """
    if posted_at is None:
        return dict.fromkeys(CHECKPOINT_HOURS)

    ordered = sorted(snapshots, key=lambda s: parse_timestamp(s.captured_at))
    return {hours: _interpolate_at_hours(ordered, posted_at, hours) for hours in CHECKPOINT_HOURS}


# --------------------------------------------------------------------------
# Outperformance
# --------------------------------------------------------------------------


@dataclass
class PostingOutcome:
    """What one posting achieved, normalised for fair comparison."""

    posting: Posting
    #: Views at each :data:`CHECKPOINT_HOURS` age; None where the posting's
    #: observation window does not cover the checkpoint.
    views_by_checkpoint: dict[float, float | None]
    #: The newest checkpoint with observed data, or the newest observed age
    #: when the posting is too fresh for any standard checkpoint.
    checkpoint_hours: float
    #: Views at ``checkpoint_hours`` — a real observation, never invented.
    views_at_checkpoint: float
    #: Engagement rate at the newest snapshot with enough views, else None.
    engagement_rate: float | None
    #: Likes+comments+shares+saves at the newest snapshot, raw.
    interactions: int
    #: Median across this posting's snapshots, when retention was reported.
    retention_pct: float | None


def build_posting_outcome(posting: Posting, snapshots: list[PerformanceSnapshot]) -> PostingOutcome:
    """Summarise one posting's logged history into comparable numbers."""
    ordered = sorted(snapshots, key=lambda s: parse_timestamp(s.captured_at))
    posted_at = parse_timestamp(posting.posted_at) if posting.posted_at else None

    by_checkpoint = checkpoint_views(ordered, posted_at)
    observed = {h: v for h, v in by_checkpoint.items() if v is not None}
    if observed:
        checkpoint_hours = max(observed)
    elif posted_at is not None and ordered:
        # Too fresh for any standard checkpoint: report at the newest observed
        # age instead. Baselines are keyed by standard checkpoints, so this
        # posting's views_at_checkpoint is honest while outperformance stays
        # None — it simply is not comparable yet.
        checkpoint_hours = max(_snapshot_age_hours(s, posted_at) for s in ordered)
    else:
        # No posting time: nothing is placeable on a timeline.
        checkpoint_hours = CHECKPOINT_HOURS[0]

    if observed:
        views_at_checkpoint = observed[checkpoint_hours]
    elif ordered:
        views_at_checkpoint = float(ordered[-1].views)
    else:
        views_at_checkpoint = 0.0

    usable = [s for s in ordered if s.views >= MIN_VIEWS_FOR_ENGAGEMENT]
    engagement: float | None = None
    if usable:
        metrics = [cumulative_metrics(s) for s in usable]
        with_rates = [m["engagement_rate"] for m in metrics if m["engagement_rate"] is not None]
        if with_rates:
            # Median, not mean: one snapshot logged mid-spike shouldn't skew.
            engagement = statistics.median(with_rates)

    latest = ordered[-1] if ordered else None

    return PostingOutcome(
        posting=posting,
        views_by_checkpoint=by_checkpoint,
        checkpoint_hours=checkpoint_hours,
        views_at_checkpoint=views_at_checkpoint,
        engagement_rate=engagement,
        interactions=(
            (latest.likes + latest.comments + latest.shares + latest.saves) if latest else 0
        ),
        retention_pct=(
            statistics.median([s.retention_pct for s in ordered if s.retention_pct is not None])
            if any(s.retention_pct is not None for s in ordered)
            else None
        ),
    )


def hours_since(posted_at: datetime | None, snapshots: list[PerformanceSnapshot]) -> float:
    """Age of the newest snapshot relative to posting; infinity when unplaceable.

    A posting with snapshots but no ``posted_at`` still has observations, so it
    gets full checkpoint credit rather than being silently dropped.
    """
    if not snapshots:
        return 0.0
    if posted_at is None:
        return math.inf
    newest = max(parse_timestamp(s.captured_at) for s in snapshots)
    return max(0.0, (newest - posted_at).total_seconds() / 3600.0)


def compute_baselines(outcomes: list[PostingOutcome]) -> dict[str, dict[float, float]]:
    """Median checkpoint views per platform — the account's own normal.

    Returns ``{platform: {checkpoint_hours: median_views}}``. A checkpoint
    enters the median only through postings whose observation window actually
    covered it (``None`` values are skipped), so a batch of snapshots all taken
    at 30h cannot drag the 24h baseline toward zero. Platforms without data are
    absent, and absence means "no baseline": outperformance for those postings
    is None, and they contribute nothing to learning.
    """
    by_platform: dict[str, dict[float, list[float]]] = {}
    for outcome in outcomes:
        platform = outcome.posting.platform
        bucket = by_platform.setdefault(platform, {h: [] for h in CHECKPOINT_HOURS})
        for hours, views in outcome.views_by_checkpoint.items():
            if views is not None:
                bucket[hours].append(views)

    baselines: dict[str, dict[float, float]] = {}
    for platform_name, per_checkpoint in by_platform.items():
        medians = {
            hours: statistics.median(values) for hours, values in per_checkpoint.items() if values
        }
        if medians:
            baselines[platform_name] = medians
    return baselines


def outperformance(outcome: PostingOutcome, baseline: float | None) -> float | None:
    """Ratio of achieved views to the platform baseline, floored at a small positive.

    None when there is no baseline (no comparable postings) or nothing was
    observed. The floor keeps a dead-on-arrival clip a small number rather than
    zero, so it can still shrink a multiplier instead of zeroing it out.
    """
    if baseline is None or baseline <= 0 or outcome.views_at_checkpoint <= 0:
        return None
    return max(outcome.views_at_checkpoint / baseline, 1e-6)


# --------------------------------------------------------------------------
# Feature extraction and the ranking model
# --------------------------------------------------------------------------


def duration_band(duration_s: float) -> str:
    """Coarse duration bucket a clip belongs to."""
    if duration_s < 30:
        return "short"
    if duration_s <= 60:
        return "medium"
    return "long"


@dataclass
class RankingModel:
    """Empirical per-feature multipliers, shrunk toward neutral.

    Learned only from this account's own logged outcomes; with no or thin data
    every multiplier is 1.0 and ranking is exactly the LLM-score order. Each
    feature's estimate is :math:`1 + (r - 1) * n / (n + SHRINKAGE_K)` where
    ``r`` is the mean outperformance of postings carrying that feature and
    ``n`` the number of such postings.
    """

    style_multipliers: dict[str, float] = field(default_factory=dict)
    duration_multipliers: dict[str, float] = field(default_factory=dict)
    platform_multipliers: dict[str, float] = field(default_factory=dict)
    #: Postings that contributed signal.
    sample_count: int = 0

    @property
    def is_trivial(self) -> bool:
        """True when nothing learned would change an ordering."""
        return (
            self.sample_count == 0
            and not self.style_multipliers
            and not self.duration_multipliers
            and not self.platform_multipliers
        )

    def multiplier_for(self, clip: Clip, posting_platform: str = "") -> float:
        """Product of this clip's feature multipliers, clamped to sane bounds."""
        if self.is_trivial:
            return 1.0

        multiplier = 1.0
        if (m := self.style_multipliers.get(clip_style_key(clip))) is not None:
            multiplier *= m
        if (m := self.duration_multipliers.get(duration_band(clip.duration_s))) is not None:
            multiplier *= m
        if posting_platform and (m := self.platform_multipliers.get(posting_platform)) is not None:
            multiplier *= m

        return max(MULTIPLIER_MIN, min(MULTIPLIER_MAX, multiplier))


def clip_style_key(clip: Clip) -> str:
    """The caption style a clip would carry — its override, else the default."""
    edit = store.get_clip_edit(clip.id)
    return edit.caption_style if edit else "bold_pop"


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _shrunk_multiplier(ratios: list[float]) -> float:
    """Shrunk multiplier from observed outperformance ratios."""
    mean_ratio = _mean_or_none(ratios)
    if mean_ratio is None:
        return 1.0
    n = len(ratios)
    estimate = 1.0 + (mean_ratio - 1.0) * n / (n + SHRINKAGE_K)
    return max(MULTIPLIER_MIN, min(MULTIPLIER_MAX, estimate))


def build_ranking_model(
    outcomes: list[PostingOutcome],
    baselines: dict[str, dict[float, float]],
    *,
    min_snapshots: int = MIN_SNAPSHOTS_FOR_SIGNAL,
) -> RankingModel:
    """Learn per-feature multipliers from the account's own outcomes.

    Every outcome with a baseline at its checkpoint contributes to the features
    it carries: the clip's caption style, its duration band, and the platform
    it was posted to. Postings without a comparable baseline are skipped —
    including them would silently mix account-size effects into clip features.
    """
    observations: list[tuple[Clip | None, str, float]] = []
    for outcome in outcomes:
        if len(store.list_snapshots(outcome.posting.id)) < max(1, min_snapshots):
            continue
        baseline = baselines.get(outcome.posting.platform, {}).get(outcome.checkpoint_hours)
        ratio = outperformance(outcome, baseline)
        if ratio is None:
            continue
        clip = store.get_clip(outcome.posting.clip_id)
        observations.append((clip, outcome.posting.platform, ratio))

    model = RankingModel(sample_count=len(observations))
    if not observations:
        return model

    style_ratios: dict[str, list[float]] = {}
    duration_ratios: dict[str, list[float]] = {}
    platform_ratios: dict[str, list[float]] = {}

    for clip, platform, ratio in observations:
        platform_ratios.setdefault(platform, []).append(ratio)
        if clip is None:
            continue
        style_ratios.setdefault(clip_style_key(clip), []).append(ratio)
        duration_ratios.setdefault(duration_band(clip.duration_s), []).append(ratio)

    model.style_multipliers = {k: _shrunk_multiplier(v) for k, v in style_ratios.items()}
    model.duration_multipliers = {k: _shrunk_multiplier(v) for k, v in duration_ratios.items()}
    model.platform_multipliers = {k: _shrunk_multiplier(v) for k, v in platform_ratios.items()}
    return model


# --------------------------------------------------------------------------
# Application to new candidates
# --------------------------------------------------------------------------


def rerank_clips(clips: list[Clip], model: RankingModel) -> list[Clip]:
    """Reorder clips by LLM score times the learned multipliers.

    Sort is stable, so with a trivial model (all multipliers 1.0) the original
    score order comes out untouched. Ranks are reassigned 1..n.
    """
    if model.is_trivial:
        return clips

    ordered = sorted(
        clips,
        key=lambda c: c.score * model.multiplier_for(c),
        reverse=True,
    )
    for rank, clip in enumerate(ordered, start=1):
        clip.rank = rank
    return ordered


# --------------------------------------------------------------------------
# Few-shot examples from the account's own results
# --------------------------------------------------------------------------

#: This many top and bottom performers become prompt examples.
FEW_SHOT_COUNT = 3

#: A posting needs at least this outperformance to serve as a positive example.
GOOD_THRESHOLD = 1.5
#: Below this it serves as a negative example.
POOR_THRESHOLD = 0.5


@dataclass
class FewShotExample:
    """One own-history clip pair shown to the model as calibration."""

    hook: str
    title: str
    outcome_line: str


def build_few_shot_examples(
    outcomes: list[PostingOutcome],
    baselines: dict[str, dict[float, float]],
    *,
    count: int = FEW_SHOT_COUNT,
) -> list[FewShotExample]:
    """Pick the account's best and worst performers as prompt examples.

    Real winners teach the model this audience's taste better than any generic
    advice. Examples are drawn from both ends — top performers *and* flops —
    because knowing what died is as instructive as knowing what flew.
    """
    scored: list[tuple[float, PostingOutcome]] = []
    for outcome in outcomes:
        baseline = baselines.get(outcome.posting.platform, {}).get(outcome.checkpoint_hours)
        ratio = outperformance(outcome, baseline)
        if ratio is None:
            continue
        scored.append((ratio, outcome))

    if not scored:
        return []

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:count]
    bottom = list(reversed(scored[-count:]))

    examples: list[FewShotExample] = []
    seen: set[str] = set()

    def add(ratio: float, outcome: PostingOutcome, verdict: str) -> None:
        clip = store.get_clip(outcome.posting.clip_id)
        if clip is None or clip.id in seen:
            return
        seen.add(clip.id)
        hook = clip.hook.strip() or clip.title.strip()
        if not hook:
            return
        examples.append(
            FewShotExample(
                hook=hook,
                title=clip.title.strip() or hook,
                outcome_line=f"{verdict} ({ratio:.1f}x the account baseline)",
            )
        )

    for ratio, outcome in top:
        if ratio >= GOOD_THRESHOLD:
            add(ratio, outcome, "posted and strongly outperformed")
    for ratio, outcome in bottom:
        if ratio <= POOR_THRESHOLD:
            add(ratio, outcome, "posted and underperformed")

    return examples


def render_few_shot_block(examples: list[FewShotExample]) -> str:
    """Format examples for injection into the detection prompt."""
    if not examples:
        return ""
    lines = ["", "## Examples from this creator's actual results", ""]
    for example in examples:
        lines.append(f'- Opening: "{example.hook}" — {example.outcome_line}.')
    lines.append("")
    lines.append(
        "Weight your scoring toward what has worked for this account, "
        "without copying the examples' topics."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Convenience facade
# --------------------------------------------------------------------------


def collect_outcomes() -> tuple[list[PostingOutcome], dict[str, dict[float, float]]]:
    """Load every logged posting, summarise it, and compute the baselines."""
    outcomes: list[PostingOutcome] = []
    for posting in store.list_all_postings():
        snapshots = store.list_snapshots(posting.id)
        if not snapshots:
            continue
        outcomes.append(build_posting_outcome(posting, snapshots))
    return outcomes, compute_baselines(outcomes)


def ranking_model_from_db(min_snapshots: int = MIN_SNAPSHOTS_FOR_SIGNAL) -> RankingModel:
    """Build the ranking model from everything logged so far."""
    outcomes, baselines = collect_outcomes()
    return build_ranking_model(outcomes, baselines, min_snapshots=min_snapshots)


def few_shot_examples_from_db() -> list[FewShotExample]:
    """Build few-shot examples from everything logged so far."""
    outcomes, baselines = collect_outcomes()
    return build_few_shot_examples(outcomes, baselines)


def summary_for_posting(
    posting: Posting,
    snapshots: list[PerformanceSnapshot],
    *,
    baselines: dict[str, dict[float, float]] | None = None,
) -> dict[str, Any]:
    """Flat dict describing one posting — the API and CLI report shape.

    ``baselines`` should be the account-wide baselines when the caller has them
    (the report endpoint passes :func:`collect_outcomes`' result); computing
    from a single posting alone would make the baseline equal the posting's own
    views and outperformance a meaningless 1.0. Without it, baseline and
    outperformance report as None — honest, not circular.
    """
    outcome = build_posting_outcome(posting, snapshots)
    if baselines is None:
        baseline: float | None = None
    else:
        baseline = baselines.get(posting.platform, {}).get(outcome.checkpoint_hours)
    return {
        "posting_id": posting.id,
        "clip_id": posting.clip_id,
        "platform": posting.platform,
        "url": posting.url,
        "posted_at": posting.posted_at,
        "checkpoint_hours": outcome.checkpoint_hours,
        "views_at_checkpoint": outcome.views_at_checkpoint,
        "engagement_rate": outcome.engagement_rate,
        "interactions": outcome.interactions,
        "retention_pct": outcome.retention_pct,
        "snapshot_count": len(snapshots),
        "baseline_views": baseline,
        "outperformance": outperformance(outcome, baseline),
    }
