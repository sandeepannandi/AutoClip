import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { api, type TrackingReport } from '../api'
import { ErrorNote } from '../components/ErrorNote'

function formatPct(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`
}

function formatMultiplier(value: number | null): string {
  return value === null ? '—' : `${value.toFixed(2)}×`
}

const CHECKPOINT_LABELS: { key: string; label: string }[] = [
  { key: '24', label: '24h' },
  { key: '72', label: '72h' },
  { key: '168', label: '7d' },
]

/**
 * The account-level view of the feedback loop.
 *
 * Baselines are medians of the user's own logged postings per platform, and
 * outperformance is always relative to them — raw views compare accounts,
 * baselines compare clips.
 */
export function Performance() {
  const [report, setReport] = useState<TrackingReport | null>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    api
      .trackingReport()
      .then(setReport)
      .catch((err) => setError(err as Error))
  }, [])

  return (
    <div className="pt-10">
      <div className="border-b border-ink-800 pb-5">
        <h1 className="font-display text-[clamp(1.5rem,3vw,2.25rem)] leading-tight text-ink-100">
          Performance
        </h1>
        <p className="mt-2 max-w-prose text-sm leading-relaxed text-ink-400">
          What happened to clips after they were posted. Baselines are the median of your
          own postings per platform; the ranking learns from outperformance against them.
          Data comes from what you log per clip — nothing is fetched on its own.
        </p>
      </div>

      {error && (
        <div className="mt-6 max-w-3xl">
          <ErrorNote error={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {!report ? (
        <p className="pt-16 text-sm text-ink-500">Loading…</p>
      ) : report.postings.length === 0 ? (
        <div className="max-w-xl pt-16">
          <h2 className="font-display text-3xl text-ink-200">Nothing tracked yet.</h2>
          <p className="mt-4 text-sm leading-relaxed text-ink-400">
            Open a job's clips, pick one, and use the Performance panel to log where you
            posted it. After a day or two, log views and likes as a new snapshot.
          </p>
          <p className="mt-3 text-sm leading-relaxed text-ink-400">
            The CLI works too: <span className="numeric">autoclip track post</span>, then{' '}
            <span className="numeric">autoclip track stats</span>.
          </p>
        </div>
      ) : (
        <div className="mt-8 space-y-12">
          {report.baselines.length > 0 && (
            <section>
              <p className="eyebrow border-b border-ink-800 pb-2">Baselines per platform</p>
              <table className="mt-4 w-full max-w-xl text-left text-sm">
                <thead>
                  <tr className="text-xs text-ink-500">
                    <th className="pb-2 font-medium">Platform</th>
                    {CHECKPOINT_LABELS.map(({ label }) => (
                      <th key={label} className="numeric pb-2 text-right font-medium">
                        {label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {report.baselines.map((baseline) => (
                    <tr key={baseline.platform} className="border-t border-ink-850">
                      <td className="py-2 text-ink-100">{baseline.platform}</td>
                      {CHECKPOINT_LABELS.map(({ key }) => (
                        <td key={key} className="numeric py-2 text-right text-ink-300">
                          {baseline.by_checkpoint[key] !== undefined
                            ? Math.round(baseline.by_checkpoint[key])
                            : '—'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          <section>
            <p className="eyebrow border-b border-ink-800 pb-2">Postings</p>
            <table className="mt-4 w-full text-left text-sm">
              <thead>
                <tr className="text-xs text-ink-500">
                  <th className="pb-2 font-medium">Clip</th>
                  <th className="pb-2 font-medium">Platform</th>
                  <th className="numeric pb-2 text-right font-medium">Views@ckpt</th>
                  <th className="numeric pb-2 text-right font-medium">Baseline</th>
                  <th className="numeric pb-2 text-right font-medium">vs base</th>
                  <th className="numeric pb-2 text-right font-medium">Engagement</th>
                  <th className="numeric pb-2 text-right font-medium">Snapshots</th>
                </tr>
              </thead>
              <tbody>
                {report.postings.map((posting) => (
                  <tr key={posting.id} className="border-t border-ink-850">
                    <td className="max-w-[16rem] truncate py-2 text-ink-100">
                      <Link
                        to="/"
                        className="hover:text-sodium-500"
                        title={posting.clip_id}
                      >
                        {posting.clip_id}
                      </Link>
                    </td>
                    <td className="py-2 text-ink-300">{posting.platform}</td>
                    <td className="numeric py-2 text-right text-ink-200">
                      {Math.round(posting.views_at_checkpoint)}
                    </td>
                    <td className="numeric py-2 text-right text-ink-400">
                      {posting.baseline_views === null
                        ? '—'
                        : Math.round(posting.baseline_views)}
                    </td>
                    <td
                      className={[
                        'numeric py-2 text-right',
                        posting.outperformance !== null && posting.outperformance >= 1.5
                          ? 'text-signal-good'
                          : posting.outperformance !== null && posting.outperformance <= 0.5
                            ? 'text-signal-bad'
                            : 'text-ink-200',
                      ].join(' ')}
                    >
                      {formatMultiplier(posting.outperformance)}
                    </td>
                    <td className="numeric py-2 text-right text-ink-200">
                      {formatPct(posting.engagement_rate)}
                    </td>
                    <td className="numeric py-2 text-right text-ink-400">
                      {posting.snapshot_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </div>
      )}
    </div>
  )
}
