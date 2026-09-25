import { useCallback, useEffect, useState } from 'react'

import {
  api,
  type Platform,
  type Posting,
  type PostingSummary,
  type Snapshot,
} from '../api'
import { ErrorNote } from './ErrorNote'

const PLATFORMS: { key: Platform; label: string }[] = [
  { key: 'tiktok', label: 'TikTok' },
  { key: 'youtube', label: 'YouTube' },
  { key: 'instagram', label: 'Instagram' },
  { key: 'other', label: 'Other' },
]

function formatPct(value: number | null): string {
  return value === null ? '—' : `${(value * 100).toFixed(1)}%`
}

function formatMultiplier(value: number | null): string {
  return value === null ? '—' : `${value.toFixed(2)}×`
}

/**
 * Posted-clip tracking for one clip: log where it went, log what happened,
 * and read the derived numbers back.
 *
 * Manual entry only — this is the loop's data source, not a scraper. Every
 * derived number (checkpoint views, engagement, outperformance) is computed
 * server-side by pipeline.outcomes so the UI can never disagree with the
 * ranking model.
 */
export function PerformancePanel({ clipId }: { clipId: string }) {
  const [postings, setPostings] = useState<PostingSummary[] | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [platform, setPlatform] = useState<Platform>('tiktok')
  const [url, setUrl] = useState('')
  const [caption, setCaption] = useState('')
  const [saving, setSaving] = useState(false)

  const load = useCallback(() => {
    api
      .clipPerformance(clipId)
      .then(setPostings)
      .catch((err) => setError(err as Error))
  }, [clipId])

  useEffect(() => {
    setPostings(null)
    setError(null)
    setShowForm(false)
    load()
  }, [load])

  const logPosting = async () => {
    setSaving(true)
    setError(null)
    try {
      await api.createPosting(clipId, {
        platform,
        url: url || null,
        caption_used: caption || null,
        notes: null,
        posted_at: null,
      })
      setUrl('')
      setCaption('')
      setShowForm(false)
      load()
    } catch (err) {
      setError(err as Error)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div>
      <div className="flex items-baseline justify-between border-b border-ink-800 pb-2">
        <p className="eyebrow">Performance</p>
        <button onClick={() => setShowForm((v) => !v)} className="btn btn-quiet text-xs">
          {showForm ? 'Cancel' : '+ Log posting'}
        </button>
      </div>

      {error && (
        <div className="mt-3">
          <ErrorNote error={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {showForm && (
        <div className="mt-3 space-y-2">
          <select
            value={platform}
            onChange={(e) => setPlatform(e.target.value as Platform)}
            className="field cursor-pointer text-sm"
            aria-label="Platform"
          >
            {PLATFORMS.map(({ key, label }) => (
              <option key={key} value={key} className="bg-ink-850">
                {label}
              </option>
            ))}
          </select>
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="Posted URL (optional)"
            className="field text-sm"
          />
          <input
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            placeholder="Caption used (optional)"
            className="field text-sm"
          />
          <button onClick={logPosting} disabled={saving} className="btn btn-primary w-full">
            {saving ? 'Saving…' : 'Log posting'}
          </button>
        </div>
      )}

      {postings === null ? (
        <p className="mt-3 text-xs text-ink-600">Loading…</p>
      ) : postings.length === 0 ? (
        <p className="mt-3 text-xs leading-relaxed text-ink-500">
          Nothing logged. Log a posting after you upload, then add stats over time — the
          ranking learns from your real results.
        </p>
      ) : (
        <ul className="mt-3 space-y-4">
          {postings.map((posting) => (
            <li key={posting.id}>
              <PostingRow posting={posting} onChanged={load} onError={setError} />
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function PostingRow({
  posting,
  onChanged,
  onError,
}: {
  posting: PostingSummary
  onChanged: () => void
  onError: (err: Error) => void
}) {
  const [showSnapshotForm, setShowSnapshotForm] = useState(false)
  const [views, setViews] = useState('')
  const [likes, setLikes] = useState('')
  const [saving, setSaving] = useState(false)

  const addSnapshot = async () => {
    setSaving(true)
    try {
      await api.addSnapshot(posting.id, {
        views: Number(views) || 0,
        likes: Number(likes) || 0,
        comments: 0,
        shares: 0,
        saves: 0,
        avg_watch_seconds: null,
        retention_pct: null,
      })
      setViews('')
      setLikes('')
      setShowSnapshotForm(false)
      onChanged()
    } catch (err) {
      onError(err as Error)
    } finally {
      setSaving(false)
    }
  }

  const platformLabel =
    PLATFORMS.find((p) => p.key === posting.platform)?.label ?? posting.platform

  return (
    <div className="border-l-2 border-ink-700 pl-3">
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-sm text-ink-100">{platformLabel}</span>
        <button
          onClick={() =>
            api
              .deletePosting(posting.id)
              .then(onChanged)
              .catch(onError)
          }
          className="btn btn-quiet text-xs text-ink-600 hover:text-signal-bad"
          aria-label="Delete posting"
        >
          ✕
        </button>
      </div>
      {posting.url && (
        <a
          href={posting.url}
          target="_blank"
          rel="noreferrer"
          className="mt-0.5 block truncate text-xs text-sodium-500 underline underline-offset-4"
        >
          {posting.url}
        </a>
      )}

      <dl className="numeric mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <dt className="text-ink-500">Views at checkpoint</dt>
        <dd className="text-right text-ink-200">{Math.round(posting.views_at_checkpoint)}</dd>
        <dt className="text-ink-500">Baseline</dt>
        <dd className="text-right text-ink-400">
          {posting.baseline_views === null ? '—' : Math.round(posting.baseline_views)}
        </dd>
        <dt className="text-ink-500">vs baseline</dt>
        <dd
          className={[
            'text-right',
            posting.outperformance !== null && posting.outperformance >= 1.5
              ? 'text-signal-good'
              : posting.outperformance !== null && posting.outperformance <= 0.5
                ? 'text-signal-bad'
                : 'text-ink-200',
          ].join(' ')}
        >
          {formatMultiplier(posting.outperformance)}
        </dd>
        <dt className="text-ink-500">Engagement</dt>
        <dd className="text-right text-ink-200">{formatPct(posting.engagement_rate)}</dd>
        <dt className="text-ink-500">Snapshots</dt>
        <dd className="text-right text-ink-400">{posting.snapshot_count}</dd>
      </dl>

      {showSnapshotForm ? (
        <div className="mt-2 flex items-end gap-2">
          <input
            value={views}
            onChange={(e) => setViews(e.target.value)}
            inputMode="numeric"
            placeholder="Views"
            className="field text-sm"
            aria-label="Views"
          />
          <input
            value={likes}
            onChange={(e) => setLikes(e.target.value)}
            inputMode="numeric"
            placeholder="Likes"
            className="field text-sm"
            aria-label="Likes"
          />
          <button onClick={addSnapshot} disabled={saving} className="btn btn-ghost shrink-0 text-xs">
            {saving ? '…' : 'Save'}
          </button>
        </div>
      ) : (
        <button
          onClick={() => setShowSnapshotForm(true)}
          className="mt-2 text-xs text-ink-400 hover:text-ink-100"
        >
          + Log stats
        </button>
      )}
    </div>
  )
}

// Re-exported for the dashboard page, which also lists snapshots per posting.
export type { Posting, Snapshot }
