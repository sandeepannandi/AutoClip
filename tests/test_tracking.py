"""Tracking API and CLI contract tests, plus the schema v5 migration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from autoclip import db
from autoclip.app import create_app
from autoclip.db import store
from autoclip.db.models import Clip, Job, PerformanceSnapshot, Posting, Source, new_id
from fastapi.testclient import TestClient

NOW = datetime.now(UTC)


def iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AUTOCLIP_NO_WORKER", "1")
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def clip_with_job(initialised_db: int) -> Clip:
    source = store.create_source(
        Source(id=new_id(), type="upload", path="C:/media/v.mp4", title="T")
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id, status="done"))
    clip = Clip(
        id=new_id(),
        job_id=job.id,
        rank=1,
        start_s=0.0,
        end_s=30.0,
        start_word=0,
        end_word=60,
        title="Test clip",
        score=80,
    )
    store.replace_clips(job.id, [clip])
    return clip


class TestMigrationV5:
    def test_postings_table_exists(self, initialised_db: int) -> None:
        with db.connection() as conn:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        names = {r["name"] for r in rows}

        assert {"postings", "performance_snapshots"} <= names

    def test_existing_database_migrates_forward(self, autoclip_home) -> None:
        # An init on top of an existing database stays at the current version
        # (idempotency); forward migration from a genuinely older file is
        # covered by the from-scratch migration in test_db.py.
        first = db.init()
        second = db.init()

        assert first == second == db.SCHEMA_VERSION

    def test_platform_is_constrained(self, initialised_db: int, clip_with_job: Clip) -> None:
        import sqlite3

        with pytest.raises(sqlite3.IntegrityError):
            store.create_posting(Posting(id=new_id(), clip_id=clip_with_job.id, platform="myspace"))

    def test_deleting_a_clip_cascades_to_postings(
        self, initialised_db: int, clip_with_job: Clip
    ) -> None:
        posting = store.create_posting(
            Posting(id=new_id(), clip_id=clip_with_job.id, platform="tiktok")
        )
        store.add_snapshot(PerformanceSnapshot(id=new_id(), posting_id=posting.id, views=100))

        with db.connection() as conn:
            conn.execute("DELETE FROM clips WHERE id = ?", (clip_with_job.id,))

        assert store.get_posting(posting.id) is None
        assert store.list_snapshots(posting.id) == []


class TestPostingEndpoints:
    def test_create_and_list(self, client: TestClient, clip_with_job: Clip) -> None:
        created = client.post(
            f"/api/clips/{clip_with_job.id}/postings",
            json={"platform": "tiktok", "url": "https://tiktok.com/x"},
        )

        assert created.status_code == 201
        assert created.json()["platform"] == "tiktok"

        listed = client.get(f"/api/clips/{clip_with_job.id}/postings").json()
        assert len(listed) == 1
        assert listed[0]["url"] == "https://tiktok.com/x"

    def test_missing_clip_is_404(self, client: TestClient) -> None:
        response = client.post("/api/clips/nope/postings", json={"platform": "tiktok"})

        assert response.status_code == 404

    def test_unknown_platform_is_422(self, client: TestClient, clip_with_job: Clip) -> None:
        response = client.post(
            f"/api/clips/{clip_with_job.id}/postings", json={"platform": "myspace"}
        )

        assert response.status_code == 422

    def test_patch_and_delete(self, client: TestClient, clip_with_job: Clip) -> None:
        posting_id = client.post(
            f"/api/clips/{clip_with_job.id}/postings", json={"platform": "other"}
        ).json()["id"]

        patched = client.patch(f"/api/postings/{posting_id}", json={"notes": "cross-posted"})
        assert patched.status_code == 200
        assert patched.json()["notes"] == "cross-posted"

        deleted = client.delete(f"/api/postings/{posting_id}")
        assert deleted.status_code == 204
        assert client.get(f"/api/clips/{clip_with_job.id}/postings").json() == []


class TestSnapshotEndpoints:
    def test_add_and_list(self, client: TestClient, clip_with_job: Clip) -> None:
        posting_id = client.post(
            f"/api/clips/{clip_with_job.id}/postings", json={"platform": "youtube"}
        ).json()["id"]

        added = client.post(
            f"/api/postings/{posting_id}/snapshots",
            json={"views": 1200, "likes": 90, "retention_pct": 62.5},
        )

        assert added.status_code == 201
        assert added.json()["views"] == 1200

        listed = client.get(f"/api/postings/{posting_id}/snapshots").json()
        assert len(listed) == 1
        assert listed[0]["retention_pct"] == 62.5

    def test_missing_posting_is_404(self, client: TestClient) -> None:
        assert client.post("/api/postings/nope/snapshots", json={"views": 1}).status_code == 404

    def test_negative_views_are_rejected(self, client: TestClient, clip_with_job: Clip) -> None:
        posting_id = client.post(
            f"/api/clips/{clip_with_job.id}/postings", json={"platform": "tiktok"}
        ).json()["id"]

        response = client.post(f"/api/postings/{posting_id}/snapshots", json={"views": -5})

        assert response.status_code == 422


class TestPerformanceEndpoints:
    def test_clip_performance_summary(self, client: TestClient, clip_with_job: Clip) -> None:
        posting_id = client.post(
            f"/api/clips/{clip_with_job.id}/postings",
            json={"platform": "tiktok", "posted_at": iso(NOW - timedelta(hours=80))},
        ).json()["id"]
        # Snapshot taken 72h after the posting, so it covers the 72h checkpoint
        # exactly (posting -80h, capture -8h).
        client.post(
            f"/api/postings/{posting_id}/snapshots",
            json={"views": 1500, "likes": 100, "captured_at": iso(NOW - timedelta(hours=8))},
        )

        body = client.get(f"/api/clips/{clip_with_job.id}/performance").json()

        assert len(body) == 1
        summary = body[0]
        # The snapshot sits at exactly 72h of age, so it covers that checkpoint.
        assert summary["checkpoint_hours"] == 72.0
        assert summary["views_at_checkpoint"] == pytest.approx(1500.0)
        assert summary["engagement_rate"] == pytest.approx(100 / 1500)
        assert summary["snapshot_count"] == 1
        # One platform, one posting: the baseline IS this posting's views.
        assert summary["outperformance"] == pytest.approx(1.0)

    def test_report_baselines_and_signal(self, client: TestClient, clip_with_job: Clip) -> None:
        posting_id = client.post(
            f"/api/clips/{clip_with_job.id}/postings",
            json={"platform": "youtube", "posted_at": iso(NOW - timedelta(hours=80))},
        ).json()["id"]
        # Capture 72h after posting (posting -80h, capture -8h) so the 72h
        # checkpoint is covered and the platform baseline exists.
        client.post(
            f"/api/postings/{posting_id}/snapshots",
            json={"views": 4000, "captured_at": iso(NOW - timedelta(hours=8))},
        )

        body = client.get("/api/tracking/report").json()

        assert body["has_signal"] is True
        assert len(body["postings"]) == 1
        assert body["baselines"][0]["platform"] == "youtube"
        assert body["baselines"][0]["by_checkpoint"]["72"] == pytest.approx(4000.0)

    def test_report_is_empty_without_data(self, client: TestClient) -> None:
        body = client.get("/api/tracking/report").json()

        assert body["postings"] == []
        assert body["baselines"] == []
        assert body["has_signal"] is False


class TestPromptInjection:
    def test_few_shot_block_reaches_the_window_prompt(self) -> None:
        from autoclip.providers import DetectionConfig, TranscriptWindow
        from autoclip.providers.base import render_window_prompt

        window = TranscriptWindow(text="[0]hi", first_word=0, last_word=1)
        config = DetectionConfig(few_shot_block='## Examples\n- Opening: "Hook"')

        prompt = render_window_prompt(window, config)

        assert 'Opening: "Hook"' in prompt
        assert "[0]hi" in prompt
        # The examples sit between the instructions and the transcript.
        assert prompt.index("Examples") < prompt.index("[0]hi")

    def test_empty_block_leaves_the_prompt_unchanged(self) -> None:
        from autoclip.providers import DetectionConfig, TranscriptWindow
        from autoclip.providers.base import render_window_prompt

        window = TranscriptWindow(text="[0]hi", first_word=0, last_word=1)

        with_block = render_window_prompt(window, DetectionConfig(few_shot_block="x"))
        without = render_window_prompt(window, DetectionConfig())

        # The block is inserted between the instructions and the transcript;
        # removing it restores the base prompt exactly.
        assert with_block.replace("x\n\n---", "---") == without


class TestCliTrack:
    """The CLI commands go through Typer's real runner."""

    def test_post_stats_and_list(self, autoclip_home, clip_with_job: Clip) -> None:
        from autoclip.cli import app
        from typer.testing import CliRunner

        runner = CliRunner()

        post = runner.invoke(
            app,
            ["track", "post", clip_with_job.id, "--platform", "tiktok", "--url", "u1"],
        )
        assert post.exit_code == 0, post.output

        posting_id = store.list_postings(clip_with_job.id)[0].id

        stats = runner.invoke(
            app,
            [
                "track",
                "stats",
                posting_id,
                "--views",
                "2000",
                "--likes",
                "150",
                "--at",
                iso(NOW - timedelta(hours=30)),
            ],
        )
        assert stats.exit_code == 0, stats.output

        listing = runner.invoke(app, ["track", "list"])
        assert listing.exit_code == 0, listing.output
        assert "2000" in listing.output

        report = runner.invoke(app, ["track", "report"])
        assert report.exit_code == 0, report.output
        assert "Baselines" in report.output

    def test_post_rejects_unknown_platform(self, autoclip_home, clip_with_job: Clip) -> None:
        from autoclip.cli import app
        from typer.testing import CliRunner

        result = CliRunner().invoke(app, ["track", "post", clip_with_job.id, "--platform", "bebo"])

        assert result.exit_code == 2

    def test_stats_for_missing_posting_fails(self, autoclip_home) -> None:
        from autoclip.cli import app
        from typer.testing import CliRunner

        result = CliRunner().invoke(app, ["track", "stats", "nope", "--views", "10"])

        assert result.exit_code == 1
