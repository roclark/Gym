# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch

from responses_api_agents.harbor_agent import prepare_install


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _configure_archive(
    monkeypatch: MonkeyPatch,
    archive_path: Path,
    content: bytes,
) -> None:
    monkeypatch.setattr(prepare_install, "ARCHIVE_PATH", archive_path)
    monkeypatch.setattr(prepare_install, "HARBOR_SHA256", _sha256(content))
    monkeypatch.setattr(prepare_install.time, "sleep", lambda _seconds: None)


def test_reuses_verified_cached_archive(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    content = b"pinned Harbor source"
    archive_path = tmp_path / "deps" / "harbor.tar.gz"
    archive_path.parent.mkdir()
    archive_path.write_bytes(content)
    _configure_archive(monkeypatch, archive_path, content)
    download = MagicMock()
    monkeypatch.setattr(prepare_install, "_download_once", download)

    assert prepare_install.prepare_harbor_archive() == archive_path
    download.assert_not_called()


def test_retries_then_atomically_caches_archive(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    content = b"pinned Harbor source"
    archive_path = tmp_path / "deps" / "harbor.tar.gz"
    _configure_archive(monkeypatch, archive_path, content)
    attempts = 0

    def download(destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise urllib.error.URLError("temporary outage")
        destination.write_bytes(content)

    monkeypatch.setattr(prepare_install, "_download_once", download)

    assert prepare_install.prepare_harbor_archive() == archive_path
    assert archive_path.read_bytes() == content
    assert attempts == 3
    assert list(archive_path.parent.glob("*.tmp")) == []


def test_rejects_archive_with_wrong_checksum(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    expected_content = b"pinned Harbor source"
    archive_path = tmp_path / "deps" / "harbor.tar.gz"
    _configure_archive(monkeypatch, archive_path, expected_content)
    monkeypatch.setattr(
        prepare_install,
        "_download_once",
        lambda destination: destination.write_bytes(b"unexpected source"),
    )

    with pytest.raises(RuntimeError, match="Unable to download pinned Harbor source"):
        prepare_install.prepare_harbor_archive()

    assert not archive_path.exists()
