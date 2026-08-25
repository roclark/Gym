# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize Harbor's pinned source archive before uv installs requirements."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


HARBOR_COMMIT = "9dddd797b57ab8a0f9d6352a20fce73abbb29573"
HARBOR_SHA256 = "e18b429bb5c0ea7b817ef09b055e96ace0207ade3f4058a198e2c2551ce0925e"
HARBOR_URL = f"https://codeload.github.com/harbor-framework/harbor/tar.gz/{HARBOR_COMMIT}"
ARCHIVE_PATH = Path(__file__).resolve().parent / ".deps" / f"harbor-{HARBOR_COMMIT}.tar.gz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_once(destination: Path) -> None:
    request = urllib.request.Request(
        HARBOR_URL,
        headers={"User-Agent": "nemo-gym-harbor-bootstrap"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def prepare_harbor_archive() -> Path:
    """Download and verify the exact Harbor source used by this integration."""
    if ARCHIVE_PATH.is_file() and _sha256(ARCHIVE_PATH) == HARBOR_SHA256:
        print(f"Using cached Harbor source archive: {ARCHIVE_PATH}")
        return ARCHIVE_PATH

    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()

    last_error: Exception | None = None
    for attempt in range(1, 4):
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=ARCHIVE_PATH.parent,
                prefix=f".{ARCHIVE_PATH.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            print(f"Downloading Harbor {HARBOR_COMMIT} (attempt {attempt}/3)")
            _download_once(temporary_path)
            actual_sha256 = _sha256(temporary_path)
            if actual_sha256 != HARBOR_SHA256:
                raise RuntimeError(
                    "Harbor archive checksum mismatch: "
                    f"expected {HARBOR_SHA256}, got {actual_sha256}"
                )
            os.replace(temporary_path, ARCHIVE_PATH)
            return ARCHIVE_PATH
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(attempt)

    raise RuntimeError(f"Unable to download pinned Harbor source from {HARBOR_URL}") from last_error


if __name__ == "__main__":
    prepare_harbor_archive()
