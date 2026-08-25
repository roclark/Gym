# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest
import yaml


TASK_SERVER_PATH = Path(__file__).parents[1] / "task_template" / "environment" / "task-server" / "task_server.py"
COMPOSE_OVERRIDE_PATH = Path(__file__).parents[1] / "docker-compose.external-arvo.yaml"


def _unused_port() -> int:
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return server_socket.getsockname()[1]


def _start_service(
    tmp_path: Path,
    *,
    name: str,
    role: str,
    runner_exit_code: int,
    port: int,
    runner_token: str,
    task_token: str = "",
    task_token_file: Path | None = None,
    fixed_runner_url: str = "",
) -> tuple[subprocess.Popen, Path]:
    service_dir = tmp_path / name
    service_dir.mkdir()
    poc_path = service_dir / "poc"
    poc_path.write_bytes((name + "-ground-truth").encode())
    runner_path = service_dir / "runner.sh"
    runner_path.write_text(f"#!/bin/sh\nexit {runner_exit_code}\n")
    runner_path.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "AUTH_TOKEN": task_token,
            "AUTH_TOKEN_FILE": str(task_token_file or ""),
            "CYBERGYM_ARVO_POC_PATH": str(poc_path),
            "CYBERGYM_ARVO_RUNNER_CWD": str(service_dir),
            "CYBERGYM_ARVO_RUNNER_PATH": str(runner_path),
            "CYBERGYM_ARVO_SERVICE_ROLE": role,
            "FIXED_RUNNER_URL": fixed_runner_url,
            "MAX_OUTPUT_BYTES": "4096",
            "MAX_POC_BYTES": "4096",
            "POC_TIMEOUT_SEC": "2",
            "PORT": str(port),
            "RUNNER_AUTH_TOKEN": runner_token,
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-u", str(TASK_SERVER_PATH)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, poc_path


def _wait_for_health(url: str, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output, _ = process.communicate()
            raise AssertionError(f"external ARVO service exited early: {output}")
        try:
            with urlopen(url + "/health", timeout=0.2) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            time.sleep(0.05)
    raise AssertionError(f"external ARVO service did not become healthy: {url}")


def _stop_service(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover
        process.kill()
        process.wait(timeout=5)


def test_raw_arvo_services_preserve_task_server_contract(tmp_path: Path) -> None:
    runner_token = "runner-token-for-tests"
    task_token = "task-token-for-tests"
    fixed_port = _unused_port()
    vulnerable_port = _unused_port()
    fixed_url = f"http://127.0.0.1:{fixed_port}"
    vulnerable_url = f"http://127.0.0.1:{vulnerable_port}"
    fixed_process, fixed_poc = _start_service(
        tmp_path,
        name="fixed",
        role="fixed",
        runner_exit_code=0,
        port=fixed_port,
        runner_token=runner_token,
    )
    vulnerable_process = None
    task_token_file = tmp_path / "task-auth.json"
    task_token_file.write_text(json.dumps({"auth_token": task_token}))
    try:
        _wait_for_health(fixed_url, fixed_process)
        vulnerable_process, vulnerable_poc = _start_service(
            tmp_path,
            name="vulnerable",
            role="vulnerable",
            runner_exit_code=134,
            port=vulnerable_port,
            runner_token=runner_token,
            task_token_file=task_token_file,
            fixed_runner_url=fixed_url,
        )
        _wait_for_health(vulnerable_url, vulnerable_process)

        with urlopen(Request(vulnerable_url + "/submit", data=b"candidate", method="POST"), timeout=2) as response:
            assert json.load(response)["exit_code"] == 134

        with pytest.raises(HTTPError) as unauthorized:
            urlopen(Request(vulnerable_url + "/verify", data=b"candidate", method="POST"), timeout=2)
        assert unauthorized.value.code == 401

        verify_request = Request(
            vulnerable_url + "/verify",
            data=b"candidate",
            headers={"Authorization": f"Bearer {task_token}"},
            method="POST",
        )
        with urlopen(verify_request, timeout=2) as response:
            assert json.load(response) == {"vul_exit_code": 134, "fix_exit_code": 0}

        solve_request = Request(
            vulnerable_url + "/solve",
            headers={"Authorization": f"Bearer {task_token}"},
        )
        with urlopen(solve_request, timeout=2) as response:
            assert response.read() == b"vulnerable-ground-truth"

        assert vulnerable_poc.read_bytes() == b"vulnerable-ground-truth"
        assert fixed_poc.read_bytes() == b"fixed-ground-truth"
    finally:
        if vulnerable_process is not None:
            _stop_service(vulnerable_process)
        _stop_service(fixed_process)


def test_external_arvo_compose_override_matches_lab_services() -> None:
    compose = yaml.safe_load(COMPOSE_OVERRIDE_PATH.read_text())
    services = compose["services"]

    assert services["arvo_3938_vul"]["image"] == "n132/arvo:3938-vul"
    for service_name in ("arvo_3938_vul", "arvo_3938_fix", "arvo_47101_vul", "arvo_47101_fix"):
        assert services[service_name]["user"] == "1011:1011"
    assert services["arvo_3938_vul"]["environment"] == [
        "CYBERGYM_ARVO_SERVICE_ROLE=vulnerable",
        "AUTH_TOKEN_FILE=/cybergym-data/cybergym/harbor_tasks/level1/cybergym_arvo_3938/environment/"
        "cybergym-apptainer.json",
        "RUNNER_AUTH_TOKEN=${CYBERGYM_EXTERNAL_RUNNER_TOKEN:?set CYBERGYM_EXTERNAL_RUNNER_TOKEN before launch}",
        "FIXED_RUNNER_URL=http://arvo_3938_fix:9121",
        "PORT=9111",
        "POC_TIMEOUT_SEC=10",
        "MAX_POC_BYTES=10485760",
        "MAX_OUTPUT_BYTES=1048576",
    ]
    assert services["arvo_47101_vul"]["environment"][3] == "FIXED_RUNNER_URL=http://arvo_47101_fix:9121"
    assert set(services) == {
        "lab",
        "arvo_3938_fix",
        "arvo_3938_vul",
        "arvo_47101_fix",
        "arvo_47101_vul",
    }
    assert "CYBERGYM_LOCAL_EXECUTION=true" in services["lab"]["environment"]
    assert "CYBERGYM_EXTERNAL_TASK_SERVER_URL=http://{runner_service}:9111" in services["lab"]["environment"]
    assert (
        "CYBERGYM_DATASET_PATH=${CYBERGYM_DATASET_CONTAINER_PATH:-/dli/task/data/cybergym/cybergym.jsonl}"
        in services["lab"]["environment"]
    )
    assert (
        "CYBERGYM_VALIDATION_DATASET_PATH="
        "${CYBERGYM_VALIDATION_DATASET_CONTAINER_PATH:-/dli/task/data/cybergym/cybergym_validation.jsonl}"
        in services["lab"]["environment"]
    )
