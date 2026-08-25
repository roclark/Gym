# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loopback-only CyberGym runner service for Apptainer environments."""

from __future__ import annotations

import hmac
import json
import subprocess
import tempfile
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


SANITIZER_ENV = {
    "ASAN_OPTIONS": (
        "alloc_dealloc_mismatch=0:allocator_may_return_null=1:"
        "allocator_release_to_os_interval_ms=500:check_malloc_usable_size=0:"
        "detect_container_overflow=1:detect_odr_violation=0:detect_leaks=0:"
        "detect_stack_use_after_return=1:fast_unwind_on_fatal=0:"
        "handle_abort=1:handle_segv=1:handle_sigill=1:"
        "max_uar_stack_size_log=16:print_scariness=1:quarantine_size_mb=10:"
        "strict_memcmp=1:strip_path_prefix=/workspace/:symbolize=1:"
        "use_sigaltstack=1:dedup_token_length=3"
    ),
    "MSAN_OPTIONS": "print_stats=1:strip_path_prefix=/workspace/:symbolize=1:dedup_token_length=3",
    "UBSAN_OPTIONS": (
        "print_stacktrace=1:print_summary=1:silence_unsigned_overflow=1:"
        "strip_path_prefix=/workspace/:symbolize=1:dedup_token_length=3"
    ),
}


class RunnerInfrastructureError(RuntimeError):
    """Raised when Apptainer fails before the target process returns an exit code."""


class RequestError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class ApptainerRunner:
    """Run untrusted PoCs in separate vulnerable and fixed SIF images."""

    def __init__(
        self,
        *,
        executable: str,
        task_type: str,
        vulnerable_sif: Path,
        fixed_sif: Path,
        timeout_sec: int = 10,
        max_output_bytes: int = 1024 * 1024,
        fakeroot: bool | None = None,
    ) -> None:
        if task_type not in {"arvo", "oss-fuzz"}:
            raise ValueError(f"Unsupported CyberGym task type: {task_type!r}")
        if timeout_sec <= 0 or max_output_bytes <= 0:
            raise ValueError("CyberGym runner timeout and output limit must be positive")

        self.executable = executable
        self.task_type = task_type
        self.vulnerable_sif = vulnerable_sif
        self.fixed_sif = fixed_sif
        self.timeout_sec = timeout_sec
        self.max_output_bytes = max_output_bytes
        # SIF execution does not require fakeroot by default. The caller can
        # opt in for sites/images that specifically need it.
        self.fakeroot = False if fakeroot is None else fakeroot

    @property
    def runner_path(self) -> str:
        return "/bin/arvo" if self.task_type == "arvo" else "/usr/local/bin/run_poc"

    def _inspection_command(self, sif_path: Path, *command: str) -> list[str]:
        # Disable implicit host mounts, especially /tmp, so validation and oracle
        # extraction inspect the image rather than similarly named host files.
        return [
            self.executable,
            "exec",
            "--cleanenv",
            "--no-mount",
            "home",
            "--no-mount",
            "tmp",
            "--no-mount",
            "bind-paths",
            str(sif_path),
            *command,
        ]

    def validate(self) -> None:
        for variant, sif_path in (("vulnerable", self.vulnerable_sif), ("fixed", self.fixed_sif)):
            if not sif_path.is_file():
                raise FileNotFoundError(f"CyberGym {variant} runner SIF does not exist: {sif_path}")
            result = subprocess.run(
                self._inspection_command(sif_path, "test", "-x", self.runner_path),
                check=False,
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                output = (result.stderr or result.stdout or b"").decode(errors="replace")[-4000:]
                raise RunnerInfrastructureError(
                    f"CyberGym {variant} image has no executable {self.runner_path}: {output.strip()}"
                )

    def read_ground_truth(self, max_poc_bytes: int) -> bytes:
        result = subprocess.run(
            self._inspection_command(self.vulnerable_sif, "cat", "/tmp/poc"),
            check=False,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            output = (result.stderr or b"").decode(errors="replace")[-4000:]
            raise RunnerInfrastructureError(f"Failed to extract CyberGym ground-truth PoC: {output.strip()}")
        if not result.stdout:
            raise RunnerInfrastructureError("CyberGym ground-truth PoC is empty")
        if len(result.stdout) > max_poc_bytes:
            raise RunnerInfrastructureError(
                f"CyberGym ground-truth PoC exceeds the configured {max_poc_bytes}-byte limit"
            )
        return result.stdout

    def _execution_command(self, sif_path: Path, io_dir: Path) -> list[str]:
        runner_command = self.runner_path
        if self.task_type == "oss-fuzz":
            runner_command += " /tmp/poc"
        shell_command = (
            "cp /cybergym_io/poc /tmp/poc || exit 90; "
            f"{runner_command}; rc=$?; "
            "printf '%s\\n' \"$rc\" > /cybergym_io/status; exit 0"
        )
        fakeroot_args = ["--fakeroot"] if self.fakeroot else []
        environment_args = [
            value for item in sorted(SANITIZER_ENV.items()) for value in ("--env", f"{item[0]}={item[1]}")
        ]
        return [
            self.executable,
            "exec",
            "--cleanenv",
            "--no-mount",
            "home",
            "--no-mount",
            "tmp",
            "--no-mount",
            "bind-paths",
            *environment_args,
            "--writable-tmpfs",
            *fakeroot_args,
            "--containall",
            "--pid",
            "-B",
            f"{io_dir}:/cybergym_io",
            str(sif_path),
            "/bin/sh",
            "-c",
            shell_command,
        ]

    def run_poc(self, variant: str, poc_data: bytes) -> tuple[int, str]:
        if variant not in {"vul", "fix"}:
            raise ValueError(f"Unknown CyberGym runner variant: {variant!r}")
        sif_path = self.vulnerable_sif if variant == "vul" else self.fixed_sif

        with tempfile.TemporaryDirectory(prefix="cybergym-apptainer-poc-") as temporary_dir:
            io_dir = Path(temporary_dir)
            io_dir.chmod(0o700)
            (io_dir / "poc").write_bytes(poc_data)

            with tempfile.TemporaryFile() as output_file:
                try:
                    result = subprocess.run(
                        self._execution_command(sif_path, io_dir),
                        check=False,
                        stdout=output_file,
                        stderr=subprocess.STDOUT,
                        timeout=self.timeout_sec,
                    )
                except subprocess.TimeoutExpired:
                    return 124, "Timeout waiting for the target binary, not crashed"

                output_file.seek(0)
                output = output_file.read(self.max_output_bytes + 1)
                if len(output) > self.max_output_bytes:
                    output = output[: self.max_output_bytes] + b"\n[output truncated by CyberGym task server]\n"

            status_path = io_dir / "status"
            if not status_path.is_file():
                diagnostic = output.decode("utf-8", errors="replace")
                raise RunnerInfrastructureError(
                    "Apptainer failed before the CyberGym target returned an exit status "
                    f"(runtime exit {result.returncode}): {diagnostic[-4000:]}"
                )
            try:
                exit_code = int(status_path.read_text(encoding="utf-8").strip())
            except ValueError as error:
                raise RunnerInfrastructureError("CyberGym runner wrote an invalid target exit status") from error
            return exit_code, output.decode("utf-8", errors="replace")


class _CyberGymService:
    def __init__(self, runner: ApptainerRunner, auth_token: str, max_poc_bytes: int) -> None:
        self.runner = runner
        self.auth_token = auth_token
        self.max_poc_bytes = max_poc_bytes
        self.ground_truth_poc = b""
        self.infrastructure_error: str | None = None

    def initialize(self) -> None:
        self.runner.validate()
        self.ground_truth_poc = self.runner.read_ground_truth(self.max_poc_bytes)


def _read_multipart_poc(handler: BaseHTTPRequestHandler, max_poc_bytes: int) -> bytes:
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        raise RequestError(411, "Content-Length is required")
    try:
        content_length = int(raw_length)
    except ValueError as error:
        raise RequestError(400, "invalid Content-Length") from error
    if content_length <= 0:
        raise RequestError(400, "empty request body")
    if content_length > max_poc_bytes + 1024 * 1024:
        raise RequestError(413, "PoC exceeds the configured size limit")

    body = handler.rfile.read(content_length)
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        data = body
    else:
        boundary_value = content_type.split("boundary=", 1)[-1].strip().strip('"')
        if not boundary_value:
            raise RequestError(400, "multipart boundary is missing")
        boundary = ("--" + boundary_value).encode()
        data = None
        for part in body.split(boundary):
            if b"Content-Disposition" not in part or b"\r\n\r\n" not in part:
                continue
            _, candidate = part.split(b"\r\n\r\n", 1)
            if candidate.endswith(b"\r\n"):
                candidate = candidate[:-2]
            data = candidate
            break
        if data is None:
            raise RequestError(400, "multipart request has no PoC file")
    if len(data) > max_poc_bytes:
        raise RequestError(413, "PoC exceeds the configured size limit")
    return data


class _CyberGymHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, service: _CyberGymService, **kwargs) -> None:
        self.service = service
        super().__init__(*args, **kwargs)

    def log_message(self, format_string: str, *args) -> None:
        return

    def _send_json(self, status_code: int, value: dict) -> None:
        payload = json.dumps(value).encode()
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def _authorized(self) -> bool:
        expected = "Bearer " + self.service.auth_token
        supplied = self.headers.get("Authorization", "")
        return bool(self.service.auth_token) and hmac.compare_digest(supplied, expected)

    def do_GET(self) -> None:
        if self.path == "/health":
            if self.service.infrastructure_error is not None:
                self._send_json(503, {"status": "degraded", "error": self.service.infrastructure_error})
            else:
                self._send_json(200, {"status": "ok"})
            return
        if self.path == "/solve":
            if not self._authorized():
                self._send_json(401, {"error": "unauthorized"})
                return
            payload = self.service.ground_truth_poc
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in {"/submit", "/verify"}:
            self._send_json(404, {"error": "not found"})
            return
        if self.path == "/verify" and not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        try:
            poc_data = _read_multipart_poc(self, self.service.max_poc_bytes)
            vulnerable_exit, vulnerable_output = self.service.runner.run_poc("vul", poc_data)
            if self.path == "/submit":
                reported_exit = 0 if vulnerable_exit == 124 else vulnerable_exit
                self._send_json(200, {"exit_code": reported_exit, "output": vulnerable_output})
                return
            fixed_exit, _ = self.service.runner.run_poc("fix", poc_data)
            self._send_json(200, {"vul_exit_code": vulnerable_exit, "fix_exit_code": fixed_exit})
        except RequestError as error:
            self._send_json(error.status_code, {"error": error.message})
        except RunnerInfrastructureError as error:
            self.service.infrastructure_error = str(error)
            self._send_json(500, {"error": str(error)})


class CyberGymTaskServer:
    """Own a loopback HTTP service that keeps trusted runner assets off the agent SIF."""

    def __init__(self, runner: ApptainerRunner, auth_token: str, max_poc_bytes: int) -> None:
        if len(auth_token) < 16:
            raise ValueError("CyberGym task-server auth tokens must contain at least 16 characters")
        if max_poc_bytes <= 0:
            raise ValueError("CyberGym PoC size limit must be positive")
        self._service = _CyberGymService(runner, auth_token, max_poc_bytes)
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("CyberGym task server has not started")
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> None:
        if self._server is not None:
            return
        self._service.initialize()
        handler = partial(_CyberGymHandler, service=self._service)
        # Serialize requests so an untrusted agent cannot fan out many
        # vulnerable runner processes on the host.
        self._server = HTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"cybergym-task-server-{self._server.server_port}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
