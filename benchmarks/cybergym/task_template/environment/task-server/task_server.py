#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from harbor-framework/harbor/adapters/cybergym and modified for
# pinned-Harbor compatibility, bounded I/O, and infrastructure error handling.
"""Private CyberGym task sidecar for vulnerable/fixed PoC execution.

Compatible with Python 3.5 because some ARVO runner images use Ubuntu 16.04.
The implementation follows CyberGym's canonical runner scripts rather than
invoking a guessed fuzz-target binary directly.
"""

import hmac
import json
import os
import shutil
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PORT = int(os.environ.get("PORT", "9111"))
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
AUTH_TOKEN_FILE = os.environ.get("AUTH_TOKEN_FILE", "").strip()
RUNNER_AUTH_TOKEN = os.environ.get("RUNNER_AUTH_TOKEN", "")
SERVICE_ROLE = os.environ.get("CYBERGYM_ARVO_SERVICE_ROLE", "").strip().lower()
FIXED_RUNNER_URL = os.environ.get("FIXED_RUNNER_URL", "").strip().rstrip("/")
POC_TIMEOUT_SEC = int(os.environ.get("POC_TIMEOUT_SEC", "10"))
MAX_POC_BYTES = int(os.environ.get("MAX_POC_BYTES", str(10 * 1024 * 1024)))
MAX_OUTPUT_BYTES = int(os.environ.get("MAX_OUTPUT_BYTES", str(1024 * 1024)))
GROUND_TRUTH_POC = "/tests/ground_truth_poc"
RAW_RUNNER_PATH = os.environ.get("CYBERGYM_ARVO_RUNNER_PATH", "/bin/arvo")
RAW_POC_PATH = os.environ.get("CYBERGYM_ARVO_POC_PATH", "/tmp/poc")
RAW_RUNNER_CWD = os.environ.get("CYBERGYM_ARVO_RUNNER_CWD", "/")
INFRASTRUCTURE_ERROR = None

if SERVICE_ROLE and os.path.isfile(RAW_POC_PATH):
    with open(RAW_POC_PATH, "rb") as _ground_truth_source:
        RAW_ORIGINAL_POC = _ground_truth_source.read(MAX_POC_BYTES + 1)
else:
    RAW_ORIGINAL_POC = None

SANITIZER_ENV = dict(os.environ)
SANITIZER_ENV["ASAN_OPTIONS"] = (
    "alloc_dealloc_mismatch=0:allocator_may_return_null=1:"
    "allocator_release_to_os_interval_ms=500:check_malloc_usable_size=0:"
    "detect_container_overflow=1:detect_odr_violation=0:detect_leaks=0:"
    "detect_stack_use_after_return=1:fast_unwind_on_fatal=0:"
    "handle_abort=1:handle_segv=1:handle_sigill=1:"
    "max_uar_stack_size_log=16:print_scariness=1:quarantine_size_mb=10:"
    "strict_memcmp=1:strip_path_prefix=/workspace/:symbolize=1:"
    "use_sigaltstack=1:dedup_token_length=3"
)
SANITIZER_ENV["MSAN_OPTIONS"] = "print_stats=1:strip_path_prefix=/workspace/:symbolize=1:dedup_token_length=3"
SANITIZER_ENV["UBSAN_OPTIONS"] = (
    "print_stacktrace=1:print_summary=1:silence_unsigned_overflow=1:"
    "strip_path_prefix=/workspace/:symbolize=1:dedup_token_length=3"
)


class RequestError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _runner_path(variant):
    if SERVICE_ROLE:
        local_variant = "vul" if SERVICE_ROLE == "vulnerable" else "fix"
        if variant != local_variant:
            return None
        return RAW_RUNNER_PATH if os.path.isfile(RAW_RUNNER_PATH) and os.access(RAW_RUNNER_PATH, os.X_OK) else None
    base = "/cybergym/" + variant + "/bin"
    for name in ("arvo", "run_poc"):
        path = base + "/" + name
        if os.path.isfile(path):
            return path
    return None


def _activate_out_dir(variant):
    if SERVICE_ROLE:
        return
    target = "/cybergym/" + variant + "/out"
    if os.path.lexists("/out"):
        if os.path.isdir("/out") and not os.path.islink("/out"):
            shutil.rmtree("/out")
        else:
            os.unlink("/out")
    os.symlink(target, "/out")


def _read_capped_output(output_file):
    output_file.seek(0)
    data = output_file.read(MAX_OUTPUT_BYTES + 1)
    if len(data) > MAX_OUTPUT_BYTES:
        data = data[:MAX_OUTPUT_BYTES] + b"\n[output truncated by CyberGym task server]\n"
    return data.decode("utf-8", errors="replace")


def _remote_fixed_run(poc_path):
    if not FIXED_RUNNER_URL or not RUNNER_AUTH_TOKEN:
        return None, "fixed runner URL or authentication token is missing"
    with open(poc_path, "rb") as source:
        poc_data = source.read(MAX_POC_BYTES + 1)
    request = Request(
        FIXED_RUNNER_URL + "/run",
        data=poc_data,
        headers={
            "Authorization": "Bearer " + RUNNER_AUTH_TOKEN,
            "Content-Type": "application/octet-stream",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=POC_TIMEOUT_SEC + 5) as response:
            response_data = response.read(MAX_OUTPUT_BYTES + 65537)
    except HTTPError as error:
        detail = error.read(4096).decode("utf-8", errors="replace")
        return None, "fixed runner returned HTTP %d: %s" % (error.code, detail)
    except (OSError, URLError) as error:
        return None, "fixed runner request failed: %s" % error
    if len(response_data) > MAX_OUTPUT_BYTES + 65536:
        return None, "fixed runner response exceeded the configured output limit"
    try:
        value = json.loads(response_data.decode("utf-8"))
        return int(value["exit_code"]), str(value.get("output", ""))
    except (KeyError, TypeError, ValueError) as error:
        return None, "fixed runner returned an invalid response: %s" % error


def _restore_raw_poc():
    if not SERVICE_ROLE:
        return
    if RAW_ORIGINAL_POC is None:
        if os.path.exists(RAW_POC_PATH):
            os.unlink(RAW_POC_PATH)
        return
    with open(RAW_POC_PATH, "wb") as destination:
        destination.write(RAW_ORIGINAL_POC)


def _run_poc(variant, poc_path):
    if SERVICE_ROLE == "vulnerable" and variant == "fix":
        return _remote_fixed_run(poc_path)
    runner = _runner_path(variant)
    if runner is None:
        return None, "runner script not found"

    _activate_out_dir(variant)
    shutil.copyfile(poc_path, RAW_POC_PATH if SERVICE_ROLE else "/tmp/poc")
    command = [runner]
    if runner.endswith("/run_poc"):
        command.append(RAW_POC_PATH if SERVICE_ROLE else "/tmp/poc")

    with tempfile.TemporaryFile() as output_file:
        try:
            result = subprocess.run(
                command,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                timeout=POC_TIMEOUT_SEC,
                env=SANITIZER_ENV,
                cwd=RAW_RUNNER_CWD if SERVICE_ROLE else "/",
            )
            return result.returncode, _read_capped_output(output_file)
        except subprocess.TimeoutExpired:
            return 124, "Timeout waiting for the target binary, not crashed"
        except Exception as error:
            return None, "%s: %s" % (type(error).__name__, error)
        finally:
            _restore_raw_poc()


def _authorized(handler):
    supplied = handler.headers.get("Authorization", "")
    auth_token = _auth_token()
    expected = "Bearer " + auth_token
    return bool(auth_token) and hmac.compare_digest(supplied, expected)


def _auth_token():
    if AUTH_TOKEN:
        return AUTH_TOKEN
    if not AUTH_TOKEN_FILE or not os.path.isfile(AUTH_TOKEN_FILE):
        return ""
    try:
        with open(AUTH_TOKEN_FILE, "rb") as source:
            value = json.loads(source.read(1024 * 1024).decode("utf-8"))
        token = str(value.get("auth_token", ""))
    except (OSError, TypeError, ValueError):
        return ""
    return token if len(token) >= 16 else ""


def _runner_authorized(handler):
    supplied = handler.headers.get("Authorization", "")
    expected = "Bearer " + RUNNER_AUTH_TOKEN
    return bool(RUNNER_AUTH_TOKEN) and hmac.compare_digest(supplied, expected)


def _fixed_runner_healthy():
    if not FIXED_RUNNER_URL:
        return False, "FIXED_RUNNER_URL is missing"
    try:
        with urlopen(FIXED_RUNNER_URL + "/health", timeout=5) as response:
            value = json.loads(response.read(65536).decode("utf-8"))
        if response.status == 200 and value.get("status") == "ok":
            return True, None
        return False, "fixed runner reported degraded health"
    except (HTTPError, OSError, URLError, TypeError, ValueError) as error:
        return False, "fixed runner health check failed: %s" % error


def _health_status():
    if INFRASTRUCTURE_ERROR:
        return False, INFRASTRUCTURE_ERROR
    if SERVICE_ROLE == "fixed":
        if not RUNNER_AUTH_TOKEN:
            return False, "RUNNER_AUTH_TOKEN is missing"
        if not _runner_path("fix"):
            return False, "fixed runner script not found"
        return True, None
    if SERVICE_ROLE == "vulnerable":
        if not _auth_token():
            return False, "AUTH_TOKEN or a readable AUTH_TOKEN_FILE is missing"
        if not RUNNER_AUTH_TOKEN:
            return False, "RUNNER_AUTH_TOKEN is missing"
        if not _runner_path("vul"):
            return False, "vulnerable runner script not found"
        if RAW_ORIGINAL_POC is None or len(RAW_ORIGINAL_POC) > MAX_POC_BYTES:
            return False, "ground truth PoC is missing or too large"
        return _fixed_runner_healthy()
    ready = bool(_runner_path("vul") and _runner_path("fix") and os.path.isfile(GROUND_TRUTH_POC))
    return ready, None if ready else "local vulnerable/fixed runner assets are missing"


def _ground_truth_poc():
    if SERVICE_ROLE == "vulnerable":
        return RAW_ORIGINAL_POC
    if not os.path.isfile(GROUND_TRUTH_POC):
        return None
    with open(GROUND_TRUTH_POC, "rb") as source:
        return source.read(MAX_POC_BYTES + 1)


def _read_multipart_poc(handler):
    raw_length = handler.headers.get("Content-Length")
    if raw_length is None:
        raise RequestError(411, "Content-Length is required")
    try:
        content_length = int(raw_length)
    except ValueError:
        raise RequestError(400, "invalid Content-Length")
    if content_length <= 0:
        raise RequestError(400, "empty request body")
    if content_length > MAX_POC_BYTES + 1024 * 1024:
        raise RequestError(413, "PoC exceeds the configured size limit")

    body = handler.rfile.read(content_length)
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        data = body
    else:
        boundary_value = content_type.split("boundary=", 1)[-1].strip().strip('"')
        if not boundary_value:
            raise RequestError(400, "multipart boundary is missing")
        boundary = ("--" + boundary_value).encode("utf-8")
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

    if len(data) > MAX_POC_BYTES:
        raise RequestError(413, "PoC exceeds the configured size limit")
    return data


class TaskServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format_string, *args):
        return

    def _send_json(self, status_code, value):
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_json(self, status_code, message):
        self._send_json(status_code, {"error": message})

    def do_GET(self):
        if self.path == "/health":
            ready, detail = _health_status()
            response = {"status": "ok" if ready else "degraded"}
            if detail:
                response["error"] = detail
            self._send_json(200 if ready else 503, response)
            return

        if self.path == "/solve":
            if SERVICE_ROLE == "fixed":
                self._send_error_json(404, "not found")
                return
            if not _authorized(self):
                self._send_error_json(401, "unauthorized")
                return
            payload = _ground_truth_poc()
            if payload is None or len(payload) > MAX_POC_BYTES:
                self._send_error_json(404, "ground truth PoC not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            return

        self._send_error_json(404, "not found")

    def do_POST(self):
        global INFRASTRUCTURE_ERROR
        if SERVICE_ROLE == "fixed":
            if self.path != "/run":
                self._send_error_json(404, "not found")
                return
            if not _runner_authorized(self):
                self._send_error_json(401, "unauthorized")
                return
            try:
                poc_data = _read_multipart_poc(self)
            except RequestError as error:
                self._send_error_json(error.status_code, error.message)
                return
            temporary_dir = tempfile.mkdtemp(prefix="cybergym-fixed-poc-")
            try:
                poc_path = os.path.join(temporary_dir, "poc")
                with open(poc_path, "wb") as destination:
                    destination.write(poc_data)
                fixed_exit, fixed_output = _run_poc("fix", poc_path)
                if fixed_exit is None:
                    INFRASTRUCTURE_ERROR = fixed_output
                    self._send_error_json(500, fixed_output)
                    return
                self._send_json(200, {"exit_code": fixed_exit, "output": fixed_output})
            finally:
                shutil.rmtree(temporary_dir, ignore_errors=True)
            return

        if self.path not in ("/submit", "/verify"):
            self._send_error_json(404, "not found")
            return
        if self.path == "/verify" and not _authorized(self):
            self._send_error_json(401, "unauthorized")
            return

        try:
            poc_data = _read_multipart_poc(self)
        except RequestError as error:
            self._send_error_json(error.status_code, error.message)
            return

        temporary_dir = tempfile.mkdtemp(prefix="cybergym-poc-")
        try:
            poc_path = os.path.join(temporary_dir, "poc")
            with open(poc_path, "wb") as destination:
                destination.write(poc_data)

            vulnerable_exit, vulnerable_output = _run_poc("vul", poc_path)
            if vulnerable_exit is None:
                INFRASTRUCTURE_ERROR = vulnerable_output
                self._send_error_json(500, vulnerable_output)
                return

            if self.path == "/submit":
                reported_exit = 0 if vulnerable_exit == 124 else vulnerable_exit
                self._send_json(200, {"exit_code": reported_exit, "output": vulnerable_output})
                return

            fixed_exit, fixed_output = _run_poc("fix", poc_path)
            if fixed_exit is None:
                INFRASTRUCTURE_ERROR = fixed_output
                self._send_error_json(500, fixed_output)
                return
            self._send_json(200, {"vul_exit_code": vulnerable_exit, "fix_exit_code": fixed_exit})
        finally:
            shutil.rmtree(temporary_dir, ignore_errors=True)


if __name__ == "__main__":
    if SERVICE_ROLE not in ("", "vulnerable", "fixed"):
        raise ValueError("CYBERGYM_ARVO_SERVICE_ROLE must be 'vulnerable' or 'fixed'")
    description = SERVICE_ROLE + " service" if SERVICE_ROLE else "task server"
    print("CyberGym %s listening on port %d" % (description, PORT))
    HTTPServer(("0.0.0.0", PORT), TaskServerHandler).serve_forever()
