# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from responses_api_agents.harbor_agent.custom_envs.cybergym.cybergym import (
    RUNTIME_DOCKER_COMPOSE,
    CyberGymApptainerEnvironment,
    CyberGymApptainerManifest,
    CyberGymDockerEnvironment,
)
from responses_api_agents.harbor_agent.custom_envs.cybergym.task_server import (
    ApptainerRunner,
    CyberGymTaskServer,
    RunnerInfrastructureError,
)
from responses_api_agents.harbor_agent.custom_envs.singularity.singularity import SingularityEnvironment


CYBERGYM_TEMPLATE_DIR = Path(__file__).parents[3] / "benchmarks" / "cybergym" / "task_template"


def test_runtime_detection_prefers_apptainer(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {"apptainer": "/usr/bin/apptainer", "singularity": "/usr/bin/singularity"}
    monkeypatch.setattr("shutil.which", paths.get)

    assert SingularityEnvironment._resolve_singularity_executable() == "/usr/bin/apptainer"
    assert SingularityEnvironment._resolve_singularity_executable("singularity") == "/usr/bin/singularity"
    assert CyberGymApptainerEnvironment._find_host_python_mounts() == []


def test_manifest_validation(tmp_path: Path) -> None:
    path = tmp_path / "cybergym-apptainer.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_type": "oss-fuzz",
                "vulnerable_image": "cybergym/oss-fuzz:7-vul",
                "fixed_image": "cybergym/oss-fuzz:7-fix",
                "auth_token": "a" * 64,
            }
        )
    )

    manifest = CyberGymApptainerManifest.load(path)

    assert manifest.task_type == "oss-fuzz"
    assert manifest.fixed_image.endswith(":7-fix")


def test_apptainer_runner_uses_inner_status_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vulnerable_sif = tmp_path / "vul.sif"
    fixed_sif = tmp_path / "fix.sif"
    vulnerable_sif.touch()
    fixed_sif.touch()
    runner = ApptainerRunner(
        executable="/usr/bin/apptainer",
        task_type="oss-fuzz",
        vulnerable_sif=vulnerable_sif,
        fixed_sif=fixed_sif,
        fakeroot=False,
    )

    def fake_run(command, **kwargs):
        bind_source = Path(command[command.index("-B") + 1].split(":", 1)[0])
        (bind_source / "status").write_text("134\n")
        kwargs["stdout"].write(b"AddressSanitizer: crash\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    exit_code, output = runner.run_poc("vul", b"poc")

    assert exit_code == 134
    assert "AddressSanitizer" in output


def test_cybergym_runtime_does_not_require_fakeroot_by_default(tmp_path: Path) -> None:
    runner = ApptainerRunner(
        executable="/usr/bin/apptainer",
        task_type="arvo",
        vulnerable_sif=tmp_path / "vul.sif",
        fixed_sif=tmp_path / "fix.sif",
    )
    assert "--fakeroot" not in runner._execution_command(runner.vulnerable_sif, tmp_path)

    environment = object.__new__(CyberGymApptainerEnvironment)
    environment._cybergym_fakeroot = None
    assert environment._runtime_fakeroot_args() == []
    environment._cybergym_fakeroot = True
    assert environment._runtime_fakeroot_args() == ["--fakeroot"]


def test_apptainer_launcher_failure_is_not_scored_as_a_crash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vulnerable_sif = tmp_path / "vul.sif"
    fixed_sif = tmp_path / "fix.sif"
    vulnerable_sif.touch()
    fixed_sif.touch()
    runner = ApptainerRunner(
        executable="/usr/bin/apptainer",
        task_type="arvo",
        vulnerable_sif=vulnerable_sif,
        fixed_sif=fixed_sif,
        fakeroot=False,
    )

    def fake_run(_command, **kwargs):
        kwargs["stdout"].write(b"FATAL: failed to mount image\n")
        return SimpleNamespace(returncode=255)

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RunnerInfrastructureError, match="failed before"):
        runner.run_poc("vul", b"poc")


class _FakeRunner:
    def validate(self) -> None:
        return

    def read_ground_truth(self, max_poc_bytes: int) -> bytes:
        assert max_poc_bytes >= 3
        return b"gt!"

    def run_poc(self, variant: str, poc_data: bytes) -> tuple[int, str]:
        assert poc_data == b"candidate"
        return (134, "crash") if variant == "vul" else (0, "safe")


def test_loopback_task_server_keeps_verifier_endpoints_authenticated() -> None:
    token = "secret-token-for-tests"
    server = CyberGymTaskServer(runner=_FakeRunner(), auth_token=token, max_poc_bytes=1024)
    server.start()
    try:
        with urlopen(server.url + "/health", timeout=2) as response:
            assert json.load(response) == {"status": "ok"}

        submit_request = Request(server.url + "/submit", data=b"candidate", method="POST")
        with urlopen(submit_request, timeout=2) as response:
            assert json.load(response)["exit_code"] == 134

        with pytest.raises(HTTPError) as unauthorized:
            urlopen(Request(server.url + "/verify", data=b"candidate", method="POST"), timeout=2)
        assert unauthorized.value.code == 401

        verify_request = Request(
            server.url + "/verify",
            data=b"candidate",
            headers={"Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urlopen(verify_request, timeout=2) as response:
            assert json.load(response) == {"vul_exit_code": 134, "fix_exit_code": 0}

        solve_request = Request(server.url + "/solve", headers={"Authorization": f"Bearer {token}"})
        with urlopen(solve_request, timeout=2) as response:
            assert response.read() == b"gt!"
    finally:
        server.stop()


def test_loopback_task_server_latches_runner_infrastructure_failure() -> None:
    class _FailingRunner(_FakeRunner):
        def run_poc(self, variant: str, poc_data: bytes) -> tuple[int, str]:
            raise RunnerInfrastructureError("failed to mount runner SIF")

    server = CyberGymTaskServer(
        runner=_FailingRunner(),
        auth_token="secret-token-for-tests",
        max_poc_bytes=1024,
    )
    server.start()
    try:
        with pytest.raises(HTTPError) as submission_error:
            urlopen(Request(server.url + "/submit", data=b"candidate", method="POST"), timeout=2)
        assert submission_error.value.code == 500

        with pytest.raises(HTTPError) as health_error:
            urlopen(server.url + "/health", timeout=2)
        assert health_error.value.code == 503
        assert "failed to mount runner SIF" in health_error.value.read().decode()
    finally:
        server.stop()


async def test_docker_preflight_explains_missing_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = object.__new__(CyberGymDockerEnvironment)
    environment._compose_prefix = None
    monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/docker" if command == "docker" else None)

    class _Process:
        returncode = 1

        async def communicate(self):
            return b"unknown command: compose", None

    async def fake_subprocess(*_args, **_kwargs):
        return _Process()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_subprocess)

    with pytest.raises(RuntimeError, match="config_apptainer.yaml"):
        await environment._resolve_compose_prefix()


async def test_docker_exec_disables_outer_tty() -> None:
    environment = object.__new__(CyberGymDockerEnvironment)
    environment._local_execution = False
    captured: dict = {}

    async def fake_compose(command, check=True, timeout_sec=None):
        captured.update(command=command, check=check, timeout_sec=timeout_sec)
        return SimpleNamespace(return_code=0, stdout="ok", stderr=None)

    environment._run_docker_compose_command = fake_compose

    result = await environment.exec(
        "tmux -V",
        cwd="/workspace",
        env={"EXAMPLE": "two words"},
        timeout_sec=17,
    )

    assert result.return_code == 0
    assert captured == {
        "command": [
            "exec",
            "-T",
            "-w",
            "/workspace",
            "-e",
            "EXAMPLE='two words'",
            "main",
            "bash",
            "-ic",
            "tmux -V",
        ],
        "check": False,
        "timeout_sec": 17,
    }


@pytest.mark.parametrize(
    ("harbor_mode", "cybergym_override", "expected_docker_mode"),
    [
        ("bridge", None, "bridge"),
        ("none", None, "none"),
        ("bridge", "restricted", "none"),
        ("none", "public", "bridge"),
    ],
)
def test_docker_compose_network_uses_builtin_modes(
    harbor_mode: str,
    cybergym_override: str | None,
    expected_docker_mode: str,
) -> None:
    environment = object.__new__(CyberGymDockerEnvironment)
    environment._task_server_auth_token = "a" * 64
    values = {"NETWORK_MODE": harbor_mode}
    if cybergym_override is not None:
        values["CYBERGYM_NETWORK_MODE"] = cybergym_override
    environment._env_vars = SimpleNamespace(to_env_dict=lambda include_os_env: values.copy())

    compose_environment = environment._compose_environment()

    assert compose_environment["NETWORK_MODE"] == expected_docker_mode
    assert "CYBERGYM_DOCKER_NETWORK_MODE" not in compose_environment
    assert compose_environment["CYBERGYM_TASK_SERVER_AUTH_TOKEN"] == "a" * 64


def test_docker_compose_network_rejects_invalid_override() -> None:
    environment = object.__new__(CyberGymDockerEnvironment)
    environment._task_server_auth_token = "a" * 64
    environment._env_vars = SimpleNamespace(
        to_env_dict=lambda include_os_env: {
            "NETWORK_MODE": "bridge",
            "CYBERGYM_NETWORK_MODE": "invalid-mode",
        }
    )

    with pytest.raises(ValueError, match="public.*restricted"):
        environment._compose_environment()


async def test_docker_preflight_reports_stopped_container(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = object.__new__(CyberGymDockerEnvironment)
    environment._local_execution = False

    async def fake_parent_start(_self, force_build):
        assert force_build is False

    async def fake_exec(_command):
        return SimpleNamespace(
            return_code=1,
            stdout=None,
            stderr="Error response from daemon: Container abc123 is not running",
        )

    diagnostic_commands: list[list[str]] = []

    async def fake_compose(command, check=True, timeout_sec=None):
        assert check is False
        assert timeout_sec == 30
        diagnostic_commands.append(command)
        output = "main exited (1)" if command == ["ps", "-a"] else "firewall setup failed"
        return SimpleNamespace(return_code=0, stdout=output, stderr=None)

    monkeypatch.setattr(DockerEnvironment, "start", fake_parent_start)
    environment.exec = fake_exec
    environment._run_docker_compose_command = fake_compose

    with pytest.raises(RuntimeError, match="container lifecycle failure") as error:
        await environment.start(force_build=False)

    assert diagnostic_commands == [
        ["ps", "-a"],
        ["logs", "--no-color", "--tail", "200", "main"],
    ]
    assert "main exited (1)" in str(error.value)
    assert "firewall setup failed" in str(error.value)


def test_docker_runtime_compose_ignores_legacy_task_networks(tmp_path: Path) -> None:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "docker-compose.yaml").write_text(
        """
services:
  task-server:
    environment:
      - AUTH_TOKEN=0123456789abcdef0123456789abcdef
networks:
  task-internal: {}
  agent-egress: {}
"""
    )

    token = CyberGymDockerEnvironment._load_task_server_auth_token(environment_dir)

    assert token == "0123456789abcdef0123456789abcdef"
    assert 'network_mode: "service:task-server"' in RUNTIME_DOCKER_COMPOSE
    assert "network_mode: ${NETWORK_MODE:-bridge}" in RUNTIME_DOCKER_COMPOSE
    assert "entrypoint: []" in RUNTIME_DOCKER_COMPOSE
    assert "cap_add:" not in RUNTIME_DOCKER_COMPOSE
    assert "\nnetworks:" not in RUNTIME_DOCKER_COMPOSE

    trial_paths = TrialPaths(tmp_path / "trial")
    environment = CyberGymDockerEnvironment(
        environment_dir=environment_dir,
        environment_name="cybergym_test",
        session_id="cybergym_test__trial",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
    )

    assert environment._docker_compose_path == trial_paths.trial_dir / "cybergym-runtime-compose.yaml"
    assert environment._docker_compose_path.read_text() == RUNTIME_DOCKER_COMPOSE
    assert "agent-egress" not in environment._docker_compose_path.read_text()
    assert "agent-egress" in (environment_dir / "docker-compose.yaml").read_text()


async def test_local_lab_execution_uses_isolated_workspace_and_runner_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("JUPYTER_TOKEN", "must-not-reach-agent-shell")
    environment_dir = tmp_path / "cybergym_arvo_3938" / "environment"
    task_data = environment_dir / "task_data"
    task_data.mkdir(parents=True)
    (environment_dir / "docker-compose.yaml").write_text("services: {}\n")
    (environment_dir / "submit.sh").write_text("#!/bin/bash\n")
    (task_data / "description.txt").write_text("vulnerability description")
    monkeypatch.chdir(tmp_path)
    trial_paths = TrialPaths(Path("trial"))
    trial_paths.mkdir()
    environment = CyberGymDockerEnvironment(
        environment_dir=environment_dir,
        environment_name="cybergym_arvo_3938",
        session_id="cybergym_arvo_3938__trial",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        cybergym_local_execution=True,
        cybergym_external_task_server_url="http://{runner_service}:9111",
    )

    await asyncio.to_thread(environment._prepare_local_workspace_sync)
    tmux_dir = environment._local_tmux_dir
    tmux_socket = tmux_dir / f"tmux-{os.getuid()}" / "default"
    result = await environment.exec(
        'test -z "${JUPYTER_TOKEN:-}"; printf "%s" "$CYBERGYM_TASK_SERVER_URL"; printf candidate > /workspace/poc',
        timeout_sec=10,
    )
    source_tests = tmp_path / "source-tests"
    source_tests.mkdir()
    (source_tests / "test.sh").write_text("test")
    await environment.upload_dir(source_tests, "/tests")

    assert result.return_code == 0
    assert "http://arvo_3938_vul:9111" in (result.stdout or "")
    assert (environment._local_workspace / "poc").read_text() == "candidate"
    assert (environment._local_workspace / "task_data" / "description.txt").is_file()
    assert (environment._local_tests / "test.sh").read_text() == "test"
    assert not environment._runtime_compose_path.exists()
    assert tmux_dir.parent == Path("/tmp/cybergym-tmux")
    assert len(os.fsencode(tmux_socket)) < 108

    await environment.stop(delete=True)
    assert not environment._local_root.exists()
    assert not tmux_dir.exists()


async def test_local_lab_execution_uses_noninteractive_isolated_tmux_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TMUX", "/tmp/lab-tmux/default,123,0")
    monkeypatch.setenv("TMUX_PANE", "%1")
    environment_dir = tmp_path / "cybergym_arvo_3938" / "environment"
    task_data = environment_dir / "task_data"
    task_data.mkdir(parents=True)
    (environment_dir / "docker-compose.yaml").write_text("services: {}\n")
    (environment_dir / "submit.sh").write_text("#!/bin/bash\n")
    (task_data / "description.txt").write_text("vulnerability description")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    environment = CyberGymDockerEnvironment(
        environment_dir=environment_dir,
        environment_name="cybergym_arvo_3938",
        session_id="cybergym_arvo_3938__trial",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        cybergym_local_execution=True,
        cybergym_external_task_server_url="http://arvo_3938_vul:9111",
    )

    await asyncio.to_thread(environment._prepare_local_workspace_sync)
    result = await environment.exec(
        'case "$-" in *i*) exit 91;; esac; test -z "${TMUX:-}"; test -z "${TMUX_PANE:-}"',
        env={"TMUX": "/tmp/explicit-tmux/default,456,0", "TMUX_PANE": "%2"},
        timeout_sec=10,
    )

    assert result.return_code == 0, result.stdout
    assert "no job control" not in (result.stdout or "")
    await environment.stop(delete=True)


async def test_local_lab_exec_does_not_wait_for_detached_child_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    environment_dir = tmp_path / "cybergym_arvo_3938" / "environment"
    task_data = environment_dir / "task_data"
    task_data.mkdir(parents=True)
    (environment_dir / "docker-compose.yaml").write_text("services: {}\n")
    (environment_dir / "submit.sh").write_text("#!/bin/bash\n")
    (task_data / "description.txt").write_text("vulnerability description")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    environment = CyberGymDockerEnvironment(
        environment_dir=environment_dir,
        environment_name="cybergym_arvo_3938",
        session_id="cybergym_arvo_3938__trial",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        cybergym_local_execution=True,
        cybergym_external_task_server_url="http://arvo_3938_vul:9111",
    )

    await asyncio.to_thread(environment._prepare_local_workspace_sync)
    try:
        # The shell exits immediately while its detached child retains stdout.
        # Pipe-based communicate() waits for the child; file-backed capture must not.
        result = await asyncio.wait_for(environment.exec("(sleep 2) & printf ready"), timeout=1)
        assert result.return_code == 0
        assert result.stdout == "ready"
    finally:
        await environment.stop(delete=True)


@pytest.mark.skipif(
    shutil.which("tmux") is None or shutil.which("script") is None,
    reason="tmux and script are required",
)
async def test_local_lab_harbor_tmux_setup_completes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym

    monkeypatch.setenv("TMUX", "/tmp/lab-tmux/default,123,0")
    environment_dir = tmp_path / "cybergym_arvo_3938" / "environment"
    task_data = environment_dir / "task_data"
    task_data.mkdir(parents=True)
    (environment_dir / "docker-compose.yaml").write_text("services: {}\n")
    (environment_dir / "submit.sh").write_text("#!/bin/bash\n")
    (task_data / "description.txt").write_text("vulnerability description")
    trial_paths = TrialPaths(tmp_path / ("trial-" + "x" * 160))
    trial_paths.mkdir()
    environment = CyberGymDockerEnvironment(
        environment_dir=environment_dir,
        environment_name="cybergym_arvo_3938",
        session_id="cybergym_arvo_3938__trial",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        cybergym_local_execution=True,
        cybergym_external_task_server_url="http://arvo_3938_vul:9111",
    )
    agent = Terminus2NemoGym.__new__(Terminus2NemoGym)
    agent.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    agent._record_terminal_session = False
    agent._tmux_pane_width = 160
    agent._tmux_pane_height = 40
    executed_commands: list[str] = []
    original_exec = environment.exec

    async def recording_exec(command, *args, **kwargs):
        executed_commands.append(command)
        return await original_exec(command, *args, **kwargs)

    environment.exec = recording_exec

    await asyncio.to_thread(environment._prepare_local_workspace_sync)
    try:
        await asyncio.wait_for(agent.setup(environment), timeout=10)
        assert await agent._session.is_session_alive()
        assert not any("script -qc" in command for command in executed_commands)
    finally:
        await environment.stop(delete=True)


async def test_local_lab_tmux_daemon_cannot_hold_setup_output_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from responses_api_agents.harbor_agent.custom_agents.terminus_2_nemo_gym import Terminus2NemoGym

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-V" ]; then echo "tmux fake"; exit 0; fi\n'
        'if [ "$1" = "new-session" ]; then (sleep 2) & exit 0; fi\n'
        "exit 0\n"
    )
    fake_tmux.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    environment_dir = tmp_path / "cybergym_arvo_3938" / "environment"
    task_data = environment_dir / "task_data"
    task_data.mkdir(parents=True)
    (environment_dir / "docker-compose.yaml").write_text("services: {}\n")
    (environment_dir / "submit.sh").write_text("#!/bin/bash\n")
    (task_data / "description.txt").write_text("vulnerability description")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    environment = CyberGymDockerEnvironment(
        environment_dir=environment_dir,
        environment_name="cybergym_arvo_3938",
        session_id="cybergym_arvo_3938__trial",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        cybergym_local_execution=True,
        cybergym_external_task_server_url="http://arvo_3938_vul:9111",
    )
    agent = Terminus2NemoGym.__new__(Terminus2NemoGym)
    agent.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    agent._record_terminal_session = False
    agent._tmux_pane_width = 160
    agent._tmux_pane_height = 40

    await asyncio.to_thread(environment._prepare_local_workspace_sync)
    try:
        # The fake detached server deliberately retains its stdout for two
        # seconds. Setup must not wait for that inherited descriptor.
        await asyncio.wait_for(agent.setup(environment), timeout=1)
        assert (trial_paths.agent_dir / "tmux-start.log").is_file()
    finally:
        await environment.stop(delete=True)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_local_lab_tmux_uses_short_socket_for_long_trial_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    environment_dir = tmp_path / "cybergym_arvo_3938" / "environment"
    task_data = environment_dir / "task_data"
    task_data.mkdir(parents=True)
    (environment_dir / "docker-compose.yaml").write_text("services: {}\n")
    (environment_dir / "submit.sh").write_text("#!/bin/bash\n")
    (task_data / "description.txt").write_text("vulnerability description")
    monkeypatch.chdir(tmp_path)
    trial_paths = TrialPaths(Path("trial-" + "x" * 160))
    trial_paths.mkdir()
    environment = CyberGymDockerEnvironment(
        environment_dir=environment_dir,
        environment_name="cybergym_arvo_3938",
        session_id="cybergym_arvo_3938__trial",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        cybergym_local_execution=True,
        cybergym_external_task_server_url="http://arvo_3938_vul:9111",
    )

    environment._local_tmux_dir.mkdir(mode=0o700, parents=True)
    child_environment = environment._local_child_environment()
    tmux_socket = environment._local_tmux_dir / f"tmux-{os.getuid()}" / "default"
    try:
        result = subprocess.run(
            ["tmux", "new-session", "-d", "-s", "cybergym-test"],
            env=child_environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert len(os.fsencode(tmux_socket)) < 108
        assert result.returncode == 0, result.stderr
    finally:
        subprocess.run(
            ["tmux", "kill-server"],
            env=child_environment,
            check=False,
            capture_output=True,
            timeout=10,
        )
        shutil.rmtree(environment._local_tmux_dir, ignore_errors=True)


async def test_local_lab_submission_and_verifier_use_external_runner(tmp_path: Path) -> None:
    auth_token = "local-lab-verifier-token"
    server = CyberGymTaskServer(runner=_FakeRunner(), auth_token=auth_token, max_poc_bytes=1024)
    server.start()
    environment_dir = tmp_path / "cybergym_arvo_3938" / "environment"
    task_data = environment_dir / "task_data"
    task_data.mkdir(parents=True)
    (environment_dir / "docker-compose.yaml").write_text("services: {}\n")
    shutil.copy2(CYBERGYM_TEMPLATE_DIR / "environment" / "submit.sh", environment_dir / "submit.sh")
    (task_data / "description.txt").write_text("vulnerability description")
    source_tests = tmp_path / "source-tests"
    source_tests.mkdir()
    test_script = (CYBERGYM_TEMPLATE_DIR / "tests" / "test.sh").read_text()
    (source_tests / "test.sh").write_text(
        test_script.replace("{auth_token}", auth_token).replace("{scoring_mode}", "final")
    )
    shutil.copy2(CYBERGYM_TEMPLATE_DIR / "tests" / "verify.py", source_tests / "verify.py")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    environment = CyberGymDockerEnvironment(
        environment_dir=environment_dir,
        environment_name="cybergym_arvo_3938",
        session_id="cybergym_arvo_3938__trial",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(),
        cybergym_local_execution=True,
        cybergym_external_task_server_url=server.url,
    )
    try:
        await asyncio.to_thread(environment._prepare_local_workspace_sync)
        submission = await environment.exec(
            "printf candidate > /workspace/candidate && bash /workspace/submit.sh /workspace/candidate",
            timeout_sec=10,
        )
        await environment.upload_dir(source_tests, "/tests")
        verification = await environment.exec(
            "bash /tests/test.sh 2>&1 | tee /logs/verifier/test-stdout.txt",
            timeout_sec=10,
        )

        assert submission.return_code == 0
        assert '"exit_code": 134' in (submission.stdout or "")
        assert verification.return_code == 0
        assert (trial_paths.verifier_dir / "reward.txt").read_text() == "1.0"
        assert (trial_paths.agent_dir / "artifacts" / "poc").read_bytes() == b"candidate"
    finally:
        await environment.stop(delete=True)
        server.stop()
