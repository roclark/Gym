# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import runpy
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from benchmarks.cybergym.prepare import (
    SUBSET_TASK_IDS,
    CyberGymRecord,
    CyberGymTaskGenerator,
    prepare,
    prepare_records,
    select_records,
)
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.train_data_utils import TrainDataProcessor


def _record(task_id: str = "arvo:1065") -> CyberGymRecord:
    task_type, numeric_id = task_id.split(":", 1)
    prefix = f"data/{task_type}/{numeric_id}"
    return CyberGymRecord.from_dict(
        {
            "task_id": task_id,
            "project_name": f"project-{numeric_id}",
            "project_homepage": "https://example.test",
            "project_main_repo": "https://example.test/repo.git",
            "project_language": "C++",
            "vulnerability_description": "A test vulnerability",
            "task_difficulty": {
                "level0": [f"{prefix}/repo-vul.tar.gz"],
                "level1": [f"{prefix}/repo-vul.tar.gz", f"{prefix}/description.txt"],
            },
        }
    )


def test_subset_selection_preserves_canonical_order() -> None:
    records = [_record(task_id) for task_id in reversed(SUBSET_TASK_IDS)]

    selected = select_records(records)

    assert tuple(record.task_id for record in selected) == SUBSET_TASK_IDS


def test_explicit_ids_override_subset_and_filters() -> None:
    records = [_record("arvo:1065"), _record("oss-fuzz:42535201")]

    selected = select_records(records, task_ids=["oss-fuzz:42535201", "arvo:1065"], task_type="arvo")

    assert [record.task_id for record in selected] == ["arvo:1065"]


def test_prepare_local_task_and_rollout_row(tmp_path: Path) -> None:
    record = _record()
    local_data = tmp_path / "hf"
    for relative_path in record.task_difficulty["level1"]:
        path = local_data / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{path.name}".encode())

    output_path = tmp_path / "cybergym.jsonl"
    tasks_root = tmp_path / "tasks"
    prepare_records(
        [record],
        difficulty="level1",
        scoring_mode="final",
        dataset_revision="abc123",
        tasks_root=tasks_root,
        output_path=output_path,
        data_dir=local_data,
        auth_token_factory=lambda: "a" * 64,
    )

    task_dir = tasks_root / "level1" / "cybergym_arvo_1065"
    task_toml = tomllib.loads((task_dir / "task.toml").read_text())
    compose = yaml.safe_load((task_dir / "environment" / "docker-compose.yaml").read_text())
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
    apptainer_definition = (task_dir / "environment" / "Apptainer.agent.def").read_text()
    apptainer_manifest = json.loads((task_dir / "environment" / "cybergym-apptainer.json").read_text())
    row = json.loads(output_path.read_text())

    assert task_toml["version"] == "1.0"
    assert task_toml["metadata"]["cybergym_task_id"] == "arvo:1065"
    assert task_toml["metadata"]["cybergym_scoring_mode"] == "final"
    assert set(compose["services"]) == {"main", "task-server"}
    assert compose["services"]["main"]["depends_on"]["task-server"]["condition"] == "service_healthy"
    assert "networks" not in compose
    assert compose["services"]["main"]["network_mode"] == "service:task-server"
    assert compose["services"]["main"]["entrypoint"] == []
    assert "cap_add" not in compose["services"]["main"]
    assert compose["services"]["task-server"]["network_mode"] == "${NETWORK_MODE:-bridge}"
    assert "CYBERGYM_TASK_SERVER_URL=http://127.0.0.1:9111" in compose["services"]["main"]["environment"]
    assert "COPY task_data/ /workspace/task_data/" in dockerfile
    assert "bsdutils" in dockerfile
    assert "ADD https://codeload.github.com/tmux/tmux/tar.gz/refs/tags/3.5a" in dockerfile
    assert "49e68b41dec0bf408990160ee12fa29b06dee8f74c1f0b4b71c9d2a1477dd910" in dockerfile
    assert "sh autogen.sh" in dockerfile
    assert "Bootstrap: docker\nFrom: n132/arvo:1065-vul" in apptainer_definition
    assert "bsdutils" in apptainer_definition
    assert "/etc/ssl/certs/ca-certificates.crt /tmp/host-ca-certificates.crt" in apptainer_definition
    assert "curl -fL --retry 5 --retry-delay 2" in apptainer_definition
    assert "export UV_SYSTEM_CERTS=true" in apptainer_definition
    assert "task_data/description.txt /workspace/task_data/description.txt" in apptainer_definition
    assert "rm -rf /out /bin/arvo /tmp/poc" in apptainer_definition
    assert "UV_PYTHON_INSTALL_DIR=/opt/uv-python uv python install 3.12.3" in apptainer_definition
    assert "fastapi==0.141.1" in apptainer_definition
    assert apptainer_manifest == {
        "schema_version": 1,
        "task_type": "arvo",
        "vulnerable_image": "n132/arvo:1065-vul",
        "fixed_image": "n132/arvo:1065-fix",
        "auth_token": "a" * 64,
    }
    assert (task_dir / "environment" / "task_data" / "description.txt").is_file()
    assert row["instance_id"] == "cybergym::cybergym_arvo_1065"
    assert row["responses_create_params"] == {"input": []}
    assert row["cybergym_dataset_revision"] == "abc123"

    secret = "a" * 64
    assert secret in (task_dir / "tests" / "test.sh").read_text()
    assert secret in (task_dir / "solution" / "solve.sh").read_text()
    assert secret not in (task_dir / "environment" / "submit.sh").read_text()
    assert secret not in apptainer_definition
    assert secret not in (task_dir / "instruction.md").read_text()


def test_remote_task_data_is_pinned_and_url_encoded(tmp_path: Path) -> None:
    generator = CyberGymTaskGenerator(
        tmp_path,
        difficulty="level1",
        scoring_mode="any",
        dataset_revision="refs/pr/7",
        auth_token_factory=lambda: "b" * 64,
    )

    task_dir = generator.generate(_record())
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
    apptainer_definition = (task_dir / "environment" / "Apptainer.agent.def").read_text()

    assert "/resolve/refs%2Fpr%2F7/data/arvo/1065/repo-vul.tar.gz" in dockerfile
    assert "/resolve/refs%2Fpr%2F7/data/arvo/1065/repo-vul.tar.gz" in apptainer_definition
    assert "wget -qO /workspace/task_data/repo-vul.tar.gz" in apptainer_definition
    assert "This task uses the `any` scoring mode." in (task_dir / "instruction.md").read_text()


def test_download_task_data_materializes_local_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    download_dir = tmp_path / "downloads"
    calls = []

    def fake_download(*, repo_id, filename, repo_type, revision):
        calls.append((repo_id, filename, repo_type, revision))
        path = download_dir / Path(filename).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("downloaded:" + filename).encode())
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    generator = CyberGymTaskGenerator(
        tmp_path / "tasks",
        difficulty="level1",
        scoring_mode="final",
        dataset_revision="abc123",
        download_task_data=True,
        auth_token_factory=lambda: "c" * 64,
    )

    task_dir = generator.generate(_record())

    assert len(calls) == 2
    assert all(call[0] == "sunblaze-ucb/cybergym" and call[2:] == ("dataset", "abc123") for call in calls)
    assert "COPY task_data/ /workspace/task_data/" in (task_dir / "environment" / "Dockerfile").read_text()
    assert (
        "task_data/description.txt /workspace/task_data/description.txt"
        in (task_dir / "environment" / "Apptainer.agent.def").read_text()
    )
    assert (task_dir / "environment" / "task_data" / "repo-vul.tar.gz").is_file()


def test_custom_output_uses_distinct_validation_alias_and_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = _record()
    monkeypatch.setattr("benchmarks.cybergym.prepare.load_records", lambda **_: ([record], "abc123"))
    output_path = tmp_path / "cybergym.jsonl"
    validation_output_path = tmp_path / "cybergym_validation.jsonl"

    prepare(
        task_ids=[record.task_id],
        tasks_root=tmp_path / "tasks",
        output_path=output_path,
    )

    assert validation_output_path.read_bytes() == output_path.read_bytes()

    monkeypatch.setenv("CYBERGYM_DATASET_PATH", str(output_path))
    monkeypatch.setenv("CYBERGYM_VALIDATION_DATASET_PATH", str(validation_output_path))
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": ["benchmarks/cybergym/config.yaml"]},
    )
    parser = GlobalConfigDictParser()
    config = parser.parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    server_instances = [
        server
        for server in parser.filter_for_server_instance_configs(config)
        if server.SERVER_TYPE in ("responses_api_agents", "resources_servers") and server.datasets
    ]

    TrainDataProcessor().validate_samples_and_aggregate_metrics(
        server_instances,
        overwrite_metrics_conflicts=False,
    )

    assert (tmp_path / "cybergym_metrics.json").is_file()
    assert (tmp_path / "cybergym_validation_metrics.json").is_file()
    assert not (tmp_path / "cybergym_metrics_conflict.json").exists()


def test_rejects_unsafe_dataset_paths() -> None:
    value = _record().raw
    value["task_difficulty"] = {"level1": ["../reference-poc"]}

    try:
        CyberGymRecord.from_dict(value)
    except ValueError as error:
        assert "Unsafe CyberGym data path" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unsafe path was accepted")


def test_reward_matches_dual_runner_contract() -> None:
    verifier_path = Path(__file__).parents[1] / "task_template" / "tests" / "verify.py"
    score_exit_codes = runpy.run_path(str(verifier_path))["score_exit_codes"]

    assert score_exit_codes(1, 0) == 1.0
    assert score_exit_codes(-6, 124) == 1.0
    assert score_exit_codes(1, 1) == 0.0
    assert score_exit_codes(124, 0) == 0.0
    assert score_exit_codes(125, 0) == 1.0


def test_templates_have_valid_python_and_shell_syntax() -> None:
    template_dir = Path(__file__).parents[1] / "task_template"
    task_server = template_dir / "environment" / "task-server" / "task_server.py"
    verifier = template_dir / "tests" / "verify.py"

    ast.parse(task_server.read_text(), filename=str(task_server), feature_version=(3, 5))
    ast.parse(verifier.read_text(), filename=str(verifier))

    for script in (
        template_dir / "environment" / "restrict-network.sh",
        template_dir / "environment" / "submit.sh",
        template_dir / "tests" / "test.sh",
        template_dir / "solution" / "solve.sh",
    ):
        subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.parametrize(
    ("config_name", "instance_name", "environment_class"),
    [
        ("config.yaml", "cybergym", "CyberGymDockerEnvironment"),
        ("config_apptainer.yaml", "cybergym_apptainer", "CyberGymApptainerEnvironment"),
        ("config_training.yaml", "cybergym_training", "CyberGymDockerEnvironment"),
        ("config_training_apptainer.yaml", "cybergym_training", "CyberGymApptainerEnvironment"),
    ],
)
def test_config_variants_resolve_to_one_server(config_name: str, instance_name: str, environment_class: str) -> None:
    config_path = f"benchmarks/cybergym/{config_name}"
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {"config_paths": [config_path]},
    )
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
        )
    )
    instances = [
        key for key, value in config.items() if isinstance(value, DictConfig) and value.get("responses_api_agents")
    ]

    assert instances == [instance_name]
    agent = config[instance_name].responses_api_agents.harbor_agent
    assert agent.harbor_environment_import_path.endswith(f":{environment_class}")
    assert agent.harbor_fail_on_trial_error is True
    if environment_class == "CyberGymDockerEnvironment":
        assert "cybergym_local_execution" in agent.harbor_environment_kwargs
        assert "cybergym_external_task_server_url" in agent.harbor_environment_kwargs
