# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare CyberGym metadata and pinned-Harbor-compatible task directories.

The task conversion is derived from the Apache-2.0 CyberGym adapter in
harbor-framework/harbor. NeMo Gym pins an older Harbor revision whose Docker
backend expects a task docker-compose.yaml to be self-contained, so the local
template includes Harbor's base main service as well as CyberGym's verifier
sidecar.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shlex
import shutil
import stat
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import quote


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "cybergym_benchmark.jsonl"
VALIDATION_OUTPUT_FPATH = DATA_DIR / "cybergym_validation.jsonl"
TRAIN_OUTPUT_FPATH = DATA_DIR / "cybergym_train.jsonl"
DEFAULT_TASKS_ROOT = DATA_DIR / "harbor_tasks"
TEMPLATE_DIR = BENCHMARK_DIR / "task_template"
APPTAINER_MANIFEST_FILENAME = "cybergym-apptainer.json"
APPTAINER_AGENT_DEFINITION_FILENAME = "Apptainer.agent.def"

HF_DATASET = "sunblaze-ucb/cybergym"
HF_METADATA_FILE = "tasks.json"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve"

SUBSET_TASK_IDS = (
    "arvo:47101",
    "arvo:3938",
    "arvo:24993",
    "arvo:1065",
    "arvo:10400",
    "arvo:368",
    "oss-fuzz:42535201",
    "oss-fuzz:42535468",
    "oss-fuzz:370689421",
    "oss-fuzz:385167047",
)
TASK_TYPES = ("arvo", "oss-fuzz")
DIFFICULTY_LEVELS = ("level0", "level1", "level2", "level3")
SCORING_MODES = ("final", "any")

Difficulty = Literal["level0", "level1", "level2", "level3"]
ScoringMode = Literal["final", "any"]

FILE_DESCRIPTIONS = {
    "repo-vul.tar.gz": "source code of the vulnerable program",
    "description.txt": "the description of the vulnerability",
    "error.txt": "the vulnerable program's output for the reference PoC",
    "repo-fix.tar.gz": "source code of the patched program",
    "patch.diff": "the patch commit diff",
}


@dataclass(frozen=True)
class CyberGymRecord:
    task_id: str
    project_name: str
    project_homepage: str
    project_main_repo: str
    project_language: str
    vulnerability_description: str
    task_difficulty: dict[str, list[str]]
    raw: dict = field(default_factory=dict, compare=False)

    @classmethod
    def from_dict(cls, value: dict) -> "CyberGymRecord":
        task_id = str(value["task_id"]).strip()
        task_type, separator, numeric_id = task_id.partition(":")
        if not separator or task_type not in TASK_TYPES or not numeric_id.isdigit():
            raise ValueError(f"Unsupported CyberGym task_id: {task_id!r}")

        difficulty = value.get("task_difficulty")
        if not isinstance(difficulty, dict):
            raise ValueError(f"CyberGym task {task_id!r} has no task_difficulty mapping")

        normalized_difficulty: dict[str, list[str]] = {}
        for level, file_paths in difficulty.items():
            if level not in DIFFICULTY_LEVELS or not isinstance(file_paths, list):
                continue
            normalized_difficulty[level] = [_validate_relative_data_path(str(path)) for path in file_paths]

        return cls(
            task_id=task_id,
            project_name=str(value.get("project_name", "")).strip(),
            project_homepage=str(value.get("project_homepage", "")).strip(),
            project_main_repo=str(value.get("project_main_repo", "")).strip(),
            project_language=str(value.get("project_language", "")).strip(),
            vulnerability_description=str(value.get("vulnerability_description", "")).strip(),
            task_difficulty=normalized_difficulty,
            raw=dict(value),
        )

    @property
    def task_type(self) -> str:
        return self.task_id.partition(":")[0]

    @property
    def numeric_id(self) -> str:
        return self.task_id.partition(":")[2]

    @property
    def task_dir_name(self) -> str:
        return f"cybergym_{self.task_id.replace(':', '_')}"

    @property
    def docker_registry(self) -> str:
        return "n132/arvo" if self.task_type == "arvo" else "cybergym/oss-fuzz"

    @property
    def vulnerable_image(self) -> str:
        return f"{self.docker_registry}:{self.numeric_id}-vul"

    @property
    def fixed_image(self) -> str:
        return f"{self.docker_registry}:{self.numeric_id}-fix"


def _validate_relative_data_path(raw_path: str) -> str:
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe CyberGym data path: {raw_path!r}")
    return path.as_posix()


def _resolved_hf_revision(downloaded_path: Path, requested_revision: str) -> str:
    # hf_hub_download returns .../snapshots/<commit>/tasks.json. Keep the
    # requested revision as a fallback for custom HF cache implementations.
    if downloaded_path.parent.parent.name == "snapshots":
        return downloaded_path.parent.name
    return requested_revision


def load_records(metadata_path: str | Path | None = None, revision: str = "main") -> tuple[list[CyberGymRecord], str]:
    """Load the small CyberGym tasks.json manifest, not the 240 GB payload."""
    if metadata_path is None:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(repo_id=HF_DATASET, filename=HF_METADATA_FILE, repo_type="dataset", revision=revision)
        )
        resolved_revision = _resolved_hf_revision(downloaded, revision)
        source_path = downloaded
    else:
        source_path = Path(metadata_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"CyberGym metadata file does not exist: {source_path}")
        resolved_revision = revision

    raw_records = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError(f"Expected a JSON list in {source_path}")
    return [CyberGymRecord.from_dict(value) for value in raw_records], resolved_revision


def select_records(
    records: Sequence[CyberGymRecord],
    *,
    subset: bool = True,
    task_ids: Sequence[str] | None = None,
    task_type: str | None = None,
    exclude_task_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[CyberGymRecord]:
    """Select records deterministically; explicit task_ids preserve caller order."""
    by_id = {record.task_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("CyberGym metadata contains duplicate task IDs")

    requested_ids: Sequence[str] | None = task_ids
    if requested_ids is None and subset:
        requested_ids = SUBSET_TASK_IDS

    if requested_ids is None:
        selected = list(records)
    else:
        missing = [task_id for task_id in requested_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"Unknown CyberGym task ID(s): {', '.join(missing)}")
        selected = [by_id[task_id] for task_id in requested_ids]

    if task_type is not None:
        if task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {TASK_TYPES}, got {task_type!r}")
        selected = [record for record in selected if record.task_type == task_type]

    excluded = set(exclude_task_ids or ())
    selected = [record for record in selected if record.task_id not in excluded]

    if limit is not None:
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        selected = selected[:limit]
    if not selected:
        raise ValueError("CyberGym task selection is empty")
    return selected


def _toml_string(value: str) -> str:
    # JSON basic strings are valid TOML basic strings for these metadata values.
    return json.dumps(value, ensure_ascii=False)


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class CyberGymTaskGenerator:
    def __init__(
        self,
        output_dir: Path,
        *,
        difficulty: Difficulty,
        scoring_mode: ScoringMode,
        dataset_revision: str,
        data_dir: Path | None = None,
        download_task_data: bool = False,
        agent_timeout_sec: float = 1200.0,
        verifier_timeout_sec: float = 180.0,
        auth_token_factory: Callable[[], str] | None = None,
    ) -> None:
        if difficulty not in DIFFICULTY_LEVELS:
            raise ValueError(f"difficulty must be one of {DIFFICULTY_LEVELS}, got {difficulty!r}")
        if scoring_mode not in SCORING_MODES:
            raise ValueError(f"scoring_mode must be one of {SCORING_MODES}, got {scoring_mode!r}")
        if agent_timeout_sec <= 0 or verifier_timeout_sec <= 0:
            raise ValueError("CyberGym timeouts must be positive")

        self.output_dir = output_dir
        self.difficulty = difficulty
        self.scoring_mode = scoring_mode
        self.dataset_revision = dataset_revision
        self.data_dir = data_dir.resolve() if data_dir else None
        self.download_task_data = download_task_data
        self.agent_timeout_sec = agent_timeout_sec
        self.verifier_timeout_sec = verifier_timeout_sec
        self.auth_token_factory = auth_token_factory or (lambda: secrets.token_hex(32))

    def _task_data_directives(self, record: CyberGymRecord, environment_dir: Path) -> str:
        file_paths = record.task_difficulty.get(self.difficulty)
        if not file_paths:
            raise ValueError(f"CyberGym task {record.task_id} has no files for {self.difficulty}")

        filenames = [PurePosixPath(file_path).name for file_path in file_paths]
        if len(set(filenames)) != len(filenames):
            raise ValueError(f"CyberGym task {record.task_id} has duplicate filenames for {self.difficulty}")

        if self.data_dir is not None or self.download_task_data:
            task_data_dir = environment_dir / "task_data"
            task_data_dir.mkdir(parents=True, exist_ok=True)
            for file_path, filename in zip(file_paths, filenames, strict=True):
                if self.data_dir is not None:
                    source = (self.data_dir / file_path).resolve()
                    if not source.is_relative_to(self.data_dir) or not source.is_file():
                        raise FileNotFoundError(f"CyberGym data file does not exist: {source}")
                else:
                    from huggingface_hub import hf_hub_download

                    source = Path(
                        hf_hub_download(
                            repo_id=HF_DATASET,
                            filename=file_path,
                            repo_type="dataset",
                            revision=self.dataset_revision,
                        )
                    )
                shutil.copy2(source, task_data_dir / filename)
            return "COPY task_data/ /workspace/task_data/"

        encoded_revision = quote(self.dataset_revision, safe="")
        directives = []
        for file_path, filename in zip(file_paths, filenames, strict=True):
            encoded_path = quote(file_path, safe="/")
            url = f"{HF_BASE_URL}/{encoded_revision}/{encoded_path}"
            directives.append(f"ADD {url} /workspace/task_data/{filename}")
        return "\n".join(directives)

    def _files_description(self, record: CyberGymRecord) -> str:
        file_paths = record.task_difficulty[self.difficulty]
        return "\n".join(
            f"- `{filename}`: {FILE_DESCRIPTIONS.get(filename, filename)}"
            for filename in (PurePosixPath(file_path).name for file_path in file_paths)
        )

    def _apptainer_task_data_sections(self, record: CyberGymRecord, environment_dir: Path) -> tuple[str, str]:
        file_paths = record.task_difficulty[self.difficulty]
        filenames = [PurePosixPath(file_path).name for file_path in file_paths]
        if self.data_dir is not None or self.download_task_data:
            file_directives = "\n".join(
                f"    task_data/{filename} /workspace/task_data/{filename}" for filename in filenames
            )
            return file_directives, "    true"

        encoded_revision = quote(self.dataset_revision, safe="")
        download_directives = []
        for file_path, filename in zip(file_paths, filenames, strict=True):
            encoded_path = quote(file_path, safe="/")
            url = f"{HF_BASE_URL}/{encoded_revision}/{encoded_path}"
            download_directives.append(
                f"    wget -qO {shlex.quote('/workspace/task_data/' + filename)} {shlex.quote(url)}"
            )
        return "", "\n".join(download_directives)

    def generate(self, record: CyberGymRecord, *, overwrite: bool = True) -> Path:
        task_dir = self.output_dir / record.task_dir_name
        if task_dir.exists() and not overwrite:
            raise FileExistsError(f"CyberGym Harbor task already exists: {task_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = self.output_dir / f".{record.task_dir_name}.tmp-{secrets.token_hex(6)}"
        environment_dir = staging_dir / "environment"
        task_server_dir = environment_dir / "task-server"
        tests_dir = staging_dir / "tests"
        solution_dir = staging_dir / "solution"
        for directory in (environment_dir, task_server_dir, tests_dir, solution_dir):
            directory.mkdir(parents=True, exist_ok=True)

        try:
            auth_token = self.auth_token_factory()
            if len(auth_token) < 16:
                raise ValueError("CyberGym task-server auth tokens must contain at least 16 characters")

            instruction = (TEMPLATE_DIR / "instruction.md").read_text(encoding="utf-8")
            instruction = instruction.replace("{files_description}", self._files_description(record))
            instruction = instruction.replace("{scoring_mode}", self.scoring_mode)
            (staging_dir / "instruction.md").write_text(instruction, encoding="utf-8")

            task_toml = (TEMPLATE_DIR / "task.toml").read_text(encoding="utf-8")
            replacements = {
                "{task_name}": _toml_string(f"sunblaze-ucb/{record.task_dir_name}"),
                "{task_id}": _toml_string(record.task_id),
                "{task_type}": _toml_string(record.task_type),
                "{project_language}": _toml_string(record.project_language.lower()),
                "{difficulty}": _toml_string(self.difficulty),
                "{scoring_mode}": _toml_string(self.scoring_mode),
                "{dataset_revision}": _toml_string(self.dataset_revision),
                "{agent_timeout_sec}": f"{self.agent_timeout_sec:.1f}",
                "{verifier_timeout_sec}": f"{self.verifier_timeout_sec:.1f}",
            }
            for placeholder, value in replacements.items():
                task_toml = task_toml.replace(placeholder, value)
            (staging_dir / "task.toml").write_text(task_toml, encoding="utf-8")

            dockerfile_name = "Dockerfile.arvo" if record.task_type == "arvo" else "Dockerfile.oss_fuzz"
            dockerfile = (TEMPLATE_DIR / "environment" / dockerfile_name).read_text(encoding="utf-8")
            dockerfile = dockerfile.replace("{vulnerable_image}", record.vulnerable_image)
            task_data_directives = self._task_data_directives(record, environment_dir)
            dockerfile = dockerfile.replace("{task_data_directives}", task_data_directives)
            (environment_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")

            for filename in (
                "docker-compose.yaml",
                ".dockerignore",
                "restrict-network.sh",
                "submit.sh",
            ):
                content = (TEMPLATE_DIR / "environment" / filename).read_text(encoding="utf-8")
                content = content.replace("{auth_token}", auth_token)
                destination = environment_dir / filename
                destination.write_text(content, encoding="utf-8")
                if filename.endswith(".sh"):
                    _make_executable(destination)

            task_server_dockerfile_name = "Dockerfile.arvo" if record.task_type == "arvo" else "Dockerfile.oss_fuzz"
            task_server_dockerfile = (
                TEMPLATE_DIR / "environment" / "task-server" / task_server_dockerfile_name
            ).read_text(encoding="utf-8")
            task_server_dockerfile = task_server_dockerfile.replace(
                "{vulnerable_image}", record.vulnerable_image
            ).replace("{fixed_image}", record.fixed_image)
            (task_server_dir / "Dockerfile").write_text(task_server_dockerfile, encoding="utf-8")
            shutil.copy2(TEMPLATE_DIR / "environment" / "task-server" / "task_server.py", task_server_dir)

            apptainer_template_name = (
                "Apptainer.agent.def.arvo" if record.task_type == "arvo" else "Apptainer.agent.def.oss_fuzz"
            )
            apptainer_definition = (TEMPLATE_DIR / "environment" / apptainer_template_name).read_text(encoding="utf-8")
            file_directives, download_directives = self._apptainer_task_data_sections(record, environment_dir)
            apptainer_definition = apptainer_definition.replace("{vulnerable_image}", record.vulnerable_image)
            apptainer_definition = apptainer_definition.replace("{task_data_file_directives}", file_directives)
            apptainer_definition = apptainer_definition.replace("{task_data_download_directives}", download_directives)
            (environment_dir / APPTAINER_AGENT_DEFINITION_FILENAME).write_text(apptainer_definition, encoding="utf-8")
            apptainer_manifest = {
                "schema_version": 1,
                "task_type": record.task_type,
                "vulnerable_image": record.vulnerable_image,
                "fixed_image": record.fixed_image,
                "auth_token": auth_token,
            }
            (environment_dir / APPTAINER_MANIFEST_FILENAME).write_text(
                json.dumps(apptainer_manifest, indent=2) + "\n", encoding="utf-8"
            )

            test_script = (TEMPLATE_DIR / "tests" / "test.sh").read_text(encoding="utf-8")
            test_script = test_script.replace("{auth_token}", auth_token).replace("{scoring_mode}", self.scoring_mode)
            (tests_dir / "test.sh").write_text(test_script, encoding="utf-8")
            _make_executable(tests_dir / "test.sh")
            shutil.copy2(TEMPLATE_DIR / "tests" / "verify.py", tests_dir)

            solve_script = (TEMPLATE_DIR / "solution" / "solve.sh").read_text(encoding="utf-8")
            solve_script = solve_script.replace("{auth_token}", auth_token)
            (solution_dir / "solve.sh").write_text(solve_script, encoding="utf-8")
            _make_executable(solution_dir / "solve.sh")

            if task_dir.exists():
                shutil.rmtree(task_dir)
            staging_dir.replace(task_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        return task_dir


def write_rollout_dataset(
    output_path: Path,
    records: Sequence[CyberGymRecord],
    *,
    difficulty: Difficulty,
    scoring_mode: ScoringMode,
    dataset_revision: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for record in records:
            row = {
                "instance_id": f"cybergym::{record.task_dir_name}",
                "responses_create_params": {"input": []},
                "cybergym_task_id": record.task_id,
                "cybergym_difficulty": difficulty,
                "cybergym_scoring_mode": scoring_mode,
                "cybergym_dataset_revision": dataset_revision,
                "project_name": record.project_name,
                "project_language": record.project_language,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    return output_path


def prepare_records(
    records: Sequence[CyberGymRecord],
    *,
    difficulty: Difficulty,
    scoring_mode: ScoringMode,
    dataset_revision: str,
    tasks_root: Path,
    output_path: Path,
    data_dir: Path | None = None,
    download_task_data: bool = False,
    overwrite: bool = True,
    auth_token_factory: Callable[[], str] | None = None,
) -> Path:
    level_dir = tasks_root / difficulty
    generator = CyberGymTaskGenerator(
        level_dir,
        difficulty=difficulty,
        scoring_mode=scoring_mode,
        dataset_revision=dataset_revision,
        data_dir=data_dir,
        download_task_data=download_task_data,
        auth_token_factory=auth_token_factory,
    )
    for index, record in enumerate(records, start=1):
        task_path = generator.generate(record, overwrite=overwrite)
        print(f"[{index}/{len(records)}] Prepared {record.task_id} at {task_path}")
    return write_rollout_dataset(
        output_path,
        records,
        difficulty=difficulty,
        scoring_mode=scoring_mode,
        dataset_revision=dataset_revision,
    )


def pull_runner_images(records: Iterable[CyberGymRecord], parallel: int = 4) -> None:
    """Optionally pre-pull the vulnerable and fixed runner images."""
    if parallel < 1:
        raise ValueError(f"pull_parallel must be at least 1, got {parallel}")
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("docker is required for --pull-images")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    images = [image for record in records for image in (record.vulnerable_image, record.fixed_image)]

    def pull(image: str) -> tuple[str, subprocess.CompletedProcess[str]]:
        result = subprocess.run(  # noqa: S603
            [docker, "pull", "--platform", "linux/amd64", image],
            check=False,
            capture_output=True,
            text=True,
        )
        return image, result

    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [executor.submit(pull, image) for image in images]
        for future in as_completed(futures):
            image, result = future.result()
            if result.returncode != 0:
                raise RuntimeError(f"Failed to pull {image}: {result.stderr.strip()}")
            print(f"Pulled {image}")


def prepare(
    difficulty: Difficulty = "level1",
    scoring_mode: ScoringMode = "final",
    subset: bool = True,
    task_ids: Sequence[str] | None = None,
    task_type: str | None = None,
    exclude_task_ids: Sequence[str] | None = None,
    limit: int | None = None,
    metadata_path: str | Path | None = None,
    revision: str = "main",
    data_dir: str | Path | None = None,
    download_task_data: bool = False,
    tasks_root: str | Path = DEFAULT_TASKS_ROOT,
    output_path: str | Path = OUTPUT_FPATH,
    overwrite: bool = True,
    pull_images: bool = False,
    pull_parallel: int = 4,
) -> Path:
    """Prepare task definitions and the JSONL consumed by NeMo Gym."""
    records, resolved_revision = load_records(metadata_path=metadata_path, revision=revision)
    selected = select_records(
        records,
        subset=subset,
        task_ids=task_ids,
        task_type=task_type,
        exclude_task_ids=exclude_task_ids,
        limit=limit,
    )
    result = prepare_records(
        selected,
        difficulty=difficulty,
        scoring_mode=scoring_mode,
        dataset_revision=resolved_revision,
        tasks_root=Path(tasks_root).expanduser().resolve(),
        output_path=Path(output_path).expanduser().resolve(),
        data_dir=Path(data_dir).expanduser().resolve() if data_dir else None,
        download_task_data=download_task_data,
        overwrite=overwrite,
    )
    validation_output_path = (
        VALIDATION_OUTPUT_FPATH
        if result == OUTPUT_FPATH.resolve()
        else result.with_name(f"{result.stem}_validation{result.suffix}")
    )
    write_rollout_dataset(
        validation_output_path,
        selected,
        difficulty=difficulty,
        scoring_mode=scoring_mode,
        dataset_revision=resolved_revision,
    )
    if pull_images:
        pull_runner_images(selected, parallel=pull_parallel)
    print(
        f"Wrote {len(selected)} CyberGym rollout row(s) to {result} "
        f"and validation alias {validation_output_path}"
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CyberGym for NeMo Gym through Harbor.")
    parser.add_argument("--difficulty", choices=DIFFICULTY_LEVELS, default="level1")
    parser.add_argument("--scoring-mode", choices=SCORING_MODES, default="final")
    parser.add_argument("--all", action="store_true", help="Prepare all 1,507 tasks instead of the 10-task subset.")
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--task-type", choices=TASK_TYPES)
    parser.add_argument("--exclude-task-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--metadata-path", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--download-task-data",
        action="store_true",
        help="Download selected task files now so evaluation can run in the current container without an image build.",
    )
    parser.add_argument("--tasks-root", type=Path, default=DEFAULT_TASKS_ROOT)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_FPATH,
        help=f"Rollout JSONL to write (use {TRAIN_OUTPUT_FPATH} for the RL config).",
    )
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--pull-images", action="store_true")
    parser.add_argument("--pull-parallel", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare(
        difficulty=args.difficulty,
        scoring_mode=args.scoring_mode,
        subset=not args.all,
        task_ids=args.task_ids,
        task_type=args.task_type,
        exclude_task_ids=args.exclude_task_ids,
        limit=args.limit,
        metadata_path=args.metadata_path,
        revision=args.revision,
        data_dir=args.data_dir,
        download_task_data=args.download_task_data,
        tasks_root=args.tasks_root,
        output_path=args.output_path,
        overwrite=not args.no_overwrite,
        pull_images=args.pull_images,
        pull_parallel=args.pull_parallel,
    )


if __name__ == "__main__":
    main()
