# CyberGym in NeMo Gym

This integration runs [CyberGym](https://github.com/sunblaze-ucb/cybergym)
tasks through NeMo Gym's existing Harbor agent. It supports normal benchmark
rollout collection and preserves the multi-turn token IDs, log probabilities,
trajectory, and scalar reward required by NeMo-RL.

## Architecture

CyberGym is a benchmark/task environment, not a second model-facing agent:

```text
CyberGym task JSONL
    -> responses_api_agents/harbor_agent (Terminus2NemoGym)
    -> Harbor CyberGym environment
       -> Docker: agent container + private verifier sidecar
       -> Apptainer: stripped agent SIF + loopback trusted runner service
    -> CyberGym vulnerable/fixed runner check
    -> NeMo Gym response + reward + training token metadata
```

The task converter is based on Harbor's Apache-2.0 CyberGym adapter. NeMo Gym
currently pins an older Harbor revision whose Docker backend expects a task's
`docker-compose.yaml` to be complete, while the current upstream adapter emits
an override for Harbor's newer Compose merge behavior. The templates here are
self-contained so no Harbor dependency upgrade is required.

## Prerequisites and cost

- One supported runtime:
  - Docker with the Compose v2 plugin (the legacy `docker-compose` command is
    also detected), or
  - Apptainer/Singularity with fakeroot support, or prebuilt agent SIFs.
- An x86-64 host, or working `linux/amd64` emulation.
- The Harbor agent dependencies installed by the normal NeMo Gym server setup.
- Enough local disk for each selected task's vulnerable and fixed runner images.

Preparation downloads only CyberGym's small `tasks.json` manifest and creates
lightweight Harbor task definitions. Task source archives and runner images
are fetched during the first build for the selected runtime. The full CyberGym dataset is roughly
236-240 GB and the complete runner-image collection is roughly 10 TB, so start
with the default 10-task subset. One of those tasks,
`oss-fuzz:385167047`, is particularly large (roughly 100 GB according to the
Harbor adapter notes) and can be excluded for an initial smoke run.

## Prepare evaluation tasks

From the NeMo Gym repository root:

```bash
gym eval prepare --benchmark cybergym
```

This prepares the canonical 10-task subset at level 1 and writes:

- `benchmarks/cybergym/data/cybergym_benchmark.jsonl`
- `benchmarks/cybergym/data/cybergym_validation.jsonl` (legacy split alias)
- `benchmarks/cybergym/data/harbor_tasks/level1/cybergym_*/`

Each task directory contains both the self-contained Docker Compose definition
and an `Apptainer.agent.def` plus private `cybergym-apptainer.json` manifest.
Re-run preparation after upgrading an older prepared task tree so those files
are present. The Docker runtime no longer trusts the generated network topology,
so legacy task directories cannot re-enable old per-trial bridge networks.

The default is `final` scoring: only the last PoC submitted through
`/workspace/submit.sh` is graded. This follows CyberGym's current reporting
recommendation. The historical Harbor behavior remains available as `any`
scoring, where any submitted PoC may pass.

For a quick, smaller preparation or another difficulty:

```bash
uv run python benchmarks/cybergym/prepare.py \
  --task-ids arvo:1065 oss-fuzz:42535201

uv run python benchmarks/cybergym/prepare.py \
  --difficulty level3 \
  --exclude-task-ids oss-fuzz:385167047
```

When using a non-default difficulty or task root, point the agent at it:

```bash
export CYBERGYM_HARBOR_TASKS_DIR=benchmarks/cybergym/data/harbor_tasks/level3
```

Useful preparation flags include `--all`, `--task-ids`, `--task-type`,
`--exclude-task-ids`, `--limit`, `--scoring-mode`, `--data-dir`, and
`--pull-images`. Run `prepare.py --help` for the complete list.

## Evaluate

After running the preparation command above, the current CLI can start the
configured servers and collect the benchmark split in one command:

```bash
gym eval run \
  --benchmark cybergym \
  --model-type vllm_model \
  --model-url http://MODEL_HOST:PORT/v1 \
  --model-api-key EMPTY \
  --model MODEL_NAME \
  --split benchmark \
  --output results/cybergym/rollouts.jsonl \
  --limit 1
```

Remove `--limit 1` after the smoke task succeeds. Docker
`CYBERGYM_CONCURRENCY` defaults to 2; Apptainer
`CYBERGYM_APPTAINER_CONCURRENCY` defaults to 1 because its cold path prepares
three comparatively large SIFs.

Existing Hydra/legacy scripts work without modification. Append the CyberGym
benchmark and the model server config to `CONFIG_PATHS`; the deliberately
defined `validation` alias supports wrappers that hard-code
`++split=validation`:

```bash
CONFIG_PATHS="benchmarks/cybergym/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"

ng_e2e_collect_rollouts \
  "+config_paths=[$CONFIG_PATHS]" \
  "++split=validation" \
  "++output_jsonl_fpath=results/cybergym/rollouts.jsonl" \
  "++policy_base_url=http://MODEL_HOST:PORT/v1" \
  "++policy_api_key=EMPTY" \
  "++policy_model_name=MODEL_NAME"
```

The collated rows are routed to the outer `cybergym` server instance, which
internally uses `responses_api_agents/harbor_agent`. Each raw task ID has the
form `cybergym::cybergym_arvo_1065`.

### Evaluate with prestarted Compose containers

The Docker lifecycle above remains the default. To run the evaluator in an
unprivileged container, prepare the task normally and point it at services on
the evaluator's Compose network.

#### Existing `arvo_3938_*` and `arvo_47101_*` services

For a Compose project with `lab`, `arvo_3938_vul`, `arvo_3938_fix`,
`arvo_47101_vul`, and `arvo_47101_fix`, use the included
[docker-compose.external-arvo.yaml](docker-compose.external-arvo.yaml) as the
last Compose file. The original Compose file does not need to be edited. Its
Automodel `lab` image/build, NVIDIA runtime, command, entrypoint, volumes, and
user are preserved by Compose's per-service merge. Users enter that same
container and launch `gym eval run` from its CLI. The override:

- runs Harbor's agent and verifier commands directly in isolated per-trial
  directories inside `lab`;
- runs a small private HTTP adapter inside each raw ARVO container;
- runs that adapter as the image's UID/GID 1011 so it can safely replace and
  restore the runner-owned `/tmp/poc` file;
- uses each vulnerable service as the task-server and delegates the fixed run
  to its paired fixed service; and
- builds no task-specific image and starts no additional `main` service.

Set one private credential shared only by the vulnerable/fixed runner services,
then launch the existing Compose project. The runner containers may restart
until task preparation creates the server script in the shared data directory;
`lab` does not depend on their initial health.

```bash
export CYBERGYM_EXTERNAL_RUNNER_TOKEN="$(openssl rand -hex 32)"

docker compose \
  -f docker-compose.yml \
  -f docker-compose.production.yml \
  -f /path/to/nemo-gym/benchmarks/cybergym/docker-compose.external-arvo.yaml \
  up -d --build
```

From a terminal in `lab`, prepare the two tasks and download their source and
description files into the existing `/dli/task/data` bind mount:

```bash
cd /opt/nemo-gym
uv run python benchmarks/cybergym/prepare.py \
  --task-ids arvo:3938 arvo:47101 \
  --download-task-data \
  --tasks-root /dli/task/data/cybergym/harbor_tasks \
  --output-path /dli/task/data/cybergym/cybergym.jsonl
```

The vulnerable services read their verifier credentials directly from each
prepared task's `cybergym-apptainer.json`; task-specific secrets do not need to
be copied into the outer Compose environment. The override points
`CYBERGYM_DATASET_PATH` at the two-row JSONL above and exposes the same rows at
a separate `CYBERGYM_VALIDATION_DATASET_PATH` so Gym's per-split metrics
sidecars cannot collide. Confirm both routes from `lab`:

```bash
until curl -fsS http://arvo_3938_vul:9111/health; do sleep 1; done
until curl -fsS http://arvo_47101_vul:9111/health; do sleep 1; done
```

Run the normal evaluation from `lab` with `--limit 2`. The two requested tasks
are the first two rows of the checked-in benchmark split, and
`CYBERGYM_CONCURRENCY=2` routes them to their task-specific services. No Docker
socket or host ports are needed inside `lab`; Compose service-name DNS is the
only connection path. The local mode creates a separate task workspace and
tmux socket beneath each Harbor trial directory, so the two task rollouts do
not share agent files.

### Docker Compose diagnostic

Check the Docker backend before a long run:

```bash
docker compose version
```

If this prints Docker's top-level help or `unknown shorthand flag: 'p'`, the
Docker CLI is installed but Compose is not. Install the
[Docker Compose plugin](https://docs.docker.com/compose/install/linux/) or use
the Apptainer config below. The CyberGym Docker environment now performs this
check before building and gives a short actionable error; it also falls back to
`docker-compose` when that command is installed.

If Ray reports that the repository-root `pyproject.toml` is outside
`responses_api_agents/harbor_agent`, use the current NeMo Gym launcher. It sets
`RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` on spawned server processes while retaining
Harbor's explicit worker interpreter, so a manual shell export is no longer
required.

An old failed rollout with `agent_result: null`, `verifier_result: null`, and
`Failed to start tmux session. Error: None` never reached the policy model.
Current CyberGym images explicitly install the `script` PTY helper used by the
pinned Harbor revision, and the Docker backend uses non-TTY Compose exec while
preserving the real combined command output on failure. Re-run task preparation
after updating. If the result metadata still has `environment.import_path` set
to `null`, the run is using the old generic Docker config rather than
`CyberGymDockerEnvironment`.

Docker trials create zero user-defined networks. Harbor's generated CyberGym
tasks declare `allow_internet = true`, so the task-server uses Docker's
already-existing built-in `bridge` by default. The agent joins that same
namespace with `network_mode: service:task-server`, and their task-server
HTTP traffic stays on `127.0.0.1`. Set
`CYBERGYM_NETWORK_MODE=restricted` to select Docker's built-in `none`
network instead, or `public` to force `bridge`. Neither mode allocates a
per-trial Compose subnet.

The runtime Compose file bypasses firewall entrypoints inherited from prepared
agent images. Docker's built-in `bridge`/`none` choice is the
authoritative policy and keeps the main container alive on daemons where
in-container iptables setup is unavailable. `ALLOWED_HOSTS` is not
interpreted by this backend; Docker mode supports full egress or no egress.

The environment writes a canonical
`cybergym-runtime-compose.yaml` inside each Harbor trial directory and ignores
network sections in generated task files. In a failure message, the Compose
`-f` path should therefore point under
`results/cybergym/harbor_jobs/.../<trial>/cybergym-runtime-compose.yaml`.
If it instead points to
`benchmarks/cybergym/data/harbor_tasks/.../docker-compose.yaml`, the worker is
still executing an older `CyberGymDockerEnvironment` implementation.

### Evaluate with Apptainer

The Apptainer backend does not attempt to flatten CyberGym into one container.
It pulls separate vulnerable and fixed runner SIFs, builds a stripped agent SIF
from the generated definition, and starts a random loopback-only service that
invokes the two trusted runner images. The verifier token and ground-truth PoC
are never mounted into the agent SIF.

Confirm the runtime and select the Apptainer config in the same legacy command:

```bash
apptainer version

export CYBERGYM_APPTAINER_CACHE_DIR=/shared/cache/cybergym/apptainer
CONFIG_PATHS="benchmarks/cybergym/config_apptainer.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml"

ng_e2e_collect_rollouts \
  "+config_paths=[$CONFIG_PATHS]" \
  "++split=validation" \
  "++output_jsonl_fpath=results/cybergym/rollouts.jsonl" \
  "++policy_base_url=http://MODEL_HOST:PORT/v1" \
  "++policy_api_key=EMPTY" \
  "++policy_model_name=MODEL_NAME"
```

Auto-detection prefers `apptainer` and falls back to `singularity`. Set
`CYBERGYM_APPTAINER_EXECUTABLE` only when an explicit binary/path is needed.
The first rollout for a task pulls two runner SIFs and builds one agent SIF;
later rollouts reuse the content-keyed cache. Put
`CYBERGYM_APPTAINER_CACHE_DIR` on storage visible to every Ray worker to avoid
per-node rebuilds. Large builds may also require `APPTAINER_TMPDIR` and
`APPTAINER_CACHEDIR` on a filesystem with adequate space.

On clusters that do not permit fakeroot builds on compute nodes, build each
generated agent definition once on a build/login node and provide a path
template. Running a prebuilt SIF does not enable fakeroot by default:

```bash
task=cybergym_arvo_1065
task_env="benchmarks/cybergym/data/harbor_tasks/level1/$task/environment"
mkdir -p /shared/cybergym-agent-sifs
(cd "$task_env" && apptainer build --fakeroot \
  "/shared/cybergym-agent-sifs/$task.sif" Apptainer.agent.def)

export CYBERGYM_APPTAINER_AGENT_SIF='/shared/cybergym-agent-sifs/{task_name}.sif'
```

The vulnerable and fixed SIFs are still pulled into the configured cache. A
private registry can be configured with `APPTAINER_DOCKER_USERNAME` and
`APPTAINER_DOCKER_PASSWORD`.

## Throughput tuning

Rollout collection and Harbor have separate concurrency controls. For the
10-task subset on a sufficiently provisioned Docker cluster, run all tasks in
one wave with:

```bash
export CYBERGYM_CONCURRENCY=10

ng_e2e_collect_rollouts \
  "+config_paths=[$CONFIG_PATHS]" \
  "++split=validation" \
  "++num_samples_in_parallel=10" \
  "++output_jsonl_fpath=results/cybergym/rollouts.jsonl" \
  "++policy_base_url=http://MODEL_HOST:PORT/v1" \
  "++policy_api_key=EMPTY" \
  "++policy_model_name=MODEL_NAME"
```

For Apptainer, set `CYBERGYM_APPTAINER_CONCURRENCY=10` instead. The rollout
limit controls how many `/run` requests NeMo Gym dispatches, while the
runtime-specific environment variable controls how many Harbor trials may own
containers/SIFs concurrently. The effective concurrency is the lower of the
two. Values above the number of materialized rollouts do not help.

The zero-user-defined-network runtime applies even to old prepared task
directories, though
re-running `gym eval prepare --benchmark cybergym` remains recommended to keep
the generated artifacts consistent with the current templates.

Budget roughly two 2-CPU/4-GiB services per active Docker trial, plus image
build overhead and policy-model capacity. Ten steady-state trials therefore
need approximately 40 CPU cores and 80 GiB of host memory before allowing for
Docker, Ray, filesystem cache, and the model server. Ray spreads trial jobs
across available workers, so every eligible worker must have the selected
container runtime and access to the task and result paths.

Cold image preparation can dominate the first run and is primarily limited by
registry and filesystem bandwidth rather than inference. The generated Docker
and Apptainer recipes fetch the pinned tmux source through GitHub's source
endpoint using the builder or host trust store and verify its SHA-256 digest.
This avoids TLS failures in old CyberGym base images. Start cold Docker or
Apptainer caches at concurrency 2-4, then use 10 after images/SIFs are warm.
For Apptainer, place `CYBERGYM_APPTAINER_CACHE_DIR` on worker-visible
storage or use the prebuilt-agent-SIF mode above. The largest subset task can
consume roughly 100 GB by itself, so verify per-node scratch capacity before
enabling ten simultaneous cold builds.

Reducing `harbor_agent_kwargs.max_turns` below 100 or the 1,200-second agent
timeout can shorten individual trials, but changes the evaluation budget. Use
that only for smoke tests, not comparable benchmark results.

## Prepare a separate RL task pool

CyberGym does not publish a canonical train/evaluation split. Do not train on
the 10 reported evaluation tasks if you intend to compare benchmark results.
The opt-in `config_training.yaml` expects a separate
`cybergym_train.jsonl`. For example, create a deterministic 100-task pool from
the remaining tasks while retaining the prepared evaluation tasks as
validation:

```bash
uv run python benchmarks/cybergym/prepare.py \
  --all \
  --exclude-task-ids \
    arvo:47101 arvo:3938 arvo:24993 arvo:1065 arvo:10400 arvo:368 \
    oss-fuzz:42535201 oss-fuzz:42535468 oss-fuzz:370689421 oss-fuzz:385167047 \
  --limit 100 \
  --output-path benchmarks/cybergym/data/cybergym_train.jsonl

gym dataset collate \
  --config benchmarks/cybergym/config_training.yaml \
  --model-type vllm_model/vllm_model_for_training \
  --mode train_preparation \
  --output-dir data/cybergym
```

This produces `data/cybergym/train.jsonl` and
`data/cybergym/validation.jsonl`, including the
`cybergym_training` `agent_ref` expected by NeMo-RL.

For online GRPO, use these files as NeMo-RL's `data.train_jsonl_fpath` and
`data.validation_jsonl_fpath`, then embed the following NeMo Gym paths in the
training config:

```yaml
env:
  should_use_nemo_gym: true
  nemo_gym:
    use_absolute_ip: true  # Required when rollout workers span multiple nodes
    config_paths:
      - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
      - benchmarks/cybergym/config_training.yaml
```

For Apptainer training, replace the last path with
`benchmarks/cybergym/config_training_apptainer.yaml`. Both training variants
retain the same `cybergym_training` agent reference and trajectory/token
metadata.

The training vLLM config is required because it enables
`return_token_id_information`. The CyberGym config already uses
`Terminus2NemoGym` with `collect_rollout_details: true`, raw trajectories,
interleaved reasoning history, and summarization disabled. Set
`CYBERGYM_MAX_INPUT_TOKENS` and `CYBERGYM_MAX_OUTPUT_TOKENS` to the actual
policy limits, and tune Harbor concurrency and timeout values in the embedded
NeMo Gym config for the available runtime hosts.

See `responses_api_agents/harbor_agent/README.md` for the existing NeMo-RL
failure handling, multi-node settings, and on-policy multi-turn token notes.

## Verification and artifacts

An agent can test candidate raw-input files with
`bash /workspace/submit.sh PATH`. The verifier accepts a PoC only when it
crashes the vulnerable runner but not the patched runner. Timeouts and OOM
kills do not count as crashes. With Docker, the task server containing both
binaries and the ground-truth PoC shares a per-trial network namespace with the
agent and exposes its API only on loopback. With Apptainer, those assets stay in
separate host-controlled SIFs behind a loopback service. Protected verification
and oracle endpoints use a per-task token that Harbor exposes only after the
agent phase.

NeMo Gym writes rollout JSONL to the requested output path. Harbor's detailed
`result.json`, ATIF `trajectory.json`, verifier logs, submitted PoCs, and
terminal artifacts are grouped under `results/cybergym/harbor_jobs` by date,
dataset, and model. Override this with `CYBERGYM_HARBOR_JOBS_DIR`.

## Security

CyberGym executes attacker-controlled inputs against real vulnerable binaries.
Run it only on a host you control. Never publish the task server or Docker
network ports. The canonical tasks allow agent egress by default, matching the
upstream benchmark contract. Set `CYBERGYM_NETWORK_MODE=restricted` to
remove external routing for a locked-down run, understanding that this can
change task behavior. This integration does not implement a hostname allowlist.

The generic Singularity backend by itself manages one main container and is not
used directly for CyberGym. The CyberGym-specific Apptainer environment adds
the separate trusted runner service required to preserve the verifier boundary.

Apptainer does not provide unprivileged per-container network filtering in this
integration. Its agent SIF shares the host network namespace so it can reach the
loopback task service. On a shared cluster, enforce egress and cross-job network
isolation at the scheduler/node level. The verifier and oracle endpoints still
require a per-task bearer token, and runner SIFs are never mounted into the
agent container.

CyberGym configs enable strict Harbor failure handling: a missing container
runtime, failed image mount/build, unreachable runner, or missing verifier
result aborts the request instead of generating a synthetic reward of zero.
This is intentional for RL, where infrastructure failures must not become
negative training labels.
