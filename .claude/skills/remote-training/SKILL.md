---
name: remote-training
description: Manages remote training infrastructure on Nebius Serverless (Jobs for convert/train, Endpoints for inference). Use for dataset conversion, training jobs, serving checkpoints, and end-to-end pipeline validation. No VM lifecycle to manage.
---

# Remote Training Infrastructure (Nebius Serverless)

This skill runs the Positronic convert → train → serve pipeline on
[Nebius Serverless](https://docs.nebius.com/serverless): **Jobs** for batch work
(dataset conversion, training) and **Endpoints** for HTTP inference servers. Same
containers and scripts as before — no VM to provision, SSH into, or shut down, and
no idle compute cost.

All operations go through the wrapper scripts in `workflows/nebius/`. Run them from
the repo root. The full reference is `workflows/nebius/README.md`.

## Vendor tokens

Every script takes a `<vendor>` positional that selects the container image and `uv`
extra. Supported: `lerobot_0_3_3` (ACT), `lerobot` (SmolVLA), `openpi`, `gr00t`.

| Vendor | Image | Train/serve hardware |
|--------|-------|----------------------|
| `lerobot_0_3_3` (ACT) | `positro/positronic:latest` | H100 |
| `lerobot` (SmolVLA) | `positro/positronic:latest` | H100 |
| `openpi` | `positro/openpi:latest` | H100 |
| `gr00t` | `positro/gr00t:latest` | H100 |

Conversion always runs on CPU (`cpu-e2`, `8vcpu-32gb`); train/serve on
`gpu-h100-sxm` (`1gpu-16vcpu-200gb`). `openpi`/`gr00t` re-use the `lerobot_0_3_3`
converter with their own codec namespace.

## S3 Convention

```
s3://interim/{dataset}/{vendor}/{codec}/                    — converted LeRobot datasets
s3://checkpoints/{dataset}/{vendor}/{codec_or_experiment}/  — training output
s3://inference/{dataset}/{date_or_exp}/{vendor}/            — inference eval results
```

Every run writes `run_metadata_*.yaml` with the full CLI command — read it to
reconstruct the pipeline.

### Current Artifacts

**sim_stack** (automated testing — `@positronic.cfg.ds.phail.sim_stack_cubes`):

| Vendor | Codec | Interim | Latest Checkpoint |
|--------|-------|---------|-------------------|
| gr00t | `ee_rot6d` | `s3://interim/sim_stack/groot/ee_rot6d/` | `s3://checkpoints/sim_stack/groot/ee_rot6d/230226/` |
| lerobot (0.4.x) | `ee` | `s3://interim/sim_stack/lerobot_04/ee/` | `s3://checkpoints/sim_stack/lerobot_04/smolvla_150k/` |
| lerobot_0_3_3 | `ee` | `s3://interim/sim_stack/lerobot/ee/` | `s3://checkpoints/sim_stack/lerobot/230226-ee/` |
| openpi | `ee` | `s3://interim/sim_stack/openpi/ee/` | `s3://checkpoints/sim_stack/openpi/ee/pi05_positronic_lowmem/230226/` |

**phail_unified** (production — `@positronic.cfg.ds.phail.phail_unified`):

| Vendor | Codec | Interim | Latest Checkpoint |
|--------|-------|---------|-------------------|
| smolvla (0.4.x) | `ee` | `s3://interim/phail_unified/smolvla/ee/` | `s3://checkpoints/phail_unified/smolvla/150315_ft_150k/` |
| gr00t | `ee_rot6d` | `s3://interim/phail_unified/groot/ee_rot6d/` | — |
| lerobot (0.4.x) / SmolVLA | `ee` | — | `s3://checkpoints/phail_unified/smolvla/170316_ee/` |
| lerobot_0_3_3 | `ee` | `s3://interim/phail_unified/lerobot/ee/` | — |
| openpi | `ee` | `s3://interim/phail_unified/openpi/ee/` | — |

## Docker Images

Serverless jobs/endpoints pull `positro/<vendor>:${NEBIUS_IMAGE_TAG:-latest}` from the
registry — they do **not** mount local source, so any code change needs a rebuild + push
before it takes effect remotely.

**Tag gotcha:** locally `make push-*` pushes `:<branch>` and `:<sha>` but **not** `:latest`
(that only happens under `CI`). So after a code change, either:
- `cd docker && CI=1 make push-<x>` — updates `:latest` (what the workflow pulls by default), or
- `cd docker && make push-<x> IMAGE_TAG=<branch>` then run the workflow with
  `NEBIUS_IMAGE_TAG=<branch>` — tests a branch build without clobbering `:latest`.

Plain `make push-<x>` with no `CI`/`IMAGE_TAG`/`NEBIUS_IMAGE_TAG` leaves serverless
running the **old** `:latest` image — the change silently won't take effect.

| Image | Source | Used For |
|-------|--------|----------|
| `positro/positronic` | `positronic/docker/` | Conversion, lerobot / SmolVLA train+serve |
| `positro/openpi` | `positronic/docker/` (depends on `positro/openpi-base`) | OpenPI train+serve, openpi stats |
| `positro/gr00t` | `positronic/docker/` (depends on `positro/gr00t-base`) | GR00T train+serve |

```bash
cd docker
make push-training  # positro/positronic
make push-openpi    # positro/openpi (rebuild positro/openpi-base first if ../openpi changed)
make push-groot     # positro/gr00t (rebuild positro/gr00t-base first if ../gr00t changed)
make push           # all images
```

Cross-repo base rebuilds: `cd ../openpi/docker && make push` (or `../gr00t/docker`),
then `cd ../positronic/docker && make push-openpi`. See `docker/CONTEXTS.md`.

## One-time setup

The pipeline reads credentials from Nebius MysteryBox secrets and uses a shared
filesystem for `uv`/HF/openpi caches. This is already provisioned for the
Positronic-internal project. To (re)create it for a different project, follow
"One-time setup" in `workflows/nebius/README.md` (four MysteryBox secrets +
one `network_ssd` filesystem).

Defaults point at the Positronic-internal project; override via env when needed:

| Variable | Default | Purpose |
|---|---|---|
| `NEBIUS_PARENT_ID` | `project-e00f38wexevrr52b8j` | Project to create jobs/endpoints in |
| `NEBIUS_SUBNET_ID` | `vpcsubnet-e00pk1j1x6hjmr4m92` | VPC subnet |
| `WANDB_SECRET` | `positronic-serverless-wandb-api-key` | MysteryBox name for WandB key. Set empty to disable wandb. |
| `NEBIUS_CACHE_FS` | `computefilesystem-e00f6jyfr5wkawyrab` | Shared cache filesystem **ID** (mounted RW at `/cache`) |

## Pipeline

### 1. Convert Dataset

`convert.sh` runs the right converter + codec for the vendor as a CPU Job. For
`openpi` it **blocks until convert finishes, then chains a stats job** and prints
the `--stats_path` to use for training.

```bash
bash workflows/nebius/convert.sh lerobot_0_3_3 \
  --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.lerobot_0_3_3.codecs.ee \
  --output_dir=s3://interim/sim_stack/lerobot/ee/

bash workflows/nebius/convert.sh openpi \
  --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.openpi.codecs.ee \
  --output_dir=s3://interim/sim_stack/openpi/ee/
# → also submits openpi-stats-* ; note the printed --stats_path=<...>/stats/assets/
```

Default codecs: gr00t `ee_rot6d`, lerobot `ee`, openpi `ee`,
lerobot_0_3_3 `ee`.

### 2. Train

`train.sh` runs `python -m positronic.vendors.<vendor>.train` as an H100 Job. The
dataset bucket is mounted read-only via Mountpoint-S3 at `/mnt/input` for
`lerobot_0_3_3`; other vendors stream via `pos3` from the `s3://` path directly.
`--output_dir` / `--output_path` stays an `s3://` URL (handled by `pos3`).

Read each vendor's `positronic/vendors/<vendor>/train.py` docstring for its exact
flags. `--resume=true` resumes an interrupted run.

```bash
# ACT / SmolVLA
bash workflows/nebius/train.sh lerobot_0_3_3 \
  --input_path=s3://interim/sim_stack/lerobot/ee/ \
  --exp_name=act_sim_stack_v1 \
  --output_dir=s3://checkpoints/sim_stack/lerobot/ \
  --num_train_steps=50000 --save_freq=10000

# OpenPI — needs --stats_path from the convert step's chained stats job
bash workflows/nebius/train.sh openpi \
  --input_path=s3://interim/sim_stack/openpi/ee/ \
  --stats_path=s3://interim/sim_stack/openpi/stats/assets/ \
  --output_path=s3://checkpoints/sim_stack/openpi/ \
  --exp_name=pi_sim_stack_v1 \
  --num_train_steps=30000
# openpi checkpoint lands at <output_path>/pi05_positronic_lowmem/<exp_name>/
```

The first job after a dependency change pays the full `uv`/HF cold-download
(~10 min); later jobs reuse `/cache` and start faster.

### 3. Serve a Checkpoint

`serve.sh <vendor> <unique-endpoint-name> [server args...]` creates a public
Endpoint on H100 port 8000, blocks until a public IP is allocated, and prints a
banner with the URL + teardown command. The container then takes ~10–15 min more
to `uv sync` and load the model.

```bash
# Public ACT demo checkpoint (no S3 creds needed inside container)
bash workflows/nebius/serve.sh lerobot_0_3_3 my-act-demo demo

# Your own checkpoint (subcommand differs per vendor)
bash workflows/nebius/serve.sh lerobot_0_3_3 act-server serve \
  --checkpoints_dir=s3://checkpoints/sim_stack/lerobot/230226-ee/

bash workflows/nebius/serve.sh openpi pi-server serve \
  --checkpoints_dir=s3://checkpoints/sim_stack/openpi/ee/pi05_positronic_lowmem/230226/

bash workflows/nebius/serve.sh gr00t groot-server ee_rot6d \
  --checkpoints_dir=s3://checkpoints/sim_stack/groot/ee_rot6d/230226/
```

Sanity-check once warm:

```bash
curl http://<endpoint-ip>:8000/api/v1/models   # → {"models": ["<step>"]}
```

Tear down (releases compute + public IP):

```bash
bash workflows/nebius/stop.sh my-act-demo
```

To pause without releasing the static IP: `nebius ai endpoint stop <id>`
(`start` resumes).

### 4. Run Inference Client

Unchanged — point the existing CLI at the endpoint IP:

```bash
uv run positronic-inference sim \
  --policy=.remote --policy.host=<endpoint-ip> --policy.port=8000 \
  --output_dir=s3://inference/sim_stack_validation/<run_name>/<vendor>/
```

View results locally (top-level dir compares multiple runs):

```bash
uv run python -m positronic.cfg.eval sim \
  --dataset.base.path=s3://inference/sim_stack_validation/<run_name> --reset_cache --https
# http://localhost:5001
```

## End-to-End Validation

`e2e.sh` runs the whole pipeline for one vendor (convert → train 200 steps → serve
→ `/api/v1/models` smoke → teardown), polling Nebius and printing a per-stage
status line. ~$2–5 per run. Use it to verify a vendor still works after an
image/dependency/script change.

```bash
bash workflows/nebius/e2e.sh openpi

# All four — seed the cache with one first, then fan out warm
bash workflows/nebius/e2e.sh lerobot_0_3_3
for v in lerobot openpi gr00t; do bash workflows/nebius/e2e.sh "$v" & done; wait
```

Override via env: `E2E_S3_BASE`, `E2E_EXP_NAME`, `E2E_LOG_ROOT`, `E2E_DATASET`.

## Monitoring Jobs & Endpoints

```bash
# Jobs
nebius ai job get  <aijob-id>                 # state: PROVISIONING/STARTING/RUNNING/COMPLETED/FAILED
nebius ai job logs <aijob-id> --follow
nebius ai job list --parent-id "$NEBIUS_PARENT_ID" --format json | jq '.items[].metadata.name'

# Endpoints
nebius ai endpoint get  <endpoint-id>
nebius ai endpoint logs <endpoint-id> --follow   # wait for "INFO Started server process"
nebius ai endpoint list --parent-id "$NEBIUS_PARENT_ID" --format json
```

The `create` call streams the job ID and ready-to-paste follow-up commands.

## Common Issues

- **Job stuck in PROVISIONING/STARTING**: normal — image pull + `uv` resolve.
  First run on a cold cache is ~10 min; check `nebius ai job logs <id> --follow`.
- **OpenPI train can't find stats**: pass `--stats_path=<output_dir-sibling>/stats/assets/`
  exactly as printed by `convert.sh openpi`. Stats must be a *sibling* of the
  dataset dir (pos3 forbids upload-inside-download).
- **gr00t with read-only input mount fails**: only `lerobot_0_3_3` uses the RO
  Mountpoint-S3 mount; gr00t writes back into the dataset dir, so it streams via
  `pos3` from the `s3://` path instead (handled automatically by `train.sh`).
- **Endpoint name collision**: names must be unique in the project. Pick a fresh
  name or `stop.sh` the old one.
- **Cold cache + parallel fan-out**: 4 jobs racing to populate `/cache` thrash;
  seed with one vendor, then fan out the rest warm (see `e2e.sh` header).
- **Wipe the shared cache**: throwaway `busybox` job mounting the FS — see
  "Cleaning the shared cache" in `workflows/nebius/README.md`.

### Nebius Auth (Headless)

1. `nebius --no-browser --auth-timeout 5m iam whoami 2>&1` — extract auth URL
2. User clicks URL, browser redirects to `http://127.0.0.1:PORT/?code=XXX&state=YYY`
3. `curl -s "http://127.0.0.1:PORT/?code=XXX&state=YYY"` on the machine running nebius
4. Auth completes, scripts work
