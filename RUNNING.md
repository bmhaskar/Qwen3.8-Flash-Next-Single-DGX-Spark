# What is actually running on `spark-bharat`

A fork of [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark).
All of the code here is Mia's. This fork adds exactly three files, to answer a
question upstream deliberately leaves open: *which configuration is serving right
now, on this box?*

Upstream's `.gitignore` is an allowlist that excludes `.env` and `.last_launch.sh`,
which is correct for a general recipe and useless for reproducing a specific
known-good server. This fork re-includes them, scrubbed.

## The server

| | |
|---|---|
| container | `vllm-fn-tp1` |
| image | `vllm/vllm-openai:qwen38-flash-next` |
| checkpoint | `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` |
| served as | `qwen3.8-flash-next` |
| port | **8888** (kept so the pi client's config is unchanged) |
| launched | 2026-09-07 22:27:38 UTC, healthy after **~18 min** |
| KV pool | **389,398 tokens** — 1.49x a 262,144-token request |
| max model len | 262,144 |

## Decode is workload-dependent

The single "tok/s" figure in the upstream notes is prose. Measured here at temp 0
unless noted:

| workload | decode | draft acceptance | mean accept length |
|---|---|---|---|
| code | **42.0 – 44.6 tok/s** | 82 – 88% | 3.47 – 3.66 |
| factual | 37.1 tok/s | ~60% | 2.79 |
| prose (temp 0.7) | 26 – 30 tok/s | 37 – 42% | 2.12 – 2.25 |

The client here is coding/agentic, so **42 – 44.6 tok/s is the number that matters**.
Do not read a low prose figure as a regression.

## Launch with `.last_launch.sh`, not `./start.sh`

`.last_launch.sh` is the literal `docker run` that produced the measured-good server
above. It was generated on upstream commit `ef1af5f`; this tree sits on `78b0675`,
and `start.sh` changed by **139 lines** in between (the `ABLIT=0/1` gated-checkpoint
feature, and checkpoint resolution moving from alphabetical to state-based). That
drift is untested on this box. `.env` is likewise older than the current
`.env.sample` — it predates `ABLIT`, `EXTRA_DOCKER_ARGS`, and `MAMBA_SSM_CACHE_DTYPE`.

Reconciling those is a deliberate, measured relaunch, not something to do in passing.
Until then the launch script bypasses all of it.

## `HF_TOKEN`

The launch script carries `-e HF_TOKEN=${HF_TOKEN}` and `/.env` keeps `HF_TOKEN`
commented out. **No literal token belongs in this repo.** `HF_HUB_OFFLINE=1` is set,
so serving an already-downloaded checkpoint needs no token at all; export one in your
shell only if you are fetching weights.

## Before any relaunch: the GB10 reclaim ritual

Tearing down a large-CUDA process on this box leaks its unified memory — all of it,
on a clean exit, not just on `kill -9`. vLLM needs ~100 GiB of free *physical* memory
to load, so a relaunch without reclaiming will fail. Run these **one line at a time**,
not joined with `&&` (zsh reads a leading `!` as negation and silently skips `fmem`):

```
sudo systemctl stop gdm
fmem
sudo systemctl start gdm
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
```

The CDI regen is mandatory. `fmem` reloads the driver and bumps the `nvidia-uvm`
major; a stale `/var/run/cdi/nvidia.yaml` then produces a misleading
`CUDA unknown error` *inside* the container while the host's
`torch.cuda.is_available()` is still True.

Verify before launching:

```
grep -E 'MemFree' /proc/meminfo                      # want ~100 GiB+
grep -oE 'major: [0-9]+' /var/run/cdi/nvidia.yaml    # must match:
grep nvidia-uvm /proc/devices
```

## Run it under `tmux`

The box is dual-homed on one subnet and the SSH pipe from the client drops without
warning. On 2026-09-07 a `stop.sh` was killed mid-flight by a broken pipe *after* it
archived a still-running container's log but *before* its `docker rm -f` — one step
from force-removing a healthy server and paying the ~20-minute reboot above. A
20-minute model boot should never be hostage to the SSH link.
