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
| launched | 2026-09-20 17:39 UTC, healthy after **1176s (19.6 min)** |
| KV pool | **427,631 tokens** — 1.63x a 262,144-token request |
| max model len | 262,144 |

## Decode is workload-dependent

The single "tok/s" figure in the upstream notes is prose. Measured here at temp 0
unless noted:

| workload | decode | draft acceptance | mean accept length |
|---|---|---|---|
| code (bench/ab-code.py, warm) | **53.5 tok/s** mean, 57.0 median | 65.5% | 1.97 |
| code (before MAMBA_SSM_CACHE_DTYPE=bfloat16) | 51.0 tok/s | 63.5% | 1.91 |
| prose (temp 0.7) | 30.9 tok/s | 41.1% | 2.23 |

The code rows come from `bench/ab-code.py`, five frozen prompts at temp 0 — a fixed
fixture, so arms are comparable across boots. The acceptance figures there are a
`/metrics` delta over the whole run and are NOT comparable to the older ad-hoc
88.6–91.7% numbers, which were single-request readings on different prompts.
**Discard the first run after a boot**: warm-up cost the first two prompts 860 ms and
2,780 ms TTFT against ~460 warm.

Earlier on this same container: 42.0 – 44.6 tok/s on code (82 – 88%, 3.47 – 3.66),
37.1 factual (~60%, 2.79), 26 – 30 prose (37 – 42%, 2.12 – 2.25). Nothing was
relaunched between the passes, so the gap is the prompts, not the configuration.

The client here is coding/agentic, so **the code row is the number that matters**.
Decode tracks draft acceptance almost linearly and acceptance is a property of the
workload: do not read a low prose figure as a regression, or a high code figure as
a win.

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

## Supervision (installed 2026-09-20)

Upstream's 24/7 supervisor (#41), installed as USER units with the paths pointed at
this checkout instead of the README's `~/qwen38-flash-next`:

| unit | state | |
|---|---|---|
| `qwen38-flash-supervisor.service` | **enabled, running** | the state machine: keeps the container and memwatch up, probes /health once a minute, 5 failures -> emergency stop + relaunch, 3 emergencies in 2 h -> breaker OPEN |
| `qwen38-flash-heartbeat.timer` | **enabled** | daily 09:00, unconditional |
| `qwen38-flash-heartbeat.service` | **fork-local, see below** | |
| `qwen38-flash-maintenance.timer` | **installed, DELIBERATELY NOT ENABLED** | see below |
| `qwen38-flash-supervisor-failure@.service` | installed | `OnFailure=` alert hook |

`loginctl enable-linger bharat` is on, so these start at boot with no login. The
supervisor adopted the running container rather than restarting it (`adopt_since` in
`logs/supervisor.state`, container `StartedAt` unchanged, `RestartCount=0`).

### Fork-local fix: the heartbeat timer had no service

Upstream ships `qwen38-flash-heartbeat.timer` with no `Unit=` and **no
`qwen38-flash-heartbeat.service`**, so the README's install line creates a timer that
fires into a unit that does not exist. That is the worst unit to lose silently — the
heartbeat exists precisely so that silence cannot be read as health. Added as
`systemd/qwen38-flash-heartbeat.service`, running `scripts/heartbeat.sh`. Verified:
`Result=success`, and it emits a real payload.

### The weekly maintenance relaunch is NOT enabled, on purpose

`maintenance-relaunch.sh` is `stop.sh -> start.sh -> smoke-test.sh` with **no memory
reclaim between them**. On a GB10 that cannot work unattended. Measured on this box
today, 2026-09-20: after a graceful `./stop.sh` of the ~96 GB container, MemFree was
**11 GiB**; only `fmem` — which reloads the NVIDIA kernel modules and needs root *and*
gdm stopped — brought it to **118 GiB**. vLLM gates its launch on free *physical*
memory, so the relaunch half of that window would fail.

Enabling the timer would therefore convert a healthy server into a Sunday-04:00 outage
that no automation on this box can end. Left installed so it is one `systemctl --user
enable` away if the reclaim problem is ever solved.

**The same limit applies to the supervisor's own crash recovery.** After any real
container death the leaked memory is stranded, so its relaunch attempts will fail and
it will alert rather than recover — 3 failures and the breaker opens. What it genuinely
delivers here:

- **autostart at boot** — the one gap we actually had. A reboot clears the leak, so the
  boot path is the case where its relaunch *does* work.
- detection, health probing and alerting, memwatch keepalive, shm cleanup, log rotation.

### Alerts are not yet wired

`ALERT_WEBHOOK` is unset in `.env`, so `alert.sh` logs to `logs/alert.log` and sends
nothing — silent-safe, but it means the supervisor can currently detect a failure and
tell no one. Set it to an ntfy topic (`https://ntfy.sh/<topic>`) to close that.
