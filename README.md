# QoS-aware orchestration of local, edge and cloud LLMs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23093412.svg)](https://doi.org/10.5281/zenodo.23093412)

Code, measurement logs and analysis for the paper *QoS-aware orchestration of local and cloud large language models
in distributed information systems* (O. Kholodniak).

Each request can run on a small model on the device (LOCAL), a larger model on a LAN server (EDGE) or a large model
behind a cloud API (CLOUD). The repository contains

- an end-to-end latency model that combines round trips, request/response size, bandwidth and TCP slow start with the
  prefill and decode speed of each model and token streaming (`netmodel.py`, `policy.predict_latency`);
- a constrained selection rule: the cheapest target whose expected quality and predicted latency meet the class
  requirements, with a utility fallback (`policy.qos_aware`), and the baselines it is compared with;
- a two-router testbed that emulates LAN and WAN paths with Linux `tc netem` (`netem/`);
- measurement, scoring and trace-driven evaluation scripts, the raw logs of all runs and the derived tables and figures.

Inference times, token counts, response sizes, API costs and quality scores are measured on real requests. Network
conditions are emulated. The policy comparison replays the measured requests under emulated network scenarios and is
checked against live requests (E4).

## Layout

```
src/qos_orchestration/   library: latency model, policies, simulation, clients, scoring, testbed control
  ifeval/                IFEval instruction checkers (Apache-2.0)
scripts/                 experiments (E1, E2, E4), scoring and analysis
netem/                   Docker image and scripts of the emulated network
configs/                 experiment settings and the recorded hardware/software environment
data/                    sampled tasks and dataset revisions
results/raw/             raw logs of every request (high_load/: runs under background host load)
results/                 scored records, summary.json, CSV tables
figures/                 figures of the paper
tests/                   unit tests
```

## Setup

Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m nltk.downloader punkt_tab      # sentence tokenizer used by the IFEval checkers
pytest
```

## Reproducing the tables and figures from the published logs

```bash
python scripts/analyze_netbench.py      # E1: network model vs. measured transfers -> results/netbench_*.json
python scripts/analyze.py               # results/summary.json, results/*.csv, figures/fig3-fig5
python scripts/draw_diagrams.py         # figures/fig1-fig2
python scripts/fetch_tasks.py           # re-draws the task sample and checks it against data/tasks.jsonl
```

Re-scoring the logs runs the HumanEval tests in a container without network and needs Docker and the `qos-net:1`
image: `docker build -t qos-net:1 netem/ && python scripts/score.py`.

## Running new measurements

The published runs used an Apple M1 Pro (16 GB, macOS 14.5), Ollama 0.34.3 with `qwen2.5:3b-instruct` and
`qwen2.5:7b-instruct` (Q4_K_M), Docker on Colima, and `qwen/qwen-2.5-72b-instruct` on OpenRouter with the provider
pinned to DeepInfra and fallbacks disabled. Exact versions and model digests are in `configs/environment.txt`.

Prerequisites:

- Ollama running on the host with both models pulled. With Colima, `host.docker.internal` reaches the host's
  loopback interface. With other Docker set-ups, start Ollama with `OLLAMA_HOST=0.0.0.0` and set `EDGE_HOST` to an
  address of the host that containers can reach.
- An OpenRouter key in `~/.config/qos-llm/openrouter.key` (mounted read-only into the client container).

Measurements write to `results/` by default. Point `QOS_RESULTS` at a new directory so that a new run does not mix
with the published logs; all scripts, including scoring and analysis, use the same variable.

```bash
export QOS_RESULTS="$PWD/runs/$(date +%Y%m%d)"
export QOS_FIGURES="$QOS_RESULTS/figures"
ollama pull qwen2.5:3b-instruct && ollama pull qwen2.5:7b-instruct
docker build -t qos-net:1 netem/
netem/topology.sh up                    # client container + LAN and WAN routers; prints the edge address
export EDGE_HOST=<address printed by topology.sh>

python scripts/echo_server.py &          # E1 target on the host
python scripts/netbench.py               # E1: transfers over emulated links
python scripts/characterize.py           # E2: every task on every target over the base paths
python scripts/live_check.py             # E4: live runs under emulated impairments
python scripts/fetch_provider_stats.py   # provider-side generation times of the cloud requests
python scripts/score.py && python scripts/score.py --pattern "live_*.jsonl" --output scored_live.jsonl
python scripts/analyze_netbench.py && python scripts/analyze.py
netem/topology.sh down
```

The scripts resume a run: tasks with a successful record are skipped and failed ones are retried. Local and edge
targets run as separate blocks, and every request logs the host load average and the models loaded by Ollama, because
background load and model reloads distort timings on a shared machine.

## Data and licenses

The experiments use the first 25 tasks of each class in `data/tasks.jsonl`, sampled from NQ-open, GSM8K, HumanEval and
IFEval. The code is MIT-licensed; vendored code and dataset-derived content keep their own licenses, listed in
`THIRD_PARTY_NOTICES.md`.
