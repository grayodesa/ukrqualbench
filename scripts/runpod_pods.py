"""Create, inspect and delete the RunPod pods that serve self-hosted models over vLLM.

One pod per model, image ``vllm/vllm-openai``, endpoint behind the RunPod HTTP proxy
(``https://<pod_id>-8000.proxy.runpod.net``). Keys are read from ``.env`` and never
printed:

    UKRQUALBENCH_RUNPOD_API_KEY  RunPod REST API key
    UKRQUALBENCH_VLLM_API_KEY    key every vLLM server requires (written by ``keygen``)

Usage (from the repository root):

    uv run python scripts/runpod_pods.py keygen   # once: create the vLLM key in .env
    uv run python scripts/runpod_pods.py create [MODEL ...]  # create pods, sync ENDPOINTS
    uv run python scripts/runpod_pods.py status              # pod state, /v1/models probe
    uv run python scripts/runpod_pods.py switch OLD NEW      # restart OLD's pod with NEW
    uv run python scripts/runpod_pods.py delete [MODEL ...]  # terminate pods, sync ENDPOINTS

Without MODEL names ``create`` covers every model not yet in the state file and ``delete``
every pod in it. UKRQUALBENCH_VLLM_ENDPOINTS always mirrors the state file.

Pods bill per second from creation until ``delete``; stopping is not enough.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
STATE_FILE = ROOT / "results" / "runpod_pods.json"  # results/ is gitignored

API = "https://rest.runpod.io/v1"
IMAGE = "vllm/vllm-openai:v0.31.0"
PORT = 8000
# The client always asks for up to 8192 new tokens (models/base.py default_max_tokens),
# so the context must hold the prompt on top of that.
MAX_MODEL_LEN = 16384

RUNPOD_KEY_VAR = "UKRQUALBENCH_RUNPOD_API_KEY"
VLLM_KEY_VAR = "UKRQUALBENCH_VLLM_API_KEY"
ENDPOINTS_VAR = "UKRQUALBENCH_VLLM_ENDPOINTS"

GPU_48GB = ["NVIDIA A40", "NVIDIA RTX A6000"]
GPU_80GB = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]


@dataclass(frozen=True)
class ServedModel:
    """A model served by vLLM under ``served`` (the part after ``vllm:``)."""

    served: str
    repo: str
    revision: str  # pinned to the revision whose tokenizer is in data/fertility/registry.json
    gpu_ids: list[str]
    container_disk_gb: int


MODELS = [
    ServedModel(
        "lapa-v0.1.3-instruct",
        "lapa-llm/lapa-v0.1.3-instruct",
        "66ba3ebfa6baa17bc2b3222f3908c30152ae104f",
        GPU_48GB,
        80,
    ),
    ServedModel(
        "mamaylm-gemma-3-12b-it-v2.0",
        "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0",
        "3cb02013b6b37cfe10f4b3da0fcddc1acef7f244",
        GPU_48GB,
        80,
    ),
    ServedModel(
        "mamaylm-gemma-3-27b-it-v2.0",
        "INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0",
        "ad1bdfed9646939358adfcba9c88bb4031b50ed7",
        GPU_80GB,
        120,
    ),
    # Previous versions in BF16: the registry entries without the vllm: prefix are Q4 GGUF
    # runs in LM Studio, so a version comparison needs these on the same stack.
    ServedModel(
        "lapa-v0.1.2-instruct",
        "lapa-llm/lapa-v0.1.2-instruct",
        "10116908ee519a816cfcb4b4e90668ba8e0710a8",
        GPU_48GB,
        80,
    ),
    ServedModel(
        "mamaylm-gemma-3-12b-it-v1.0",
        "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0",
        "2d4fd7a65bef3534463a2abddeefff2c70fa0e83",
        GPU_48GB,
        80,
    ),
    # Official FP8 (compressed-tensors, E4M3 weights) build of the 27B; on Ampere vLLM runs
    # it weight-only. Meant for `switch` on the BF16 pod: same machine, BF16 vs 8-bit.
    ServedModel(
        "mamaylm-gemma-3-27b-it-v2.0-fp8",
        "INSAIT-Institute/MamayLM-Gemma-3-27B-IT-v2.0-FP8-dynamic",
        "7ef42c6e4991864f0f47bfd524c1e77c9bdb11a9",
        GPU_80GB,
        120,
    ),
]
MODEL_NAMES = [m.served for m in MODELS]


# ---------------------------------------------------------------------------
# .env and state
# ---------------------------------------------------------------------------


def read_env_var(name: str) -> str:
    """Return the value of ``name`` from .env ('' if absent), without quotes."""
    if not ENV_FILE.exists():
        return ""
    value = ""
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            value = line.split("=", 1)[1].strip().strip("\r")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
    return value


def write_env_var(name: str, value: str) -> None:
    """Replace every ``name=`` line in .env with one line, or append it."""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    kept = [line for line in lines if not line.startswith(f"{name}=")]
    kept.append(f"{name}={value}")
    ENV_FILE.write_text("\n".join(kept) + "\n", encoding="utf-8")


def load_state() -> dict[str, str]:
    if STATE_FILE.exists():
        state: dict[str, str] = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return state
    return {}


def save_state(state: dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def endpoint(pod_id: str) -> str:
    return f"https://{pod_id}-{PORT}.proxy.runpod.net"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Secrets:
    """Holds the keys and masks them in anything printed."""

    def __init__(self, runpod: str, vllm: str) -> None:
        self.runpod = runpod
        self.vllm = vllm

    def redact(self, text: str) -> str:
        for key in (self.runpod, self.vllm):
            if key:
                text = text.replace(key, "***")
        return text


def request(
    method: str, url: str, token: str, body: dict[str, Any] | None = None, timeout: float = 30
) -> tuple[int, Any]:
    """Send a JSON request; return (HTTP status, parsed JSON or raw text)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", "ukrqualbench-runpod/1.0")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read().decode()
    except urllib.error.HTTPError as err:
        status, raw = err.code, err.read().decode(errors="replace")
    except (urllib.error.URLError, TimeoutError) as err:
        return 0, str(err)
    try:
        return status, json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return status, raw


def require_keys(need_vllm: bool) -> Secrets:
    keys = Secrets(read_env_var(RUNPOD_KEY_VAR), read_env_var(VLLM_KEY_VAR))
    missing = [RUNPOD_KEY_VAR] if not keys.runpod else []
    if need_vllm and not keys.vllm:
        missing.append(f"{VLLM_KEY_VAR} (run `keygen` first)")
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")
    return keys


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_keygen() -> None:
    current = read_env_var(VLLM_KEY_VAR)
    if current:
        print(f"{VLLM_KEY_VAR} already set ({len(current)} chars), not changed")
        return
    key = secrets.token_urlsafe(32)
    write_env_var(VLLM_KEY_VAR, key)
    print(f"{VLLM_KEY_VAR} written to .env ({len(key)} chars)")


def pod_body(model: ServedModel, vllm_key: str) -> dict[str, Any]:
    return {
        "name": f"ukrqb-{model.served}"[:191],
        "imageName": IMAGE,
        "gpuTypeIds": model.gpu_ids,
        "gpuCount": 1,
        "cloudType": "SECURE",
        "containerDiskInGb": model.container_disk_gb,
        "ports": [f"{PORT}/http"],
        "dockerEntrypoint": ["vllm", "serve"],
        "dockerStartCmd": [
            model.repo,
            "--revision",
            model.revision,
            "--served-model-name",
            model.served,
            "--api-key",
            vllm_key,
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--gpu-memory-utilization",
            "0.90",
        ],
    }


def sync_endpoints(state: dict[str, str]) -> None:
    endpoints = {served: endpoint(pod_id) for served, pod_id in state.items()}
    write_env_var(ENDPOINTS_VAR, f"'{json.dumps(endpoints)}'")
    print(f"{ENDPOINTS_VAR} written to .env: {json.dumps(endpoints)}")


def cmd_create(names: list[str]) -> None:
    """Create pods for ``names`` (all models when empty) that are not in the state file."""
    keys = require_keys(need_vllm=True)
    state = load_state()
    for model in MODELS:
        if names and model.served not in names:
            continue
        if model.served in state:
            print(f"{model.served}: already in state as pod {state[model.served]}, skipped")
            continue
        status, resp = request("POST", f"{API}/pods", keys.runpod, pod_body(model, keys.vllm))
        if status not in (200, 201) or not isinstance(resp, dict) or "id" not in resp:
            print(f"{model.served}: create failed, HTTP {status}: {keys.redact(str(resp))[:500]}")
            print(
                "Pods created so far keep billing: run `status`, then `create` again or `delete`."
            )
            break
        state[model.served] = resp["id"]
        save_state(state)
        print(
            f"{model.served}: pod {resp['id']} {resp.get('desiredStatus')} "
            f"${resp.get('costPerHr')}/hr"
        )
    sync_endpoints(state)


def cmd_status() -> None:
    keys = require_keys(need_vllm=True)
    state = load_state()
    if not state:
        print("No pods in state file.")
        return
    for served, pod_id in state.items():
        status, pod = request("GET", f"{API}/pods/{pod_id}", keys.runpod)
        if status != 200 or not isinstance(pod, dict):
            print(f"{served}: pod {pod_id} HTTP {status}: {keys.redact(str(pod))[:200]}")
            continue
        probe, models = request("GET", f"{endpoint(pod_id)}/v1/models", keys.vllm, timeout=10)
        ready = [m.get("id") for m in models.get("data", [])] if isinstance(models, dict) else []
        print(
            f"{served}: pod {pod_id} {pod.get('desiredStatus')} ${pod.get('costPerHr')}/hr | "
            f"/v1/models HTTP {probe} {ready or ''}"
        )


def cmd_switch(old: str, new: str) -> None:
    """Restart the pod serving ``old`` with ``new`` (PATCH /pods/{id}: the pod is reset)."""
    keys = require_keys(need_vllm=True)
    state = load_state()
    if old not in state:
        sys.exit(f"{old} is not in the state file")
    if new in state:
        sys.exit(f"{new} already has pod {state[new]}")
    pod_id = state[old]
    model = next(m for m in MODELS if m.served == new)
    _, before = request("GET", f"{API}/pods/{pod_id}", keys.runpod)
    body = pod_body(model, keys.vllm)
    patch = {k: body[k] for k in ("dockerEntrypoint", "dockerStartCmd", "containerDiskInGb")}
    status, resp = request("PATCH", f"{API}/pods/{pod_id}", keys.runpod, patch)
    if status != 200 or not isinstance(resp, dict):
        sys.exit(f"switch failed, HTTP {status}: {keys.redact(str(resp))[:500]}")
    del state[old]
    state[new] = pod_id
    save_state(state)
    machine_before = before.get("machineId") if isinstance(before, dict) else None
    print(
        f"pod {pod_id}: {old} -> {new}, {resp.get('desiredStatus')}, "
        f"machineId {machine_before} -> {resp.get('machineId')}"
    )
    sync_endpoints(state)


def cmd_delete(names: list[str]) -> None:
    """Terminate pods of ``names`` (every pod in the state file when empty)."""
    keys = require_keys(need_vllm=False)
    state = load_state()
    targets = {s: p for s, p in state.items() if not names or s in names}
    if not targets:
        print("No matching pods in state file.")
        return
    remaining = dict(state)
    for served, pod_id in targets.items():
        status, resp = request("DELETE", f"{API}/pods/{pod_id}", keys.runpod)
        check, _ = request("GET", f"{API}/pods/{pod_id}", keys.runpod)
        gone = check == 404
        if gone:
            del remaining[served]
        print(
            f"{served}: DELETE HTTP {status}, GET after delete HTTP {check}"
            f" -> {'terminated' if gone else 'STILL PRESENT: ' + keys.redact(str(resp))[:200]}"
        )
    still_present = [s for s in targets if s in remaining]
    if still_present:
        print(f"Still present: {', '.join(still_present)}; run `delete` again or use the console.")
    if remaining:
        save_state(remaining)
    else:
        STATE_FILE.unlink()
        print("No pods left; state file removed.")
    sync_endpoints(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the RunPod vLLM pods for UkrQualBench.")
    parser.add_argument("command", choices=["keygen", "create", "status", "switch", "delete"])
    parser.add_argument(
        "models",
        nargs="*",
        help=f"create/delete: only these; switch: OLD NEW. Known: {', '.join(MODEL_NAMES)}",
    )
    args = parser.parse_args()
    unknown = [m for m in args.models if m not in MODEL_NAMES]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")
    if args.models and args.command not in ("create", "delete", "switch"):
        parser.error("model names apply only to create, switch and delete")
    if args.command == "switch" and len(args.models) != 2:
        parser.error("switch needs exactly two names: OLD NEW")
    if args.command == "keygen":
        cmd_keygen()
    elif args.command == "create":
        cmd_create(args.models)
    elif args.command == "status":
        cmd_status()
    elif args.command == "switch":
        cmd_switch(*args.models)
    else:
        cmd_delete(args.models)


if __name__ == "__main__":
    main()
