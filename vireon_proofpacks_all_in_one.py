#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    import gymnasium as gym  # type: ignore
except Exception:
    gym = None  # type: ignore

try:
    from nacl.signing import SigningKey, VerifyKey  # type: ignore
except Exception:
    SigningKey = None  # type: ignore
    VerifyKey = None  # type: ignore


def _utc_now_compact() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)
        if not line.endswith("\n"):
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def ensure(condition: bool, msg: str) -> None:
    if not condition:
        raise RuntimeError(msg)


@dataclass(frozen=True)
class ArtifactMeta:
    n_samples: int
    n_clipped: int
    min_val: float
    max_val: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_clipped": self.n_clipped,
            "min_val": float(self.min_val),
            "max_val": float(self.max_val),
        }


def create_clipped_artifact(path: Path, samples: Iterable[float], min_val: float, max_val: float) -> ArtifactMeta:
    xs = [float(x) for x in samples]
    clipped = 0
    ys: List[float] = []
    for x in xs:
        y = x
        if y < min_val:
            y = min_val
            clipped += 1
        elif y > max_val:
            y = max_val
            clipped += 1
        ys.append(float(y))
    payload = {
        "schema": "vireon_artifact_v1",
        "meta": {
            "created_utc": _utc_now_compact(),
            "bounds": {"min_val": float(min_val), "max_val": float(max_val)},
            "n_samples": len(ys),
            "n_clipped": clipped,
        },
        "samples": ys,
    }
    atomic_write_bytes(path, canonical_json_bytes(payload) + b"\n")
    return ArtifactMeta(n_samples=len(ys), n_clipped=clipped, min_val=min_val, max_val=max_val)


def load_artifact(path: Path) -> Tuple[ArtifactMeta, List[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ensure(data.get("schema") == "vireon_artifact_v1", f"Bad artifact schema: {path}")
    meta = data["meta"]
    bounds = meta["bounds"]
    min_val = float(bounds["min_val"])
    max_val = float(bounds["max_val"])
    samples = [float(x) for x in data["samples"]]
    n_samples = int(meta["n_samples"])
    n_clipped = int(meta["n_clipped"])
    ensure(n_samples == len(samples), f"Artifact n_samples mismatch: {path}")
    return ArtifactMeta(n_samples=n_samples, n_clipped=n_clipped, min_val=min_val, max_val=max_val), samples


def alpha_spending_step(alpha_total: float, step_index_1based: int) -> float:
    k = float(step_index_1based)
    return float(alpha_total) * (6.0 / (math.pi ** 2)) * (1.0 / (k * k))


def hoeffding_mean_radius(n: int, min_val: float, max_val: float, alpha: float) -> float:
    ensure(n > 0, "n must be > 0")
    ensure(alpha > 0.0 and alpha < 1.0, "alpha must be in (0,1)")
    width = float(max_val - min_val)
    ensure(width > 0.0, "bounds width must be > 0")
    return width * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


@dataclass(frozen=True)
class PacConfig:
    alpha_total: float = 0.01
    alpha_split_L: float = 1 / 3
    alpha_split_eps: float = 1 / 3
    alpha_split_kl: float = 1 / 3
    lam_eps: float = 0.5
    lam_kl: float = 1.0
    kl_target: Optional[float] = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class PacResult:
    step_index_1based: int
    alpha_step: float
    deltaL_hat: float
    deltaL_lcb: float
    eps_hat: float
    eps_ucb: float
    kl_hat: float
    kl_ucb: float
    penalty_ucb: float
    p_pac: float
    accepted: bool

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def compute_pac_from_artifacts(
    artifact_deltaL_path: Path,
    artifact_eps_path: Path,
    artifact_kl_path: Path,
    step_index_1based: int,
    cfg: PacConfig,
) -> PacResult:
    mL, sL = load_artifact(artifact_deltaL_path)
    meps, seps = load_artifact(artifact_eps_path)
    mkl, skl = load_artifact(artifact_kl_path)

    alpha_step = alpha_spending_step(cfg.alpha_total, step_index_1based)
    alpha_L = alpha_step * cfg.alpha_split_L
    alpha_eps = alpha_step * cfg.alpha_split_eps
    alpha_kl = alpha_step * cfg.alpha_split_kl

    deltaL_hat = float(sum(sL) / max(1, len(sL)))
    eps_hat = float(sum(seps) / max(1, len(seps)))
    kl_hat = float(sum(skl) / max(1, len(skl)))

    rL = hoeffding_mean_radius(mL.n_samples, mL.min_val, mL.max_val, alpha_L)
    reps = hoeffding_mean_radius(meps.n_samples, meps.min_val, meps.max_val, alpha_eps)
    rkl = hoeffding_mean_radius(mkl.n_samples, mkl.min_val, mkl.max_val, alpha_kl)

    deltaL_lcb = deltaL_hat - rL
    eps_ucb = eps_hat + reps
    kl_ucb = kl_hat + rkl

    penalty_ucb = cfg.lam_eps * eps_ucb + cfg.lam_kl * kl_ucb
    p_pac = deltaL_lcb - penalty_ucb

    accepted = (p_pac >= 0.0)
    if cfg.kl_target is not None:
        accepted = accepted and (kl_ucb <= float(cfg.kl_target))

    return PacResult(
        step_index_1based=int(step_index_1based),
        alpha_step=float(alpha_step),
        deltaL_hat=float(deltaL_hat),
        deltaL_lcb=float(deltaL_lcb),
        eps_hat=float(eps_hat),
        eps_ucb=float(eps_ucb),
        kl_hat=float(kl_hat),
        kl_ucb=float(kl_ucb),
        penalty_ucb=float(penalty_ucb),
        p_pac=float(p_pac),
        accepted=bool(accepted),
    )


def list_all_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            out.append(p)
    out.sort(key=lambda x: str(x).replace("\\", "/"))
    return out


def write_sha256sum(root: Path) -> None:
    lines: List[str] = []
    for p in list_all_files(root):
        rel = p.relative_to(root).as_posix()
        if rel in ("sha256sum.txt",):
            continue
        lines.append(f"{sha256_file(p)}  {rel}")
    atomic_write_text(root / "sha256sum.txt", "\n".join(lines) + "\n")


def build_manifest(root: Path, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    files = []
    for p in list_all_files(root):
        rel = p.relative_to(root).as_posix()
        if rel == "manifest.json":
            continue
        files.append({"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    manifest = {
        "schema": "vireon_manifest_v1",
        "created_utc": _utc_now_compact(),
        "root": root.name,
        "files": files,
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def sign_manifest_ed25519(manifest_bytes: bytes, signing_key: Any) -> Dict[str, Any]:
    sig = signing_key.sign(manifest_bytes).signature
    vk = signing_key.verify_key
    return {
        "scheme": "ed25519",
        "public_key_b64": base64.b64encode(bytes(vk)).decode("ascii"),
        "signature_b64": base64.b64encode(bytes(sig)).decode("ascii"),
        "signed_over": "canonical_json(manifest.json without signature block)",
    }


def compute_row_hash(row_without_hash: Dict[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(row_without_hash))


def row_without_fields(row: Dict[str, Any], remove: Iterable[str]) -> Dict[str, Any]:
    r = dict(row)
    for k in remove:
        r.pop(k, None)
    return r


def read_updates(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


class MLPPolicyValue(nn.Module):  # type: ignore[misc]
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def action_dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:  # type: ignore[override]
        logits = self.pi(obs)
        return torch.distributions.Categorical(logits=logits)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@dataclass
class RolloutBatch:
    obs: Any
    act: Any
    logp_old: Any
    ret: Any
    adv: Any


def collect_rollout(env: Any, model: Any, steps: int, gamma: float, lam: float, device: str) -> RolloutBatch:
    obs_list = []
    act_list = []
    logp_list = []
    rew_list = []
    val_list = []
    done_list = []

    obs, _info = env.reset()
    for _ in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        dist = model.action_dist(obs_t)
        act = dist.sample()
        logp = dist.log_prob(act)
        val = model.value(obs_t)

        next_obs, rew, terminated, truncated, _info = env.step(int(act.item()))
        done = bool(terminated or truncated)

        obs_list.append(obs_t)
        act_list.append(act)
        logp_list.append(logp)
        rew_list.append(float(rew))
        val_list.append(val)
        done_list.append(done)

        obs = next_obs
        if done:
            obs, _info = env.reset()

    with torch.no_grad():
        last_obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        last_val = float(model.value(last_obs_t).item())

    adv = [0.0] * steps
    ret = [0.0] * steps
    gae = 0.0
    for t in reversed(range(steps)):
        v_t = float(val_list[t].item())
        v_next = last_val if t == steps - 1 else float(val_list[t + 1].item())
        nonterminal = 0.0 if done_list[t] else 1.0
        delta = rew_list[t] + gamma * v_next * nonterminal - v_t
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
        ret[t] = adv[t] + v_t

    obs_b = torch.stack(obs_list)
    act_b = torch.stack(act_list)
    logp_b = torch.stack(logp_list).detach()
    adv_b = torch.tensor(adv, dtype=torch.float32, device=device)
    ret_b = torch.tensor(ret, dtype=torch.float32, device=device)

    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

    return RolloutBatch(obs=obs_b, act=act_b, logp_old=logp_b, ret=ret_b, adv=adv_b)


def ppo_update(
    model: Any,
    batch: RolloutBatch,
    lr: float,
    clip_ratio: float,
    vf_coef: float,
    ent_coef: float,
    train_iters: int,
    device: str,
) -> Dict[str, float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs, act, logp_old, ret, adv = batch.obs, batch.act, batch.logp_old, batch.ret, batch.adv

    with torch.no_grad():
        dist_old = model.action_dist(obs)

    for _ in range(train_iters):
        dist = model.action_dist(obs)
        logp = dist.log_prob(act)
        ratio = torch.exp(logp - logp_old)
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        pi_loss = -(torch.min(ratio * adv, clipped * adv)).mean()

        v = model.value(obs)
        vf_loss = 0.5 * ((v - ret) ** 2).mean()

        ent = dist.entropy().mean()
        loss = pi_loss + vf_coef * vf_loss - ent_coef * ent

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    with torch.no_grad():
        dist_new = model.action_dist(obs)
        logp_new = dist_new.log_prob(act)
        ratio = torch.exp(logp_new - logp_old)

        deltaL_hat = float(((ratio - 1.0) * adv).mean().item())

        p_old = dist_old.probs
        logp_old_all = torch.log(p_old + 1e-12)
        logp_new_all = torch.log(dist_new.probs + 1e-12)
        kl = (p_old * (logp_old_all - logp_new_all)).sum(dim=-1).mean()
        kl_hat = float(kl.item())

        eps_hat = float(torch.abs(adv).mean().item())

    return {"deltaL_hat": deltaL_hat, "kl_hat": kl_hat, "eps_hat": eps_hat}


def save_checkpoint(path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)  # type: ignore[attr-defined]


def load_checkpoint(path: Path, model: Any) -> None:
    state = torch.load(path, map_location="cpu")  # type: ignore[attr-defined]
    model.load_state_dict(state)


@dataclass
class RunConfig:
    run_id: str
    created_utc: str
    algo: str
    env_id: str
    seed: int
    pac: PacConfig
    notes: str = "demo"

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["pac"] = self.pac.to_dict()
        return d


class ProofPackEngine:
    def __init__(self, root: Path, run_cfg: RunConfig, sign: bool = False, store_private_key: bool = False):
        self.root = root
        self.run_cfg = run_cfg
        self.sign = sign
        self.store_private_key = store_private_key

        self.updates_path = self.root / "updates.jsonl"
        self.run_path = self.root / "run.json"

        self.root.mkdir(parents=True, exist_ok=True)

        atomic_write_bytes(
            self.run_path,
            canonical_json_bytes(
                {
                    "schema": "vireon_run_v1",
                    "created_utc": run_cfg.created_utc,
                    "run_id": run_cfg.run_id,
                    "algo": run_cfg.algo,
                    "env_id": run_cfg.env_id,
                    "seed": run_cfg.seed,
                    "pac": run_cfg.pac.to_dict(),
                    "notes": run_cfg.notes,
                }
            )
            + b"\n",
        )

        self.genesis_hash = sha256_file(self.run_path)

        self._signing_key = None
        self._verify_key_b64 = None
        if self.sign:
            ensure(SigningKey is not None, "Signing requested but PyNaCl not installed. pip install '.[sign]'")
            self._signing_key = SigningKey.generate()
            self._verify_key_b64 = base64.b64encode(bytes(self._signing_key.verify_key)).decode("ascii")
            signing_dir = self.root / "signing"
            signing_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(signing_dir / "PUBLIC_KEY_B64.txt", self._verify_key_b64 + "\n")
            if self.store_private_key:
                atomic_write_text(
                    signing_dir / "PRIVATE_KEY_B64_DO_NOT_SHARE.txt",
                    base64.b64encode(bytes(self._signing_key)).decode("ascii") + "\n",
                )

    def _last_row_hash(self) -> str:
        rows = read_updates(self.updates_path)
        if not rows:
            return self.genesis_hash
        return str(rows[-1]["row_hash"])

    def log_update(
        self,
        step_index_1based: int,
        theta_old_path: Path,
        theta_try_path: Path,
        artifact_deltaL_path: Path,
        artifact_eps_path: Path,
        artifact_kl_path: Path,
        extra_metrics: Dict[str, Any],
    ) -> PacResult:
        theta_old_sha = sha256_file(theta_old_path)
        theta_try_sha = sha256_file(theta_try_path)
        aL_sha = sha256_file(artifact_deltaL_path)
        aeps_sha = sha256_file(artifact_eps_path)
        akl_sha = sha256_file(artifact_kl_path)

        pac = compute_pac_from_artifacts(
            artifact_deltaL_path=artifact_deltaL_path,
            artifact_eps_path=artifact_eps_path,
            artifact_kl_path=artifact_kl_path,
            step_index_1based=step_index_1based,
            cfg=self.run_cfg.pac,
        )

        prev = self._last_row_hash()
        row = {
            "schema": "vireon_update_row_v2",
            "created_utc": _utc_now_compact(),
            "run_id": self.run_cfg.run_id,
            "step_index_1based": int(step_index_1based),
            "prev_row_hash": prev,
            "theta_old_path": theta_old_path.relative_to(self.root).as_posix(),
            "theta_old_sha256": theta_old_sha,
            "theta_try_path": theta_try_path.relative_to(self.root).as_posix(),
            "theta_try_sha256": theta_try_sha,
            "artifact_deltaL_path": artifact_deltaL_path.relative_to(self.root).as_posix(),
            "artifact_deltaL_sha256": aL_sha,
            "artifact_eps_path": artifact_eps_path.relative_to(self.root).as_posix(),
            "artifact_eps_sha256": aeps_sha,
            "artifact_kl_path": artifact_kl_path.relative_to(self.root).as_posix(),
            "artifact_kl_sha256": akl_sha,
            "pac": pac.to_dict(),
            "accepted": bool(pac.accepted),
            "extra": extra_metrics,
        }

        row_hash = compute_row_hash(row)
        row["row_hash"] = row_hash

        atomic_append_line(self.updates_path, json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return pac

    def finalize(self) -> None:
        write_sha256sum(self.root)
        manifest = build_manifest(self.root, extra={"public_key_b64": self._verify_key_b64} if self._verify_key_b64 else None)

        if self._signing_key is not None:
            manifest_bytes = canonical_json_bytes(manifest)
            manifest["signature"] = sign_manifest_ed25519(manifest_bytes, self._signing_key)

        atomic_write_bytes(self.root / "manifest.json", canonical_json_bytes(manifest) + b"\n")


@dataclass
class VerifyReport:
    ok: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "summary": self.summary}, indent=2, sort_keys=True)


def verify_pack(root: Path, tol: float = 1e-9) -> VerifyReport:
    errors: List[str] = []
    warnings: List[str] = []

    run_path = root / "run.json"
    updates_path = root / "updates.jsonl"
    manifest_path = root / "manifest.json"
    sha_path = root / "sha256sum.txt"

    if not run_path.exists():
        errors.append("Missing run.json")
        return VerifyReport(False, errors, warnings, summary={})
    if not updates_path.exists():
        errors.append("Missing updates.jsonl")
        return VerifyReport(False, errors, warnings, summary={})

    run = json.loads(run_path.read_text(encoding="utf-8"))
    ensure(run.get("schema") == "vireon_run_v1", "Bad run schema")
    genesis = sha256_file(run_path)

    pac_cfg_dict = run["pac"]
    pac_cfg = PacConfig(
        alpha_total=float(pac_cfg_dict["alpha_total"]),
        alpha_split_L=float(pac_cfg_dict["alpha_split_L"]),
        alpha_split_eps=float(pac_cfg_dict["alpha_split_eps"]),
        alpha_split_kl=float(pac_cfg_dict["alpha_split_kl"]),
        lam_eps=float(pac_cfg_dict["lam_eps"]),
        lam_kl=float(pac_cfg_dict["lam_kl"]),
        kl_target=None if pac_cfg_dict.get("kl_target", None) is None else float(pac_cfg_dict["kl_target"]),
    )

    rows = read_updates(updates_path)
    if not rows:
        errors.append("updates.jsonl is empty")
        return VerifyReport(False, errors, warnings, summary={})

    prev_expected = genesis
    accepted_count = 0
    last_accepted_theta_sha = None

    for i, row in enumerate(rows):
        try:
            ensure(row.get("schema") == "vireon_update_row_v2", f"Row {i}: bad schema")
            ensure(str(row["prev_row_hash"]) == prev_expected, f"Row {i}: prev_row_hash mismatch")
            row_hash = str(row["row_hash"])
            row_no_hash = row_without_fields(row, ["row_hash"])
            computed = compute_row_hash(row_no_hash)
            ensure(computed == row_hash, f"Row {i}: row_hash mismatch")

            def _chk(rel: str, expected_sha: str, label: str) -> Path:
                p = root / rel
                ensure(p.exists(), f"Row {i}: missing {label} file {rel}")
                got = sha256_file(p)
                ensure(got == expected_sha, f"Row {i}: {label} sha256 mismatch ({rel})")
                return p

            _chk(row["theta_old_path"], row["theta_old_sha256"], "theta_old")
            _chk(row["theta_try_path"], row["theta_try_sha256"], "theta_try")
            aL = _chk(row["artifact_deltaL_path"], row["artifact_deltaL_sha256"], "artifact_deltaL")
            aeps = _chk(row["artifact_eps_path"], row["artifact_eps_sha256"], "artifact_eps")
            akl = _chk(row["artifact_kl_path"], row["artifact_kl_sha256"], "artifact_kl")

            if last_accepted_theta_sha is not None:
                ensure(row["theta_old_sha256"] == last_accepted_theta_sha, f"Row {i}: checkpoint lineage mismatch")

            step_k = int(row["step_index_1based"])
            pac_re = compute_pac_from_artifacts(aL, aeps, akl, step_k, pac_cfg)
            pac_logged = row["pac"]

            def _close(a: float, b: float) -> bool:
                return abs(float(a) - float(b)) <= tol

            for key in ["alpha_step", "deltaL_hat", "deltaL_lcb", "eps_hat", "eps_ucb", "kl_hat", "kl_ucb", "penalty_ucb", "p_pac"]:
                if not _close(pac_re.to_dict()[key], pac_logged[key]):
                    raise RuntimeError(f"Row {i}: pac field mismatch {key}")

            if bool(pac_re.accepted) != bool(row["accepted"]):
                raise RuntimeError(f"Row {i}: accepted flag mismatch")

            prev_expected = row_hash

            if bool(row["accepted"]):
                accepted_count += 1
                last_accepted_theta_sha = row["theta_try_sha256"]
            else:
                last_accepted_theta_sha = row["theta_old_sha256"]

        except Exception as e:
            errors.append(str(e))
            break

    if sha_path.exists():
        wanted = sha_path.read_text(encoding="utf-8").splitlines()
        for line in wanted:
            if not line.strip():
                continue
            exp_sha, rel = line.split("  ", 1)
            p = root / rel
            if not p.exists():
                errors.append(f"sha256sum: missing file {rel}")
                continue
            got = sha256_file(p)
            if got != exp_sha:
                errors.append(f"sha256sum: mismatch for {rel}")
    else:
        warnings.append("Missing sha256sum.txt")

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "signature" in manifest:
            if VerifyKey is None:
                warnings.append("manifest has signature but PyNaCl not installed; cannot verify signature")
            else:
                try:
                    sig = manifest["signature"]
                    manifest_no_sig = dict(manifest)
                    manifest_no_sig.pop("signature", None)
                    mb = canonical_json_bytes(manifest_no_sig)
                    vk = VerifyKey(base64.b64decode(sig["public_key_b64"]))
                    vk.verify(mb, base64.b64decode(sig["signature_b64"]))
                except Exception as e:
                    errors.append(f"Signature verification failed: {e}")
    else:
        warnings.append("Missing manifest.json")

    ok = len(errors) == 0
    return VerifyReport(
        ok=ok,
        errors=errors,
        warnings=warnings,
        summary={"run_id": run.get("run_id"), "rows": len(rows), "accepted_rows": accepted_count, "genesis_hash": genesis},
    )


def attack_flip_deltaL(root: Path) -> None:
    updates = read_updates(root / "updates.jsonl")
    ensure(len(updates) > 0, "No updates to attack.")
    last = updates[-1]
    rel = last["artifact_deltaL_path"]
    p = root / rel
    ensure(p.exists(), f"Missing artifact {rel}")
    data = json.loads(p.read_text(encoding="utf-8"))
    samples = data["samples"]
    ensure(len(samples) > 0, "No samples to flip")
    samples[0] = float(samples[0]) + 0.123456
    data["samples"] = samples
    atomic_write_bytes(p, canonical_json_bytes(data) + b"\n")
    print(f"[attack] modified {rel}. verifier should now FAIL.")


def cmd_train(args: argparse.Namespace) -> int:
    ensure(torch is not None and gym is not None and np is not None, "Requires numpy, torch, gymnasium. pip install '.[demo]'")
    set_global_seeds(int(args.seed))

    device = "cpu"
    env = gym.make("CartPole-v1")  # type: ignore[attr-defined]
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.n)

    model = MLPPolicyValue(obs_dim, act_dim, hidden=int(args.hidden)).to(device)  # type: ignore[arg-type]
    model_old = MLPPolicyValue(obs_dim, act_dim, hidden=int(args.hidden)).to(device)  # type: ignore[arg-type]
    model_old.load_state_dict(model.state_dict())

    run_id = f"CERTPPO_CARTPOLE_RUN_{_dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    root = Path(args.out_dir) / run_id

    pac_cfg = PacConfig(
        alpha_total=float(args.alpha),
        lam_eps=float(args.lam_eps),
        lam_kl=float(args.lam_kl),
        kl_target=None if args.kl_target is None else float(args.kl_target),
    )

    engine = ProofPackEngine(
        root=root,
        run_cfg=RunConfig(run_id=run_id, created_utc=_utc_now_compact(), algo="PPO_demo_certified", env_id="CartPole-v1", seed=int(args.seed), pac=pac_cfg),
        sign=bool(args.sign),
        store_private_key=bool(args.store_private_key),
    )

    ckpt0 = root / "checkpoints" / "theta_old_step000.pt"
    save_checkpoint(ckpt0, model_old)

    total_steps = int(args.steps)
    rollout_steps = int(args.rollout_steps)
    updates = max(1, total_steps // rollout_steps)

    deltaL_bounds = (float(args.deltaL_min), float(args.deltaL_max))
    eps_bounds = (float(args.eps_min), float(args.eps_max))
    kl_bounds = (float(args.kl_min), float(args.kl_max))

    print(f"[train] pack: {root.as_posix()}")
    print(f"[train] updates={updates} rollout_steps={rollout_steps} alpha_total={pac_cfg.alpha_total}")

    theta_old_path = ckpt0

    for k in range(1, updates + 1):
        load_checkpoint(theta_old_path, model_old)
        model.load_state_dict(model_old.state_dict())

        batch = collect_rollout(env, model_old, rollout_steps, gamma=float(args.gamma), lam=float(args.lam), device=device)

        _ = ppo_update(
            model=model,
            batch=batch,
            lr=float(args.lr),
            clip_ratio=float(args.clip_ratio),
            vf_coef=float(args.vf_coef),
            ent_coef=float(args.ent_coef),
            train_iters=int(args.train_iters),
            device=device,
        )

        theta_try_path = root / "checkpoints" / f"theta_try_step{k:03d}.pt"
        save_checkpoint(theta_try_path, model)

        with torch.no_grad():
            obs = batch.obs
            act = batch.act
            adv = batch.adv
            logp_old = batch.logp_old
            dist_new = model.action_dist(obs)
            logp_new = dist_new.log_prob(act)
            ratio = torch.exp(logp_new - logp_old)
            deltaL_samples = ((ratio - 1.0) * adv).detach().cpu().numpy().tolist()

            eps_samples = torch.abs(adv).detach().cpu().numpy().tolist()

            dist_old = model_old.action_dist(obs)
            p_old = dist_old.probs
            p_new = dist_new.probs
            kl_per = (p_old * (torch.log(p_old + 1e-12) - torch.log(p_new + 1e-12))).sum(dim=-1)
            kl_samples = kl_per.detach().cpu().numpy().tolist()

        aL_path = root / "artifacts" / f"deltaL_step{k:03d}.json"
        aeps_path = root / "artifacts" / f"eps_step{k:03d}.json"
        akl_path = root / "artifacts" / f"kl_step{k:03d}.json"

        metaL = create_clipped_artifact(aL_path, deltaL_samples, deltaL_bounds[0], deltaL_bounds[1])
        metaeps = create_clipped_artifact(aeps_path, eps_samples, eps_bounds[0], eps_bounds[1])
        metakl = create_clipped_artifact(akl_path, kl_samples, kl_bounds[0], kl_bounds[1])

        pac = engine.log_update(
            step_index_1based=k,
            theta_old_path=theta_old_path,
            theta_try_path=theta_try_path,
            artifact_deltaL_path=aL_path,
            artifact_eps_path=aeps_path,
            artifact_kl_path=akl_path,
            extra_metrics={"artifact_meta": {"deltaL": metaL.to_dict(), "eps": metaeps.to_dict(), "kl": metakl.to_dict()}},
        )

        if pac.accepted:
            theta_next_old = root / "checkpoints" / f"theta_old_step{k:03d}.pt"
            shutil.copyfile(theta_try_path, theta_next_old)
            theta_old_path = theta_next_old
            status = "ACCEPT"
        else:
            theta_next_old = root / "checkpoints" / f"theta_old_step{k:03d}.pt"
            shutil.copyfile(theta_old_path, theta_next_old)
            theta_old_path = theta_next_old
            status = "REJECT"

        print(f"[train] step={k:03d} {status} p_pac={pac.p_pac:+.6f}")

    engine.finalize()
    rep = verify_pack(root)
    atomic_write_text(root / "VERIFY_REPORT.json", rep.to_json() + "\n")
    print(f"[train] verified_ok={rep.ok}")
    if not rep.ok:
        print(rep.to_json())
        return 2
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    rep = verify_pack(Path(args.pack_dir), tol=float(args.tol))
    print(rep.to_json())
    return 0 if rep.ok else 2


def cmd_attack_flip(args: argparse.Namespace) -> int:
    attack_flip_deltaL(Path(args.pack_dir))
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vireon-proofpacks")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--out-dir", default="results")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--rollout-steps", type=int, default=1000)
    t.add_argument("--hidden", type=int, default=64)

    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--clip-ratio", type=float, default=0.2)
    t.add_argument("--train-iters", dest="train_iters", type=int, default=10)
    t.add_argument("--vf-coef", type=float, default=0.5)
    t.add_argument("--ent-coef", type=float, default=0.0)
    t.add_argument("--gamma", type=float, default=0.99)
    t.add_argument("--lam", type=float, default=0.95)

    t.add_argument("--alpha", type=float, default=0.01)
    t.add_argument("--lam-eps", dest="lam_eps", type=float, default=0.5)
    t.add_argument("--lam-kl", dest="lam_kl", type=float, default=1.0)
    t.add_argument("--kl-target", type=float, default=0.02)

    t.add_argument("--deltaL-min", type=float, default=-1.0)
    t.add_argument("--deltaL-max", type=float, default=1.0)
    t.add_argument("--eps-min", type=float, default=0.0)
    t.add_argument("--eps-max", type=float, default=3.0)
    t.add_argument("--kl-min", type=float, default=0.0)
    t.add_argument("--kl-max", type=float, default=0.5)

    t.add_argument("--sign", action="store_true")
    t.add_argument("--store-private-key", action="store_true")

    v = sub.add_parser("verify")
    v.add_argument("pack_dir")
    v.add_argument("--tol", type=float, default=1e-9)

    a = sub.add_parser("attack-flip")
    a.add_argument("pack_dir")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.cmd == "train":
        if args.kl_target is not None and float(args.kl_target) < 0:
            args.kl_target = None
        return cmd_train(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    if args.cmd == "attack-flip":
        return cmd_attack_flip(args)
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
