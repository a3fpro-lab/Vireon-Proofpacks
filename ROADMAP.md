# ROADMAP — Breaking Ground

This project is about **verifiable RL progress**: proof-carrying updates you can replay, audit, and trust.

## Milestone 1 — ProofPack v0.1 (DONE)
- Row-chained `updates.jsonl` (anti-reorder / anti-delete)
- SHA256 binding for checkpoints + raw artifacts
- Replayable PAC certificate recomputation by verifier
- Optional Ed25519 manifest signing
- CartPole PPO demo + tamper demo

## Milestone 2 — “Hard Mode” PAC (next)
Goal: reduce conservatism while keeping anytime-valid guarantees.

- [ ] Add **Empirical Bernstein confidence sequences** for mean (tighter than Hoeffding when variance is low)
- [ ] Add **winsorization / robust clipping** modes (logged)
- [ ] Add **adaptive bounds reporting**: show how much clipping happened per step and how it impacts certificate width
- [ ] Add verifier-side “tightness report”: accepted/rejected reasons and margin statistics

Deliverable:
- CI demo run that shows a healthy accept rate (not 0%) while keeping α-control.

## Milestone 3 — Streaming Artifacts (scale)
Goal: handle 100k–10M samples without JSON pain.

- [ ] Add `.npz` artifact format (NumPy) with:
  - header with bounds + counts
  - stable canonical hashing
- [ ] Keep JSON as a “small/debug” format
- [ ] Verifier supports both formats identically

Deliverable:
- CartPole run with large rollouts, artifact size reduced, verification still deterministic.

## Milestone 4 — Beyond CartPole (real credibility)
Goal: show the concept survives harder environments.

- [ ] LunarLander-v2 (discrete)
- [ ] Acrobot-v1 (discrete)
- [ ] Pendulum-v1 (continuous) via discretization or Gaussian policy
- [ ] (Optional) MuJoCo via Gymnasium (requires extra deps)

Deliverable:
- At least 2 environments with proofpacks + verification reports.

## Milestone 5 — “External Auditor” Story
Goal: someone else can reproduce your claim.

- [ ] Provide a minimal “auditor script” that:
  1) downloads a pack
  2) runs verifier
  3) prints a one-page report
- [ ] Add a `papers/` folder with theorems and definitions matching implementation

Deliverable:
- One command reproduces a verified report from a released pack.

## Milestone 6 — Public Benchmark Packs
Goal: create a public corpus of proofpacks.

- [ ] Publish packs as GitHub release assets
- [ ] Index file: `PACKS.md` with hashes + metadata
- [ ] Standardize run naming

Deliverable:
- A public “ProofPack Zoo” people can test verifiers against.
