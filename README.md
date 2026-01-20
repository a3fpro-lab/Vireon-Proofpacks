# Vireon ProofPacks

**Proof-carrying RL training artifacts**: a tamper-evident evidence pack for PPO-style updates that can be **verified independently**.

What you get in each pack:
- **Append-only update ledger** (`updates.jsonl`) with **row chaining** (`prev_row_hash`) so reordering/insertion/deletion breaks verification
- **Checkpoint binding**: each referenced checkpoint file is SHA256-hashed and verified
- **Raw artifact binding**: logged samples (e.g., ΔL, ε, KL) are stored as bounded artifacts and SHA256-hashed
- **Derivation integrity**: verifier recomputes the PAC certificate from raw artifacts and checks exact match
- **Optional Ed25519 signing** of the manifest (authenticity)

This repo ships as a single-file engine:
- `vireon_proofpacks_all_in_one.py`

## Install

```bash
python -m pip install -U pip
pip install -e ".[demo]"
