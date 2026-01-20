

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

Optional signing support:

pip install -e ".[demo,sign]"

Quickstart (CartPole PPO demo)

Train and create a proof pack:

vireon-proofpacks train --steps 500 --seed 1

Verify the most recent pack:

vireon-proofpacks verify results/CERTPPO_CARTPOLE_RUN_*

What “verified: true” means

The verifier checks:
	1.	Chain integrity: each row hash matches content, and prev_row_hash links correctly from genesis.
	2.	Checkpoint integrity: every referenced checkpoint file matches its logged SHA256.
	3.	Artifact integrity: every raw artifact file matches its logged SHA256.
	4.	Derivation integrity: PAC fields are recomputed from raw artifacts and match the ledger (within tolerance).
	5.	Acceptance semantics: accepted matches the recomputed PAC gate.

Tamper demo (should fail verification)

After training, flip a value in the last ΔL artifact:

vireon-proofpacks attack-flip results/CERTPPO_CARTPOLE_RUN_*
vireon-proofpacks verify results/CERTPPO_CARTPOLE_RUN_*

License

MIT — see LICENSE.

