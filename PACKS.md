# Public ProofPacks Index

This file lists released evidence packs (ProofPacks) and how to verify them.

## How to verify any pack

```bash
pip install -e ".[demo]"
vireon-proofpacks verify <PACK_DIR>

Verification checks chain integrity, hashes, derivation recomputation, and acceptance semantics.

⸻

Packs

(Add entries here as you publish them)

Template:
	•	Pack name: CERTPPO_CARTPOLE_RUN_YYYYMMDDTHHMMSSZ
	•	Run ID: ...
	•	Env: CartPole-v1
	•	Algo: PPO_demo_certified
	•	Commit: <git sha>
	•	Pack sha256 (zip): <sha256>

vireon-proofpacks verify results/CERTPPO_CARTPOLE_RUN_*

