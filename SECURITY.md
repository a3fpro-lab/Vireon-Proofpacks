# Security Policy

## Reporting a Vulnerability

If you believe you’ve found a security issue in Vireon ProofPacks (e.g., verification bypass, signature misuse, chain integrity flaw), please report it privately.

**Email:** echoaseternity@gmail.com

Include:
- A clear description of the issue
- Steps to reproduce (minimal PoC if possible)
- What you expected to happen vs. what happened
- Any suggested fixes

We will acknowledge receipt and work to provide a fix as quickly as possible.

## Scope

In scope:
- Verifier correctness (false positives / bypasses)
- Ledger chaining / hash binding flaws
- Signature verification flaws
- Artifact parsing / derivation integrity issues

Out of scope:
- Denial-of-service via extremely large inputs (unless it causes verifier misclassification)
