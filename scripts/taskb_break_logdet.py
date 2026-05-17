#!/usr/bin/env python3
"""
Patch JaQMC LogDet to deliberately break fermionic antisymmetry.

It changes the real-valued LogDet branch from signed determinant summation to
unsigned determinant summation.

Original:
    sum sign(det_k) * exp(log|det_k|)

Broken:
    sum exp(log|det_k|)

Run from JaQMC repository root:

    python scripts/taskb_break_logdet.py
"""

from pathlib import Path

p = Path("src/jaqmc/wavefunction/output/logdet.py")
backup = Path("src/jaqmc/wavefunction/output/logdet.py.taskb_backup")

if not p.exists():
    raise SystemExit(f"Cannot find {p}. Run this script from JaQMC repository root.")

s = p.read_text()

if not backup.exists():
    backup.write_text(s)
    print(f"Backup written to {backup}")
else:
    print(f"Backup already exists: {backup}")

old = '''        signed_sum = jnp.sum(signs * jnp.exp(logdets - logmax))
        return RealLogDetOutput(
            sign_logpsi=jnp.sign(signed_sum),
            logpsi=jnp.log(jnp.abs(signed_sum)) + logmax,
            sign_logdets=signs,
            abs_logdets=logdets,
        )
'''

new = '''        # BROKEN FOR TASK B:
        # Ignore determinant signs. This destroys same-spin exchange antisymmetry.
        unsigned_sum = jnp.sum(jnp.exp(logdets - logmax))
        return RealLogDetOutput(
            sign_logpsi=jnp.ones((), dtype=logdets.dtype),
            logpsi=jnp.log(unsigned_sum) + logmax,
            sign_logdets=jnp.ones_like(signs),
            abs_logdets=logdets,
        )
'''

if new in s:
    print("LogDet is already patched.")
elif old in s:
    p.write_text(s.replace(old, new))
    print(f"Patched {p}")
else:
    raise SystemExit(
        "Expected code pattern was not found. Open "
        "src/jaqmc/wavefunction/output/logdet.py manually and patch the real-valued "
        "branch after `signs, logdets = jnp.linalg.slogdet(xs)`."
    )
