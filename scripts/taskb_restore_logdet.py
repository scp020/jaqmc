#!/usr/bin/env python3
"""
Restore original LogDet implementation after Task B.
"""

from pathlib import Path

p = Path("src/jaqmc/wavefunction/output/logdet.py")
backup = Path("src/jaqmc/wavefunction/output/logdet.py.taskb_backup")

if not backup.exists():
    raise SystemExit(f"Backup not found: {backup}")

p.write_text(backup.read_text())
print(f"Restored {p} from {backup}")
