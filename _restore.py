"""Restaura el archivo de ejercicio desde el último commit git."""
import subprocess, pathlib, sys

fname = "pages/7_\U0001f9ea_Ejercicio.py"
result = subprocess.run(
    ["git", "show", f"HEAD:{fname}"],
    capture_output=True,
    cwd=str(pathlib.Path(__file__).parent),
)
if result.returncode != 0:
    sys.stdout.write(f"ERROR: {result.stderr.decode('utf-8', errors='replace')}\n")
    sys.exit(1)

target = pathlib.Path(__file__).parent / fname
target.write_bytes(result.stdout)
sys.stdout.write(f"OK — {len(result.stdout)} bytes escritos en {target}\n")
