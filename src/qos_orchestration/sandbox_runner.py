# Executed with exec() inside the network-less container; `d` holds {"programs": [...], "marker": str}.
import json
import os
import subprocess
import sys
import tempfile

results = []
for program in d["programs"]:  # noqa: F821
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=10)
        passed = r.returncode == 0 and d["marker"].encode() in r.stdout  # noqa: F821
        results.append({"passed": passed, "err": r.stderr.decode(errors="replace")[-300:]})
    except subprocess.TimeoutExpired:
        results.append({"passed": False, "err": "timeout"})
    os.unlink(f.name)
print(json.dumps(results))
