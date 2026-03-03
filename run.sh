#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "[diag] Python: $(python3 --version 2>/dev/null || python --version 2>/dev/null || echo unknown)"
python3 - <<'PY'
try:
    import torch
    print(f"[diag] torch: {torch.__version__}")
    print(f"[diag] cuda_available: {torch.cuda.is_available()}")
    print(f"[diag] cuda_devices: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[diag] cuda_name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[diag] torch_check_error: {e}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[diag] nvidia-smi summary:"
  nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true
else
  echo "[diag] nvidia-smi not found"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log --log-level warning
