#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

read_env_value() {
  local key="$1"
  local default_value="$2"
  if [[ -f ".env" ]]; then
    local line
    line="$(grep -E "^${key}=" .env | tail -n 1 || true)"
    if [[ -n "$line" ]]; then
      local value="${line#*=}"
      value="${value%\"}"
      value="${value#\"}"
      echo "$value"
      return
    fi
  fi
  echo "$default_value"
}

# Testing mode: wipe DB/photos on each start.
# Disable with: TEST_RESET_ON_START=0 bash run.sh
test_reset_on_start="${TEST_RESET_ON_START:-1}"
if [[ "$test_reset_on_start" == "1" ]]; then
  db_path="$(read_env_value "DB_PATH" "analytics.db")"
  photo_dir="$(read_env_value "PHOTO_DIR" "captures")"
  rm -f "$db_path"
  mkdir -p "$photo_dir"
  rm -f "$photo_dir"/*.jpg
  echo "[diag] test reset: cleared DB ($db_path) and photos ($photo_dir)"
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
