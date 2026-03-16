import shutil
import subprocess
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


class SystemMetricsService:
    def __init__(self) -> None:
        self._gpu_supported = bool(shutil.which("nvidia-smi"))

    def get_stats(self) -> dict[str, Any]:
        cpu_percent = self._get_cpu_percent()
        ram_percent = self._get_ram_percent()
        gpu_percent = self._get_gpu_percent()
        return {
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "gpu_percent": gpu_percent,
            "gpu_supported": self._gpu_supported,
        }

    @staticmethod
    def _get_cpu_percent() -> float | None:
        if psutil is None:
            return None
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return None

    @staticmethod
    def _get_ram_percent() -> float | None:
        if psutil is None:
            return None
        try:
            return float(psutil.virtual_memory().percent)
        except Exception:
            return None

    def _get_gpu_percent(self) -> float | None:
        if not self._gpu_supported:
            return None
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=1.0,
            ).strip()
            if not out:
                return None
            first = out.splitlines()[0].strip()
            return float(first)
        except Exception:
            return None
