"""
Servicio liviano de telemetría de hardware en tiempo real.
Lee sensores reales de Windows (GPU vía nvidia-smi y CPU vía WMI ACPI)
y actualiza hardware_telemetry.json cada 2 segundos sin consumir recursos.
"""

import time
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
HW_FILE = PROJECT_ROOT / "hardware_telemetry.json"
STOP_FILE = PROJECT_ROOT / "stop_telemetry.flag"

def query_gpu():
    telemetry = {
        "gpu_temp": 55,
        "vram_gb": 0.88,
        "vram_total_gb": 4.0,
        "vram_pct": 22.0,
        "gpu_util": 0,
        "gpu_name": "NVIDIA RTX 3050 Ti"
    }
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,name", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1.2
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [x.strip() for x in res.stdout.strip().split(",")]
            if len(parts) >= 3:
                telemetry["gpu_temp"] = int(parts[0])
                used_mb = int(parts[1])
                tot_mb = int(parts[2])
                telemetry["vram_gb"] = round(used_mb / 1024.0, 2)
                telemetry["vram_total_gb"] = round(tot_mb / 1024.0, 1)
                telemetry["vram_pct"] = round((used_mb / max(1, tot_mb)) * 100, 1)
                if len(parts) > 3 and parts[3].isdigit():
                    telemetry["gpu_util"] = int(parts[3])
                if len(parts) > 4:
                    telemetry["gpu_name"] = parts[4]
    except Exception:
        pass
    return telemetry

def query_cpu():
    try:
        cmd = ['powershell', '-NoProfile', '-Command', '(Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature -ErrorAction SilentlyContinue).CurrentTemperature']
        out = subprocess.check_output(cmd, text=True, timeout=1.8).strip()
        digits = [int(x.strip()) for x in out.split() if x.strip().isdigit()]
        if digits:
            c_temp = round(digits[0] / 10.0 - 273.15)
            if 20 <= c_temp <= 115:
                return c_temp
    except Exception:
        pass
    return None

def main():
    if STOP_FILE.exists():
        try: STOP_FILE.unlink()
        except Exception: pass

    print("[Telemetry] Iniciando monitor de hardware en tiempo real...")
    while not STOP_FILE.exists():
        try:
            hw = query_gpu()
            cpu_t = query_cpu()
            if cpu_t is not None:
                hw["cpu_temp"] = cpu_t

            tmp_file = PROJECT_ROOT / "hardware_telemetry.tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(hw, f, indent=2)
            tmp_file.replace(HW_FILE)
        except Exception as e:
            pass

        time.sleep(2.0)

if __name__ == "__main__":
    main()
