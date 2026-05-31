from __future__ import annotations

import argparse
import csv
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor CPU/RAM/disk/GPU usage during context-seg experiments")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--count", type=int, default=0, help="Number of samples; 0 means run forever")
    parser.add_argument("--top", type=int, default=8, help="Show top N CPU processes")
    parser.add_argument("--log", type=str, default="context-seg/runs/system_monitor.csv", help="Optional CSV log path")
    parser.add_argument("--info", action="store_true", help="Print machine info once and exit")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear terminal between refreshes")
    parser.add_argument("--max-cpu-percent", type=float, default=90, help="Warn/stop when total CPU exceeds this")
    parser.add_argument("--max-ram-percent", type=float, default=90, help="Warn/stop when RAM usage exceeds this")
    parser.add_argument("--max-disk-percent", type=float, default=95, help="Warn/stop when disk usage exceeds this")
    parser.add_argument("--max-gpu-mem-percent", type=float, default=95, help="Warn/stop when any GPU memory exceeds this")
    parser.add_argument(
        "--kill-command",
        type=str,
        default="train.py",
        help="Terminate processes whose command line contains this string when a threshold is exceeded",
    )
    parser.add_argument("--terminate-grace-seconds", type=float, default=10.0)
    parser.add_argument("--kill-once", action="store_true", help="Exit monitor after terminating matching processes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    psutil = import_psutil()

    if args.info:
        print_machine_info(psutil)
        return

    log_writer = None
    log_file = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "time",
                "cpu_percent",
                "ram_used_gb",
                "ram_total_gb",
                "ram_percent",
                "disk_percent",
                "gpu_summary",
            ],
        )
        log_writer.writeheader()

    try:
        print_machine_info(psutil)
        print("\nPress Ctrl+C to stop.\n")
        sample_idx = 0
        while args.count <= 0 or sample_idx < args.count:
            sample_idx += 1
            # cpu_percent(interval=...) blocks for a stable interval sample.
            cpu_percent = psutil.cpu_percent(interval=args.interval)
            snapshot = collect_snapshot(psutil, cpu_percent, args.top)
            if not args.no_clear:
                clear_terminal()
            render_snapshot(snapshot)
            if log_writer is not None:
                log_writer.writerow(snapshot["log"])
                log_file.flush()
            alerts = threshold_alerts(snapshot, args)
            if alerts:
                print("\nALERT")
                for alert in alerts:
                    print(f"  {alert}")
                if args.kill_command:
                    killed = terminate_matching_processes(psutil, args.kill_command, args.terminate_grace_seconds)
                    print(f"  terminate pattern: {args.kill_command!r}")
                    print(f"  matched processes: {len(killed)}")
                    for proc in killed:
                        print(f"    PID {proc['pid']}: {proc['name']} | {proc['cmdline']}")
                    if args.kill_once:
                        break
                else:
                    print("  No process terminated. Pass --kill-command to stop a training process automatically.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if log_file is not None:
            log_file.close()


def import_psutil():
    try:
        import psutil
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: psutil. Install it in the active environment with:\n"
            "  python -m pip install psutil\n"
            "or reinstall context-seg after pulling the updated pyproject.toml."
        ) from exc
    return psutil


def print_machine_info(psutil) -> None:
    print("System")
    print(f"  OS: {platform.platform()}")
    print(f"  Python: {platform.python_version()} ({sys.executable})")
    print(f"  CPU: {platform.processor() or 'unknown'}")
    print(f"  CPU cores: physical={psutil.cpu_count(logical=False)} logical={psutil.cpu_count(logical=True)}")
    ram = psutil.virtual_memory()
    print(f"  RAM: {bytes_to_gb(ram.total):.2f} GB")
    disk_path = str(Path.cwd())
    disk = disk_usage(disk_path)
    print(f"  Disk ({disk_path}): {bytes_to_gb(disk.free):.2f} GB free / {bytes_to_gb(disk.total):.2f} GB")
    gpus = query_gpus()
    if gpus:
        print("GPU")
        for gpu in gpus:
            print(
                f"  GPU {gpu['index']}: {gpu['name']} | "
                f"mem {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB | "
                f"util {gpu['utilization_gpu']}%"
            )
    else:
        print("GPU")
        print("  nvidia-smi not available or no NVIDIA GPU detected")


def collect_snapshot(psutil, cpu_percent: float, top_n: int) -> dict:
    ram = psutil.virtual_memory()
    disk = disk_usage(str(Path.cwd()))
    gpus = query_gpus()
    top_processes = top_cpu_processes(psutil, top_n)
    gpu_summary = "; ".join(
        f"{gpu['index']}:{gpu['utilization_gpu']}%:{gpu['memory_used_mb']}/{gpu['memory_total_mb']}MB"
        for gpu in gpus
    )
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_percent": cpu_percent,
        "ram": ram,
        "disk": disk,
        "gpus": gpus,
        "top_processes": top_processes,
        "log": {
            "time": datetime.now().isoformat(timespec="seconds"),
            "cpu_percent": f"{cpu_percent:.1f}",
            "ram_used_gb": f"{bytes_to_gb(ram.used):.2f}",
            "ram_total_gb": f"{bytes_to_gb(ram.total):.2f}",
            "ram_percent": f"{ram.percent:.1f}",
            "disk_percent": f"{disk.percent:.1f}",
            "gpu_summary": gpu_summary,
        },
    }


def render_snapshot(snapshot: dict) -> None:
    print(f"Context-Seg System Monitor | {snapshot['time']}")
    print("=" * 78)
    print(f"CPU total: {snapshot['cpu_percent']:.1f}%")
    ram = snapshot["ram"]
    print(f"RAM: {bytes_to_gb(ram.used):.2f}/{bytes_to_gb(ram.total):.2f} GB ({ram.percent:.1f}%)")
    disk = snapshot["disk"]
    print(f"Disk: {bytes_to_gb(disk.used):.2f}/{bytes_to_gb(disk.total):.2f} GB ({disk.percent:.1f}%)")

    print("\nGPU")
    if snapshot["gpus"]:
        for gpu in snapshot["gpus"]:
            print(
                f"  [{gpu['index']}] {gpu['name']} | "
                f"util {gpu['utilization_gpu']:>3}% | "
                f"mem {gpu['memory_used_mb']:>6}/{gpu['memory_total_mb']:<6} MB | "
                f"temp {gpu['temperature_gpu']:>3} C | "
                f"power {gpu['power_draw']}"
            )
    else:
        print("  nvidia-smi not available or no NVIDIA GPU detected")

    print("\nTop CPU processes")
    print(f"  {'PID':>8} {'CPU%':>7} {'RAM MB':>9}  Name")
    for proc in snapshot["top_processes"]:
        print(f"  {proc['pid']:>8} {proc['cpu_percent']:>7.1f} {proc['rss_mb']:>9.1f}  {proc['name']}")


def top_cpu_processes(psutil, top_n: int) -> list[dict]:
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
        try:
            info = proc.info
            rss = info["memory_info"].rss if info.get("memory_info") is not None else 0
            processes.append(
                {
                    "pid": info["pid"],
                    "name": info.get("name") or "",
                    "cpu_percent": float(info.get("cpu_percent") or 0.0),
                    "rss_mb": rss / (1024 * 1024),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    processes.sort(key=lambda item: item["cpu_percent"], reverse=True)
    return processes[:top_n]


def threshold_alerts(snapshot: dict, args: argparse.Namespace) -> list[str]:
    alerts = []
    if args.max_cpu_percent is not None and snapshot["cpu_percent"] >= args.max_cpu_percent:
        alerts.append(f"CPU {snapshot['cpu_percent']:.1f}% >= {args.max_cpu_percent:.1f}%")
    ram_percent = float(snapshot["ram"].percent)
    if args.max_ram_percent is not None and ram_percent >= args.max_ram_percent:
        alerts.append(f"RAM {ram_percent:.1f}% >= {args.max_ram_percent:.1f}%")
    disk_percent = float(snapshot["disk"].percent)
    if args.max_disk_percent is not None and disk_percent >= args.max_disk_percent:
        alerts.append(f"Disk {disk_percent:.1f}% >= {args.max_disk_percent:.1f}%")
    if args.max_gpu_mem_percent is not None:
        for gpu in snapshot["gpus"]:
            used = float(gpu["memory_used_mb"])
            total = float(gpu["memory_total_mb"])
            percent = 100.0 * used / total if total else 0.0
            if percent >= args.max_gpu_mem_percent:
                alerts.append(
                    f"GPU {gpu['index']} memory {percent:.1f}% >= {args.max_gpu_mem_percent:.1f}%"
                )
    return alerts


def terminate_matching_processes(psutil, command_pattern: str, grace_seconds: float) -> list[dict]:
    """Terminate processes matching command_pattern, excluding this monitor process.

    这个函数只在用户显式传入 --kill-command 时运行，避免监控脚本误杀无关进程。
    """
    current_pid = os.getpid()
    matches = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid == current_pid:
                continue
            cmdline_parts = proc.info.get("cmdline") or []
            cmdline = " ".join(str(part) for part in cmdline_parts)
            if command_pattern not in cmdline:
                continue
            matches.append(
                {
                    "proc": proc,
                    "pid": proc.pid,
                    "name": proc.info.get("name") or "",
                    "cmdline": cmdline,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    for item in matches:
        try:
            item["proc"].terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    gone, alive = psutil.wait_procs([item["proc"] for item in matches], timeout=grace_seconds)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return [{key: value for key, value in item.items() if key != "proc"} for item in matches]


def query_gpus() -> list[dict]:
    if shutil.which("nvidia-smi") is None:
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        index, name, util, mem_used, mem_total, temp, power = parts
        gpus.append(
            {
                "index": index,
                "name": name,
                "utilization_gpu": int(float(util)),
                "memory_used_mb": int(float(mem_used)),
                "memory_total_mb": int(float(mem_total)),
                "temperature_gpu": int(float(temp)),
                "power_draw": f"{power} W",
            }
        )
    return gpus


def bytes_to_gb(value: int | float) -> float:
    return float(value) / (1024 ** 3)


def disk_usage(path: str) -> SimpleNamespace:
    usage = shutil.disk_usage(path)
    percent = 100.0 * usage.used / usage.total if usage.total else 0.0
    return SimpleNamespace(total=usage.total, used=usage.used, free=usage.free, percent=percent)


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    main()
