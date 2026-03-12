import os
import sys
import csv
import threading
import subprocess
from datetime import datetime


class GPUMonitor:
    """Background GPU utilization monitor using nvidia-smi.

    Runs as a daemon thread, logging GPU stats to CSV and printing
    periodic summaries to stdout. Designed as a context manager.
    """

    QUERY_FIELDS = (
        "index,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,temperature.gpu,power.draw"
    )
    CSV_HEADER = [
        "timestamp", "node_rank", "gpu_index", "gpu_name",
        "gpu_util_pct", "mem_util_pct", "mem_used_mib", "mem_total_mib",
        "temperature_c", "power_draw_w",
    ]

    def __init__(self, log_dir, interval=30.0, console_interval=60.0,
                 nodelist=None, n_nodes=1):
        self.log_dir = log_dir
        self.interval = interval
        self.console_interval = console_interval
        self.n_nodes = n_nodes
        self.nodelist = nodelist
        self._stop_event = threading.Event()
        self._thread = None
        self._csv_file = None
        self._csv_writer = None
        self._lock = threading.Lock()
        self._latest = None
        self._last_console_time = 0.0

    def __enter__(self):
        csv_path = os.path.join(self.log_dir, "gpu_utilization.csv")
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADER)
        self._csv_file.flush()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[GPU] Monitor started — logging to {csv_path}", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._csv_file is not None:
            self._csv_file.close()
        print("[GPU] Monitor stopped", flush=True)
        return False

    def get_latest(self):
        with self._lock:
            return self._latest

    def _monitor_loop(self):
        import time
        while not self._stop_event.is_set():
            try:
                data = self._collect_gpu_data()
                if data:
                    self._write_csv(data)
                    with self._lock:
                        self._latest = data
                    now = time.monotonic()
                    if now - self._last_console_time >= self.console_interval:
                        self._print_console_summary(data)
                        self._last_console_time = now
            except Exception as e:
                import traceback
                print(f"[GPU] Warning: monitoring error: {e}", file=sys.stderr, flush=True)
                traceback.print_exc()
            self._stop_event.wait(self.interval)

    def _collect_gpu_data(self):
        query_args = [
            "nvidia-smi",
            f"--query-gpu={self.QUERY_FIELDS}",
            "--format=csv,noheader,nounits",
        ]
        if self.n_nodes > 1:
            cmd = ["flux", "exec", "-r", "all", "-l"] + query_args
            timeout = 30
        else:
            cmd = query_args
            timeout = 10

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            print(f"[GPU] Warning: nvidia-smi returned {result.returncode}: "
                  f"{result.stderr.strip()}", file=sys.stderr, flush=True)
            return None
        return self._parse_output(result.stdout)

    def _parse_output(self, output):
        rows = []
        timestamp = datetime.now().isoformat(timespec="seconds")
        for line in output.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Multi-node: flux exec -l prefixes "RANK: " to each line
            if self.n_nodes > 1 and ": " in line:
                rank_str, rest = line.split(": ", 1)
                try:
                    rank = int(rank_str)
                except ValueError:
                    rank = rank_str
            else:
                rank = 0
                rest = line

            parts = [p.strip() for p in rest.split(",")]
            if len(parts) < 8:
                continue

            rows.append({
                "timestamp": timestamp,
                "node_rank": rank,
                "gpu_index": parts[0],
                "gpu_name": parts[1],
                "gpu_util_pct": parts[2],
                "mem_util_pct": parts[3],
                "mem_used_mib": parts[4],
                "mem_total_mib": parts[5],
                "temperature_c": parts[6],
                "power_draw_w": parts[7],
            })
        return rows

    def _write_csv(self, data):
        for row in data:
            self._csv_writer.writerow([row[h] for h in self.CSV_HEADER])
        self._csv_file.flush()

    def _print_console_summary(self, data):
        if not data:
            return
        utils = []
        mem_used = []
        mem_total = []
        for row in data:
            try:
                utils.append(float(row["gpu_util_pct"]))
                mem_used.append(float(row["mem_used_mib"]))
                mem_total.append(float(row["mem_total_mib"]))
            except (ValueError, TypeError):
                continue
        if not utils:
            return
        n = len(utils)
        avg_util = sum(utils) / n
        min_util = min(utils)
        max_util = max(utils)
        avg_mem = sum(mem_used) / n / 1024
        avg_mem_total = sum(mem_total) / n / 1024 if mem_total else 0
        now = datetime.now().strftime("%H:%M:%S")
        summary = (f"[GPU] {now} | {n} GPUs | "
                   f"Util: avg={avg_util:.0f}% min={min_util:.0f}% max={max_util:.0f}% | "
                   f"Mem: avg={avg_mem:.1f}/{avg_mem_total:.1f} GiB")

        if self.n_nodes > 1:
            # Per-node breakdown
            by_node = {}
            for row in data:
                rank = row["node_rank"]
                by_node.setdefault(rank, []).append(row)
            parts = []
            for rank in sorted(by_node.keys()):
                node_utils = []
                for r in by_node[rank]:
                    try:
                        node_utils.append(float(r["gpu_util_pct"]))
                    except (ValueError, TypeError):
                        continue
                if node_utils:
                    node_avg = sum(node_utils) / len(node_utils)
                    parts.append(f"N{rank}={node_avg:.0f}%")
            if parts:
                summary += " | " + " ".join(parts)

        print(summary, flush=True)
