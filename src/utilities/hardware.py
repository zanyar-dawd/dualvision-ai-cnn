"""Sample NVIDIA telemetry and stop conservatively on thermal/hardware faults."""
import csv
import json
import subprocess
import time
from pathlib import Path


class HardwareStop(RuntimeError):
    """Training must stop and preserve its current state."""


class TrainingBudgetExceeded(RuntimeError):
    """The wall-clock training budget expired at a batch boundary."""


class TrainingGuard:
    def __init__(self, log_path, max_minutes, max_temperature=85, interval=5.0, elapsed_offset=0.0):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.deadline = self.started + max_minutes * 60
        self.max_temperature = max_temperature  # conservative project policy, not a rated hardware limit
        self.interval = interval
        self.last_sample = float('-inf')
        self.samples = 0
        self.peak_temperature = 0
        self.peak_used_mib = 0
        self.elapsed_offset = elapsed_offset
        if elapsed_offset and self.log_path.exists():
            for line in self.log_path.read_text(encoding='utf-8').splitlines():
                previous = json.loads(line)
                self.samples += 1
                self.peak_temperature = max(self.peak_temperature, previous['temperature_c'])
                self.peak_used_mib = max(self.peak_used_mib, previous['memory_used_mib'])

    def check(self, force=False):
        now = time.monotonic()
        if now >= self.deadline:
            raise TrainingBudgetExceeded('Training wall-clock budget reached.')
        if not force and now - self.last_sample < self.interval:
            return
        self.last_sample = now
        try:
            output = subprocess.check_output([
                'nvidia-smi', '--id=0',
                '--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw',
                '--format=csv,noheader,nounits'], text=True, timeout=10,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            fields = next(csv.reader(output.strip().splitlines()))
            temperature, used, total, utilization, power = map(float, fields)
        except (OSError, subprocess.SubprocessError, ValueError, StopIteration) as error:
            raise HardwareStop(f'GPU telemetry unavailable: {error}') from error
        record = {'elapsed_seconds': self.elapsed_offset + now - self.started, 'temperature_c': temperature,
                  'memory_used_mib': used, 'memory_total_mib': total,
                  'gpu_utilization_percent': utilization, 'power_w': power}
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record) + '\n')
        self.samples += 1
        self.peak_temperature = max(self.peak_temperature, temperature)
        self.peak_used_mib = max(self.peak_used_mib, used)
        if temperature >= self.max_temperature:
            raise HardwareStop(f'GPU reached {temperature:.0f} C; project stop threshold is {self.max_temperature} C.')
        if used >= total * 0.95:
            raise HardwareStop(f'GPU memory pressure: {used:.0f}/{total:.0f} MiB used (95% stop policy).')

    def summary(self):
        return {'sample_interval_seconds': self.interval, 'sample_count': self.samples,
                'temperature_stop_c': self.max_temperature, 'memory_stop_fraction': 0.95,
                'peak_temperature_c': self.peak_temperature, 'peak_device_memory_used_mib': self.peak_used_mib}
