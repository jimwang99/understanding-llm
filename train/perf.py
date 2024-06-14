import time


class PerfMonitor:
    def __init__(
        self,
        logger,
        name: str,
        print_iterations: int = 100,
        unit: str = "ms",
        level: str = "info",
    ):
        self.logger = logger
        self.name = name
        self.print_iterations = print_iterations

        self.idx = 0

        self.acc_latency = 0.0
        self.max_latency = -1.0
        self.min_latency = 1e9
        self.begin_time = time.time()

        self.unit = unit
        self.unit_per_second = {
            "s": 1.0,
            "ms": 1000.0,
            "us": 1000000.0,
        }[self.unit]

        self.log = {
            "warning": logger.warning,
            "info": logger.info,
            "debug": logger.debug,
        }[level]

        self.latency_as_unit = 0.0
        self.max_latency_as_unit = 0.0
        self.min_latency_as_unit = 0.0

    def _end_time(self):
        end_time = time.time()
        latency = end_time - self.begin_time
        self.acc_latency += latency
        self.max_latency = latency if latency > self.max_latency else self.max_latency
        self.min_latency = latency if latency < self.min_latency else self.min_latency
        return end_time

    def _print(self):
        self.avg_latency_as_unit = (
            self.acc_latency * self.unit_per_second / self.print_iterations
        )
        self.max_latency_as_unit = self.max_latency * self.unit_per_second
        self.min_latency_as_unit = self.min_latency * self.unit_per_second
        self.log(
            f"<PerfMonitor> {self.name} | Index {self.idx} | Average {self.avg_latency_as_unit:.2f}{self.unit} | Max {self.max_latency_as_unit:.2f}{self.unit} | Min {self.min_latency_as_unit:.2f}{self.unit}"
        )
        self.acc_latency = 0.0
        self.max_latency = -1.0
        self.min_latency = 1e9

    def once(self):
        self.acc_latency = time.time() - self.begin_time
        self.latency_as_unit = self.acc_latency * self.unit_per_second
        self.log(
            f"<PerfMonitor> {self.name} | Latency {self.latency_as_unit:.2f}{self.unit}"
        )

    def begin(self):
        self.begin_time = time.time()

    def end(self):
        self._end_time()
        self.idx += 1
        if self.idx % self.print_iterations == 0:
            self._print()

    def loop(self):
        self.begin_time = time.time() if self.idx == 0 else self._end_time()
        self.idx += 1
        if self.idx % self.print_iterations == 0:
            self._print()


def test_perf_monitor():
    from loguru import logger

    pm = PerfMonitor(logger, "once")
    time.sleep(0.1)
    pm.once()

    pm = PerfMonitor(logger, "begin-end", print_iterations=20)
    for _ in range(100):
        pm.begin()
        time.sleep(0.01)
        pm.end()

    pm = PerfMonitor(logger, "loop", print_iterations=20)
    for _ in range(100):
        pm.loop()
        time.sleep(0.01)
