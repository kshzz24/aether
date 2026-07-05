import json
from collections import deque


class LoopDetector:

    def __init__(self, max_cycle_period=3, min_repeats=3) -> None:
        self._max_cycle_period = max_cycle_period
        self._min_repeats = min_repeats
        self._history = deque(maxlen=max_cycle_period *min_repeats)

    def record(self, tool_name, arguments, result) -> None:
        # `result` is matched exactly. A command whose output carries per-run
        # noise (timestamps, PIDs, elapsed ms, temp paths) will look different
        # every time and hide a real loop.
        # TODO(phase-N): normalize result noise before fingerprinting.
        fingerprint = (tool_name, json.dumps(arguments, sort_keys=True), result)
        self._history.append(fingerprint)

    def is_looping(self) -> bool:

        N = self._max_cycle_period
        min_reps = self._min_repeats
        for p in range(1, N + 1):
            window = p * min_reps
            if len(self._history) < window:
                continue
            tail = list(self._history)[-window:]
            block = tail[:p]

            if all(tail[i : i + p] == block for i in range(0, window, p)):
                return True
        return False
