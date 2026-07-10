import threading
from collections import deque
from typing import Any
import numpy as np


class MetricsTracker:
    """Thread-safe collector tracking usage, hit rate, latency and evaluation metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_requests = 0
        self.retrieval_hits = 0
        self.retrieval_total = 0
        self.retries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.latencies = deque(maxlen=1000)
        self.faithfulness_scores = deque(maxlen=1000)

    def record_request(self) -> None:
        with self._lock:
            self.total_requests += 1

    def record_retrieval(self, hits: int, total: int) -> None:
        with self._lock:
            self.retrieval_hits += hits
            self.retrieval_total += total

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def record_cache(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def record_latency(self, seconds: float) -> None:
        with self._lock:
            self.latencies.append(seconds * 1000.0)  # Store in milliseconds

    def record_faithfulness(self, score: float) -> None:
        with self._lock:
            self.faithfulness_scores.append(score)

    def get_metrics(self) -> dict[str, Any]:
        """Calculates hit rates, latencies percentiles, and average accuracy scores."""
        with self._lock:
            retrieval_hit_rate = (
                float(self.retrieval_hits / self.retrieval_total)
                if self.retrieval_total > 0
                else 0.0
            )
            cache_total = self.cache_hits + self.cache_misses
            cache_hit_rate = (
                float(self.cache_hits / cache_total)
                if cache_total > 0
                else 0.0
            )
            retry_rate = (
                float(self.retries / self.total_requests)
                if self.total_requests > 0
                else 0.0
            )
            avg_faithfulness = (
                float(sum(self.faithfulness_scores) / len(self.faithfulness_scores))
                if self.faithfulness_scores
                else 1.0
            )

            lats = list(self.latencies)
            if lats:
                p50 = float(np.percentile(lats, 50))
                p95 = float(np.percentile(lats, 95))
            else:
                p50, p95 = 0.0, 0.0

            return {
                "total_requests": self.total_requests,
                "retrieval_hit_rate": round(retrieval_hit_rate, 4),
                "cache_hit_rate": round(cache_hit_rate, 4),
                "retry_rate": round(retry_rate, 4),
                "avg_faithfulness_score": round(avg_faithfulness, 4),
                "latency_p50_ms": round(p50, 2),
                "latency_p95_ms": round(p95, 2),
            }


# Global metrics tracker singleton instance
metrics_tracker = MetricsTracker()
