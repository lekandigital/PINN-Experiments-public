"""
Performance Tracker - FPS tracking and timing utilities.

Provides rolling averages and statistics for inference performance.
"""

import time
from collections import deque
from typing import Optional


class PerformanceTracker:
    """
    Tracks inference performance over time.
    
    Maintains a rolling window of timing samples and provides
    statistics like average FPS, min/max times, etc.
    """
    
    def __init__(self, window_size: int = 60):
        """
        Initialize tracker.
        
        Args:
            window_size: Number of samples to keep in rolling window
        """
        self._window_size = window_size
        self._times_ms = deque(maxlen=window_size)
        self._start_time: Optional[float] = None
        self._total_frames = 0
        self._total_time_ms = 0.0
    
    def start_frame(self) -> None:
        """Mark the start of a frame."""
        self._start_time = time.perf_counter()
    
    def end_frame(self) -> float:
        """
        Mark the end of a frame and record timing.
        
        Returns:
            Frame time in milliseconds
        """
        if self._start_time is None:
            return 0.0
        
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self._times_ms.append(elapsed_ms)
        self._total_frames += 1
        self._total_time_ms += elapsed_ms
        self._start_time = None
        
        return elapsed_ms
    
    def record_time(self, time_ms: float) -> None:
        """
        Record a frame time directly.
        
        Args:
            time_ms: Frame time in milliseconds
        """
        self._times_ms.append(time_ms)
        self._total_frames += 1
        self._total_time_ms += time_ms
    
    def get_average_fps(self) -> float:
        """Get average FPS over the rolling window."""
        if len(self._times_ms) == 0:
            return 0.0
        
        avg_ms = sum(self._times_ms) / len(self._times_ms)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0
    
    def get_average_ms(self) -> float:
        """Get average frame time in milliseconds."""
        if len(self._times_ms) == 0:
            return 0.0
        
        return sum(self._times_ms) / len(self._times_ms)
    
    def get_min_ms(self) -> float:
        """Get minimum frame time in milliseconds."""
        if len(self._times_ms) == 0:
            return 0.0
        return min(self._times_ms)
    
    def get_max_ms(self) -> float:
        """Get maximum frame time in milliseconds."""
        if len(self._times_ms) == 0:
            return 0.0
        return max(self._times_ms)
    
    def get_current_fps(self) -> float:
        """Get current FPS (based on last frame)."""
        if len(self._times_ms) == 0:
            return 0.0
        
        last_ms = self._times_ms[-1]
        return 1000.0 / last_ms if last_ms > 0 else 0.0
    
    def get_total_frames(self) -> int:
        """Get total number of frames tracked."""
        return self._total_frames
    
    def get_total_time_ms(self) -> float:
        """Get total time tracked in milliseconds."""
        return self._total_time_ms
    
    def get_lifetime_fps(self) -> float:
        """Get FPS averaged over all tracked frames."""
        if self._total_frames == 0 or self._total_time_ms == 0:
            return 0.0
        
        avg_ms = self._total_time_ms / self._total_frames
        return 1000.0 / avg_ms
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self._times_ms.clear()
        self._start_time = None
        self._total_frames = 0
        self._total_time_ms = 0.0
    
    def get_stats(self) -> dict:
        """
        Get all statistics as a dictionary.
        
        Returns:
            Dict with all performance metrics
        """
        return {
            "average_fps": self.get_average_fps(),
            "current_fps": self.get_current_fps(),
            "average_ms": self.get_average_ms(),
            "min_ms": self.get_min_ms(),
            "max_ms": self.get_max_ms(),
            "total_frames": self._total_frames,
            "total_time_ms": self._total_time_ms,
            "lifetime_fps": self.get_lifetime_fps(),
            "window_size": len(self._times_ms),
        }
    
    def __str__(self) -> str:
        """String representation of current stats."""
        stats = self.get_stats()
        return (
            f"FPS: {stats['average_fps']:.1f} "
            f"(avg: {stats['average_ms']:.2f}ms, "
            f"min: {stats['min_ms']:.2f}ms, "
            f"max: {stats['max_ms']:.2f}ms)"
        )


# Global tracker instance
_global_tracker: Optional[PerformanceTracker] = None


def get_tracker() -> PerformanceTracker:
    """Get the global performance tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


def reset_tracker() -> None:
    """Reset the global performance tracker."""
    global _global_tracker
    if _global_tracker is not None:
        _global_tracker.reset()
