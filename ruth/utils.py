import logging
import time
from datetime import datetime, timedelta


def round_timedelta(td: timedelta, freq: timedelta):
    return freq * round(td / freq)


def round_datetime(dt: datetime, freq: timedelta):
    if freq / timedelta(hours=1) > 1:
        assert False, "Too rough rounding frequency"
    elif freq / timedelta(minutes=1) > 1:
        td = timedelta(minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond)
    elif freq / timedelta(seconds=1) > 1:
        td = timedelta(seconds=dt.second, microseconds=dt.microsecond)
    else:
        assert False, "Too fine rounding frequency"

    rest = dt - td
    td_rounded = round_timedelta(td, freq)

    return rest + td_rounded


def get_speed_limit_kph(highway_type: str, vehicle_type: str) -> float:
    """
    Returns the speed limit in km/h for a given highway type and vehicle type.
    """
    # Define speed limits based on typical European/US limits, adjusted for realism
    speed_limits = {
        'motorway': {'car': 130, 'truck': 80},
        'motorway_link': {'car': 100, 'truck': 60},
        'trunk': {'car': 100, 'truck': 70},
        'trunk_link': {'car': 80, 'truck': 50},
        'primary': {'car': 90, 'truck': 60},
        'primary_link': {'car': 70, 'truck': 50},
        'secondary': {'car': 80, 'truck': 50},
        'secondary_link': {'car': 60, 'truck': 40},
        'tertiary': {'car': 70, 'truck': 40},
        'tertiary_link': {'car': 50, 'truck': 30},
        'unclassified': {'car': 50, 'truck': 30},
        'residential': {'car': 50, 'truck': 30},
        'living_street': {'car': 20, 'truck': 20},
        'service': {'car': 30, 'truck': 30},
        'track': {'car': 30, 'truck': 20},
        'footway': {'car': 10, 'truck': 10},
        'cycleway': {'car': 10, 'truck': 10},
        'path': {'car': 10, 'truck': 10},
        'pedestrian': {'car': 10, 'truck': 10},
        'steps': {'car': 5, 'truck': 5},
    }
    # Default to 50 km/h if highway_type or vehicle_type not found
    return speed_limits.get(highway_type, {}).get(vehicle_type, 50)


def is_root_debug_logging() -> bool:
    """
    Returns true if the global (root) logger has at least `logging.DEBUG` level.
    """
    return logging.getLogger().isEnabledFor(logging.DEBUG)


class Timer:

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        self.end = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()

    @property
    def duration_ms(self):
        assert self.end is not None, "Trying to call duration on unfinished timer."
        return (self.end - self.start) * 1000


class TimerSet:

    def __init__(self):
        self.timers = []

    def get(self, name):
        self.timers.append(Timer(name))
        return self.timers[-1]

    def collect(self):
        return dict((timer.name, timer.duration_ms) for timer in self.timers)
