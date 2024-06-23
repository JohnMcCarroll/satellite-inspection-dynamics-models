"""
Script for storing constant variable definitions.
"""

NAMED_STATE_RANGES = {
    "all": slice(0, 12),
    "position": slice(0, 3),
    "velocity": slice(3, 6),
    "inspected_points": slice(6, 7),
    "uninspected_points": slice(7, 10),
    "sun_angle": slice(10, 12),
}
