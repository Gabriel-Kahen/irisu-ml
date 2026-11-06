"""Replay representation shared by the recovered event planner."""
import struct


def trace_bytes(values):
    return struct.pack(f"<{len(values)}I", *values)


def snapshot(observation):
    result = {name: int(observation.get(name, 0)) for name in (
        "tick", "score", "gauge", "level", "highest_chain",
        "qualifying_clear_count", "terminated", "truncated")}
    result.update({key: int(value) for key, value in observation.items()
                   if key.startswith("canonical_")})
    return result
