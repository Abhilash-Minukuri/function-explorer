"""Verbal rules (stub)."""

def describe_a_change(old, new):
    if new == 0:
        return "a = 0 collapses the parabola to a line."
    trend = "narrower" if abs(new) > abs(old) else "wider"
    flip = " (flips downward)" if new < 0 else ""
    return f"Increasing |a| makes the parabola {trend}{flip}."
