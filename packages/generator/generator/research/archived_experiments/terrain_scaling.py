"""Preserved experiment helpers for regression tests; no training CLI."""
def terrain_text(source, scale):
    """Exact vertical scaling of this single box, preserving ground and faces."""
    if not 0.85 <= scale <= 1.15:
        raise ValueError('Outside declared experiment range')
    lines = []
    for line in source.splitlines():
        if line.startswith('v '):
            (_, x, y, z) = line.split()
            line = f'v {x} {y} {float(z) * scale:.10f}'
        lines.append(line)
    return '\n'.join(lines) + '\n'
