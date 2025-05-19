from typing import Tuple


def lerp_colors(colors: list[Tuple[int, int, int]], t: float) -> Tuple[int, int, int]:
    """Linearly interpolate between a list of colors based on t (0 to 1)."""
    
    if len(colors) == 0:
        raise ValueError("Colors list cannot be empty")
    if len(colors) == 1:
        return colors[0]
    
    # Clamp t between 0 and 1
    t = max(0.0, min(1.0, t))
    
    # Calculate which segment t falls in
    segment_size = 1.0 / (len(colors) - 1)
    segment = min(int(t / segment_size), len(colors) - 2)
    
    # Calculate t within the segment (0 to 1)
    segment_t = (t - segment * segment_size) / segment_size
    
    # Get the two colors to interpolate between
    color1 = colors[segment]
    color2 = colors[segment + 1]
    
    # Perform the interpolation
    return (
        int(color1[0] + (color2[0] - color1[0]) * segment_t),
        int(color1[1] + (color2[1] - color1[1]) * segment_t),
        int(color1[2] + (color2[2] - color1[2]) * segment_t)
    )