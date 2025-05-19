from typing import Tuple
import pyglet


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
        int(color1[2] + (color2[2] - color1[2]) * segment_t),
    )


# Cache for storing loaded images by activity name
_image_cache = {}


def load_activity_images(activity_name: str) -> list[pyglet.image.AbstractImage]:
    # Return cached images if already loaded
    if activity_name in _image_cache:
        return _image_cache[activity_name]

    images = []
    for i in range(1, 3):
        try:
            img = pyglet.image.load(f"img/{activity_name}_{i}.png")
        except Exception as e:
            print(f"Failed to load img/{activity_name}_{i}.png: {e}")
            img = pyglet.image.SolidColorImagePattern((255, 255, 255, 255)).create_image(50, 50)

        # Set anchor points to center-bottom for consistent positioning
        img.anchor_x, img.anchor_y = img.width // 2, img.height
        images.append(img)

    # Cache the images before returning
    _image_cache[activity_name] = images
    return images
