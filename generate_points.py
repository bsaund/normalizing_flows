#!/usr/bin/env python
"""
Functions for generating 2D training points used in normalizing flows.

Points can be sampled from:
  - A PNG file (create_points)
  - A rendered text string (create_points_from_text)
"""
from __future__ import print_function

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def create_uniform_points(num_points):
    return np.array(np.random.uniform(-1, 1, (num_points, 2)), dtype='float32')


def create_points(file_name, num_points):
    """
    Samples num_points points from the non-white portions of the image at file_name
    """
    with Image.open(file_name) as image:
        w, h = image.size
        pts = []
        while len(pts) < num_points:
            pt = np.random.rand(2).astype('f')
            x = min(int(pt[0] * w), w - 1)
            y = min(int((1 - pt[1]) * h), h - 1)

            pxl = image.getpixel((x, y))
            if pxl[0] != 255:
                pts.append(pt)
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    pts *= 5
    return pts


def _find_bold_font(size):
    """
    Return a bold TrueType font at the given pixel size.
    Tries matplotlib's font finder first (works on most systems),
    then falls back to known system paths, then PIL's built-in bitmap font.
    """
    # matplotlib already scanned the system for fonts — use its result
    try:
        path = fm.findfont(fm.FontProperties(weight='bold'))
        if path and os.path.isfile(path) and path.lower().endswith(('.ttf', '.otf')):
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    # Explicit fallback paths (common on Debian/Ubuntu)
    for candidate in [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
        '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf',
    ]:
        if os.path.isfile(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def render_text_image(text, font_size=120, padding=30):
    """
    Render text as a black-on-white PIL Image, sized to fit the text.

    Returns a PIL.Image in RGB mode.  The image is padded on all sides
    so no letter clipping occurs regardless of font metrics.
    """
    font = _find_bold_font(font_size)

    # Measure actual pixel bounds of the text (accounts for descenders etc.)
    tmp  = Image.new('RGB', (1, 1))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]

    img  = Image.new('RGB', (tw + 2 * padding, th + 2 * padding), 'white')
    ImageDraw.Draw(img).text(
        (padding - bbox[0], padding - bbox[1]),
        text, fill='black', font=font)
    return img


def create_points_from_text(text, num_points, font_size=120):
    """
    Sample 2-D points from the dark pixels of a rendered text string.

    Output is in the same coordinate space as create_points():
    centred at the origin and scaled so the bounding box spans roughly ±2.5.

    text       — string to render (any length, though 4–6 chars work best)
    num_points — how many points to return
    font_size  — controls letter thickness and detail (larger = more detail)
    """
    img    = render_text_image(text, font_size=font_size)
    w, h   = img.size
    pixels = img.load()

    pts = []
    while len(pts) < num_points:
        pt = np.random.rand(2).astype('f')
        x  = min(int(pt[0] * w), w - 1)
        y  = min(int((1 - pt[1]) * h), h - 1)
        if pixels[x, y][0] != 255:       # any non-white pixel (includes antialiasing)
            pts.append(pt)

    pts  = np.array(pts)
    pts -= np.mean(pts, axis=0)
    pts *= 5
    return pts


def visualize_data(pts):
    plt.scatter(pts[:, 0], pts[:, 1], s=5)
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    pts = create_points('two_moons.png', 10000)
    visualize_data(pts)
