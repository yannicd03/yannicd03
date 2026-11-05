# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10", "numpy>=1.26"]
# ///
"""Monet-inspired pixel-art banner: a water-lily pond under a Japanese footbridge.

Everything is drawn on a small pixel grid with a fixed palette, animated with
periodic functions of the frame index (so the GIF loops seamlessly), then
upscaled with nearest-neighbour so the pixels stay crisp on GitHub.

    uv run generate.py            # writes assets/monet-pond.gif + assets/preview.png
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 240, 80  # pixel-art canvas
HORIZON = 42  # first water row
SCALE = 4  # output = 960 x 320
FRAMES = 60
FRAME_MS = 100
SEED = 1899  # the year Monet started the footbridge series
SIGNATURE = "Yannic '26"

OUT = Path(__file__).parent / "assets"

COLORS = {
    # sky, top -> bottom
    "sky0": "#bcd3e6", "sky1": "#d9c7e3", "sky2": "#efc9b8", "sky3": "#f6e3c8",
    # foliage, dark -> light
    "f0": "#1c3329", "f1": "#2a5038", "f2": "#3b6b43", "f3": "#55884b",
    "f4": "#7aa257", "f5": "#a4bf66", "f6": "#cfd47a", "f7": "#efe39a",
    # cool shadows and teal reflections
    "b0": "#22394d", "b1": "#34596a", "b2": "#4f7c86",
    "t0": "#2a4d4f", "t1": "#3d6b62", "t2": "#56857a",
    # bridge
    "br0": "#244a40", "br1": "#3f7c69", "br2": "#62a287", "br3": "#9fd0b1",
    # water
    "w0": "#26405f", "w1": "#36577f", "w2": "#4c72a0", "w3": "#6f93c0",
    "w4": "#98b0da", "w5": "#bfbbe3", "w6": "#e6e0f3",
    # lavender accents
    "v0": "#6c5f93", "v1": "#9686bb", "v2": "#c3a6d4",
    # lily pads
    "p0": "#2c5f45", "p1": "#467f4f", "p2": "#6fa35a", "p3": "#9cc46a",
    # water lilies
    "fw": "#ffffff", "fl": "#fbe0e6", "fp": "#f2a7b8", "fr": "#e07595", "fy": "#f4cf5d",
    # signature
    "sig": "#c4523b",
}
NAMES = list(COLORS)
C = {name: i for i, name in enumerate(NAMES)}
RGB = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in COLORS.values()], float)
EMPTY = -1

rng = np.random.default_rng(SEED)


def nearest(rgb):
    return int(np.argmin(((RGB - rgb) ** 2).sum(axis=1)))


# Colour transforms as palette lookup tables.
WATER_TINT = np.array([nearest(c * 0.58 + np.array([60, 95, 140]) * 0.42) for c in RGB], np.uint8)
DARKEN = np.array([nearest(c * 0.68 + np.array([20, 35, 60]) * 0.12) for c in RGB], np.uint8)


def ramp_pick(ramp, v):
    """Map v in [0, 1] (array or scalar) onto a list of colour names."""
    idx = np.clip((np.asarray(v) * len(ramp)).astype(int), 0, len(ramp) - 1)
    lut = np.array([C[n] for n in ramp], np.uint8)
    return lut[idx]


def smooth_noise(n, step):
    pts = rng.random(n // step + 2)
    xs = np.arange(n) / step
    i = xs.astype(int)
    fr = (1 - np.cos((xs - i) * np.pi)) / 2
    return pts[i] * (1 - fr) + pts[i + 1] * fr


def dab(canvas, x, y, color, length, angle, mask=None):
    dx, dy = math.cos(angle), math.sin(angle)
    for k in range(length):
        xx, yy = int(round(x + dx * k)), int(round(y + dy * k))
        if 0 <= xx < W and 0 <= yy < H and (mask is None or mask[yy, xx]):
            canvas[yy, xx] = color


# --------------------------------------------------------------------------- background

def paint_background():
    bg = np.zeros((H, W), np.uint8)
    Y, X = np.mgrid[0:H, 0:W]

    # Sky: soft gradient, dithered with noise, then horizontal strokes.
    sky_ramp = ["sky0", "sky1", "sky2", "sky3"]
    t = Y[:HORIZON] / 22 + rng.normal(0, 0.1, (HORIZON, W))
    bg[:HORIZON] = ramp_pick(sky_ramp, t)
    for _ in range(160):
        x, y = rng.uniform(0, W), rng.uniform(0, 20)
        dab(bg, x, y, ramp_pick(sky_ramp, y / 22 + rng.normal(0, 0.15)), rng.integers(3, 7), 0)

    # Foliage band with an irregular top edge that climbs at the sides.
    xs = np.arange(W)
    top = 12 + 9 * smooth_noise(W, 18) + 3 * smooth_noise(W, 5) - 4
    top -= 24 * np.clip(1 - xs / 42, 0, 1) ** 1.4 + 24 * np.clip(1 - (W - 1 - xs) / 42, 0, 1) ** 1.4
    top = np.clip(top, 0, HORIZON - 6).astype(int)
    mask = (Y >= top[None, :]) & (Y < HORIZON)

    fol_ramp = ["f0", "b0", "f1", "t0", "f2", "b1", "f3", "f4", "f5", "f6", "f7"]
    depth = np.clip((Y - top[None, :]) / np.maximum(HORIZON - top[None, :], 1), 0, 1)
    light = (1 - depth) * 0.9 + rng.normal(0, 0.13, (H, W))
    bg[mask] = ramp_pick(fol_ramp, light)[mask]

    pts = np.argwhere(mask)
    for _ in range(1700):
        y, x = pts[rng.integers(len(pts))]
        v = (1 - depth[y, x]) * 0.9 + rng.normal(0, 0.16)
        dab(bg, x, y, ramp_pick(fol_ramp, v), rng.integers(2, 5), rng.uniform(-math.pi, math.pi), mask)

    # Hanging wisteria-like blossoms in the upper foliage.
    upper = np.argwhere(mask & (depth < 0.55))
    for _ in range(110):
        y, x = upper[rng.integers(len(upper))]
        color = C[rng.choice(["v1", "v2", "fl", "v0"])]
        dab(bg, x, y, color, rng.integers(2, 4), math.pi / 2, mask)

    # Dark waterline under the far bank.
    for y in (HORIZON - 2, HORIZON - 1):
        row = rng.random(W) < (0.55 if y == HORIZON - 2 else 0.85)
        bg[y, row] = DARKEN[bg[y, row]]
    return bg


def paint_bridge(bg):
    x0, x1, arch = 34, 206, 14.5
    for x in range(x0, x1 + 1):
        u = (x - x0) / (x1 - x0)
        yd = int(round(HORIZON - 3 - arch * math.sin(math.pi * u)))
        # shade the space under the arch
        for y in range(yd + 3, HORIZON):
            bg[y, x] = DARKEN[bg[y, x]]
        # deck
        bg[yd, x] = C["br3"] if rng.random() < 0.45 else C["br2"]
        bg[yd + 1, x] = C["br1"]
        bg[yd + 2, x] = C["br0"]
        # railings
        bg[yd - 6, x] = C["br3"] if rng.random() < 0.3 else C["br2"]
        bg[yd - 3, x] = C["br1"]
        # posts
        if (x - x0 - 3) % 9 == 0:
            for y in range(yd - 6, yd):
                bg[y, x] = C["br2"]
                if x + 1 <= x1:
                    bg[y, x + 1] = C["br0"]


# --------------------------------------------------------------------------- water + lilies

@dataclass
class Pad:
    cx: int
    cy: int
    rx: float
    ry: float
    phase: float
    notch: float
    flower: str | None  # None, "pink", "white"
    seed: int


def make_water_texture():
    tex = np.full((H, W), EMPTY, np.int16)
    for _ in range(560):
        y = int(rng.integers(HORIZON, H))
        d = (y - HORIZON) / (H - HORIZON)
        if d < 0.33:
            pool = ["w0", "w1", "v0", "t0", "b0"]
        elif d < 0.66:
            pool = ["w1", "w2", "v0", "v1", "t1", "w3"]
        else:
            pool = ["w2", "w3", "w4", "v1", "w5", "t2"]
        x, length = int(rng.integers(0, W)), int(rng.integers(3, 10))
        tex[y, [(x + k) % W for k in range(length)]] = C[rng.choice(pool)]
    return tex


def make_glints():
    glints = []
    for _ in range(46):
        glints.append(dict(
            x=int(rng.integers(0, W)), y=int(rng.integers(HORIZON + 1, H)),
            length=int(rng.integers(2, 5)), phase=rng.uniform(0, 2 * math.pi),
            freq=int(rng.choice([1, 2])), color=C[rng.choice(["w6", "w5", "sky3", "fl"])],
        ))
    return glints


def make_pads():
    pads = []
    for _ in range(10):
        ccx = rng.uniform(8, W - 8)
        ccy = HORIZON + 4 + (H - HORIZON - 6) * rng.random() ** 0.7
        for _ in range(int(rng.integers(2, 7))):
            cy = int(np.clip(ccy + rng.normal(0, 2.5), HORIZON + 3, H - 2))
            cx = int(np.clip(ccx + rng.normal(0, 11), 2, W - 3))
            if cx > W - 48 and cy > H - 10:  # keep the signature corner clear
                continue
            rx = 1.6 + (cy - HORIZON) * 0.23 * rng.uniform(0.8, 1.15)
            flower = None
            if rx >= 3 and rng.random() < 0.45:
                flower = "white" if rng.random() < 0.3 else "pink"
            pads.append(Pad(cx, cy, rx, rx * 0.38, rng.uniform(0, 2 * math.pi),
                            rng.uniform(0, 2 * math.pi), flower, int(rng.integers(1 << 30))))
    pads.sort(key=lambda p: p.cy)
    return pads


FLOWER_SPRITES = {
    "small": ["p"],
    "medium": [".l.",
               "pyp"],
    "large": ["l.w.l",
              "plwlp",
              ".ryr."],
}
FLOWER_COLORS = {
    "pink": {"w": "fl", "l": "fp", "p": "fr", "r": "fr", "y": "fy"},
    "white": {"w": "fw", "l": "fw", "p": "fl", "r": "fp", "y": "fy"},
}


def draw_pad(frame, pad, ox):
    cx, cy = pad.cx + ox, pad.cy
    rx, ry = pad.rx, max(pad.ry, 0.9)
    notch = rx >= 4
    pix_rng = np.random.default_rng(pad.seed)
    speckle = pix_rng.random((int(2 * ry) + 4, int(2 * rx) + 4))
    for yy in range(int(cy - ry) - 1, int(cy + ry) + 3):
        for xx in range(int(cx - rx) - 1, int(cx + rx) + 2):
            if not (0 <= xx < W and HORIZON <= yy < H):
                continue
            nx, ny = (xx - cx) / rx, (yy - cy) / ry
            r2 = nx * nx + ny * ny
            if r2 > 1:
                # soft shadow just under the pad
                ny_s = (yy - cy - 1) / ry
                if nx * nx + ny_s * ny_s <= 1:
                    frame[yy, xx] = C["w0"]
                continue
            if notch and r2 > 0.04:
                ang = math.atan2(ny, nx)
                if abs((ang - pad.notch + math.pi) % (2 * math.pi) - math.pi) < 0.32:
                    continue
            s = -nx * 0.5 - ny * 0.85 + (speckle[yy - int(cy - ry) + 1, xx - int(cx - rx) + 1] - 0.5) * 0.6
            if r2 > 0.7 and ny > 0.2:
                color = "p0"
            elif s > 0.55:
                color = "p3"
            elif s > 0.0:
                color = "p2"
            else:
                color = "p1"
            frame[yy, xx] = C[color]

    if pad.flower:
        size = "large" if rx >= 7 else "medium" if rx >= 4.5 else "small"
        sprite, cmap = FLOWER_SPRITES[size], FLOWER_COLORS[pad.flower]
        fx = cx + int(round(rx * 0.15)) - len(sprite[0]) // 2
        fy = cy - int(round(ry * 0.3)) - len(sprite) + 1
        for j, row in enumerate(sprite):
            for i, ch in enumerate(row):
                if ch != "." and 0 <= fx + i < W and 0 <= fy + j < H:
                    frame[fy + j, fx + i] = C[cmap[ch]]


# --------------------------------------------------------------------------- foreground

def make_willows():
    canopy = np.full((H, W), EMPTY, np.int16)
    strands = []
    for side in (0, 1):
        for _ in range(170):
            x = rng.uniform(0, 40)
            y = rng.uniform(-2, 7)
            if (x / 38) ** 2 + ((y + 2) / 9) ** 2 > 1:
                continue
            xx = x if side == 0 else W - 1 - x
            dab(canopy, xx, y, C[rng.choice(["f1", "f2", "f3", "f4", "f5"])],
                int(rng.integers(2, 4)), rng.uniform(0, math.pi))
        for _ in range(26):
            x0 = rng.uniform(0, 36)
            length = int(14 + 34 * (1 - x0 / 36) * rng.uniform(0.55, 1.0))
            strands.append(dict(
                x0=x0 if side == 0 else W - 1 - x0, length=length,
                phase=x0 * 0.09 + rng.normal(0, 0.25), lean=(1 if side == 0 else -1) * rng.uniform(0, 0.05),
                seed=int(rng.integers(1 << 30)),
            ))
    return canopy, strands


def draw_strand(frame, s, f):
    t = 2 * math.pi * f / FRAMES
    pr = np.random.default_rng(s["seed"])
    shades = pr.choice([C["f3"], C["f4"], C["f5"], C["f6"], C["f2"]], size=s["length"] + 1)
    for y in range(s["length"] + 1):
        k = y / s["length"]
        sway = 2.6 * k ** 1.5 * math.sin(t + s["phase"] - y * 0.06)
        x = int(round(s["x0"] + s["lean"] * y + sway))
        color = C["f6"] if y >= s["length"] - 1 else shades[y]
        if 0 <= x < W and y < H:
            frame[y, x] = color
            if y % 3 == 0:
                side = 1 if (y // 3) % 2 else -1
                if 0 <= x + side < W:
                    frame[y, x + side] = shades[y]


def draw_dragonfly(frame, f):
    t = 2 * math.pi * f / FRAMES
    x = int(round(150 + 24 * math.sin(t)))
    y = int(round(49 + 4 * math.sin(2 * t)))
    facing = 1 if math.cos(t) > 0 else -1
    body = ["f0", "b2", "w2", "b2", "w2"]
    for k, name in enumerate(body):
        xx = x - facing * k
        if 0 <= xx < W:
            frame[y, xx] = C[name]
    wing = C["w6"] if f % 2 == 0 else C["w5"]
    wy = y - 2 if f % 2 == 0 else y - 1
    for k in (1, 2):
        xx = x - facing * k
        frame[wy, xx] = wing
        frame[wy + (1 if f % 2 == 0 else 0), xx - facing] = wing


GLYPHS = {
    "Y": ["1.1", "1.1", ".1.", ".1.", ".1."],
    "a": ["...", ".11", "1.1", "1.1", ".11"],
    "n": ["...", "11.", "1.1", "1.1", "1.1"],
    "i": ["1", ".", "1", "1", "1"],
    "c": ["...", ".11", "1..", "1..", ".11"],
    "'": ["1", "1", ".", ".", "."],
    "2": ["11.", "..1", ".1.", "1..", "111"],
    "6": [".11", "1..", "111", "1.1", "111"],
    " ": ["..", "..", "..", "..", ".."],
}


def draw_signature(frame, text):
    width = sum(len(GLYPHS[ch][0]) + 1 for ch in text) - 1
    x, y = W - width - 3, H - 7
    for ch in text:
        g = GLYPHS[ch]
        for j, row in enumerate(g):
            for i, px in enumerate(row):
                if px == "1":
                    frame[y + j, x + i] = C["sig"]
        x += len(g[0]) + 1


# --------------------------------------------------------------------------- frames

def render():
    bg = paint_background()
    paint_bridge(bg)
    tex = make_water_texture()
    glints = make_glints()
    pads = make_pads()
    canopy, strands = make_willows()

    frames = []
    for f in range(FRAMES):
        t = 2 * math.pi * f / FRAMES
        frame = bg.copy()

        # Reflection: mirror the scene above the horizon, wobble each row, tint blue.
        for y in range(HORIZON, H):
            d = (y - HORIZON) / (H - HORIZON)
            src = 2 * HORIZON - 1 - y
            dx = int(round((0.6 + 1.6 * d) * math.sin(2 * t + y * 0.8)))
            cols = np.clip(np.arange(W) + dx, 0, W - 1)
            frame[y] = WATER_TINT[bg[src, cols]]
            # painterly horizontal strokes drifting back and forth
            shift = int(round(2 * math.sin(t + y * 0.45)))
            row = np.roll(tex[y], shift)
            hit = row != EMPTY
            frame[y, hit] = row[hit]

        for g in glints:
            s = math.sin(g["freq"] * t + g["phase"])
            if s > 0.45:
                x = g["x"] + int(round(1.5 * math.sin(t + g["phase"])))
                for k in range(g["length"]):
                    if 0 <= x + k < W:
                        frame[g["y"], x + k] = g["color"]

        for p in pads:
            draw_pad(frame, p, int(round(0.9 * math.sin(t + p.phase))))

        draw_dragonfly(frame, f)

        hit = canopy != EMPTY
        frame[hit] = canopy[hit]
        for s in strands:
            draw_strand(frame, s, f)

        draw_signature(frame, SIGNATURE)
        frames.append(frame)
    return frames


def save(frames):
    OUT.mkdir(exist_ok=True)
    flat = RGB.astype(np.uint8).flatten().tolist()
    flat += [0] * (768 - len(flat))

    def to_image(fr):
        big = np.repeat(np.repeat(fr, SCALE, axis=0), SCALE, axis=1)
        img = Image.fromarray(big, mode="P")
        img.putpalette(flat)
        return img

    imgs = [to_image(fr) for fr in frames]
    gif = OUT / "monet-pond.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=FRAME_MS,
                 loop=0, optimize=False, disposal=1)
    imgs[0].convert("RGB").save(OUT / "preview.png")
    print(f"{gif}  {gif.stat().st_size / 1e6:.2f} MB  ({len(frames)} frames, {W * SCALE}x{H * SCALE})")


if __name__ == "__main__":
    save(render())
