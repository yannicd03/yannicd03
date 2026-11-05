# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=10", "numpy>=1.26"]
# ///
"""Monet-inspired pixel-art banner: a water-lily pond under a Japanese footbridge.

Everything is drawn on a pixel grid with a fixed palette, animated with periodic
functions of the frame index (so the GIF loops seamlessly), then upscaled with
nearest-neighbour so the pixels stay crisp on GitHub. If gifsicle is installed,
the GIF is optimised so that pixels which do not change are not stored again.

    uv run generate.py                       # assets/monet-pond.gif + assets/preview.png
    uv run generate.py --width 240 --height 80 --scale 4
"""

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# The scene is designed on a 240 x 80 reference grid; every coordinate is
# expressed in reference units and scaled to the real canvas via U / V.
REF_W, REF_H = 240, 80

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=384)
ap.add_argument("--height", type=int, default=150)
ap.add_argument("--scale", type=int, default=4)
ap.add_argument("--frames", type=int, default=90)
ap.add_argument("--frame-ms", type=int, default=120)
ap.add_argument("--seed", type=int, default=1899)  # the year Monet started the footbridge series
args = ap.parse_args()

W, H, SCALE = args.width, args.height, args.scale
FRAMES, FRAME_MS = args.frames, args.frame_ms
U, V = W / REF_W, H / REF_H
AREA = U * V
HORIZON = round(H * 0.5)
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
    # rowboat + fisherman
    "wd0": "#3b2a26", "wd1": "#5e4034", "wd2": "#8a6046",
    "hat": "#ecd38f", "hat2": "#c7a65e", "skin": "#e3ab86", "skin2": "#b97f62",
    "shirt": "#f1eee6", "shirt2": "#c4c7d8", "pants": "#3e4f73", "float": "#d9483a",
    # frog
    "fg0": "#2c5426", "fg1": "#5b9440", "fg2": "#9fcf5c", "fgb": "#efe5a8",
    "eye": "#16140f", "eyew": "#f3d96a",
    # dragonfly
    "df0": "#1d4f78", "df1": "#3fb6cf", "df2": "#9be3e8",
}
NAMES = list(COLORS)
C = {name: i for i, name in enumerate(NAMES)}
RGB = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in COLORS.values()], float)
EMPTY = -1

rng = np.random.default_rng(args.seed)


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
    step = max(1, round(step))
    pts = rng.random(n // step + 2)
    xs = np.arange(n) / step
    i = xs.astype(int)
    fr = (1 - np.cos((xs - i) * np.pi)) / 2
    return pts[i] * (1 - fr) + pts[i + 1] * fr


def strokes(lo, hi):
    """Random brush-stroke length, scaled to the canvas resolution."""
    return int(rng.integers(max(1, round(lo * U)), max(2, round(hi * U)) + 1))


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
    sky_span = 0.3 * H

    # Sky: soft gradient, dithered with noise, then horizontal strokes.
    sky_ramp = ["sky0", "sky1", "sky2", "sky3"]
    t = Y[:HORIZON] / sky_span + rng.normal(0, 0.09, (HORIZON, W))
    bg[:HORIZON] = ramp_pick(sky_ramp, t)
    for _ in range(int(160 * AREA)):
        x, y = rng.uniform(0, W), rng.uniform(0, sky_span)
        dab(bg, x, y, ramp_pick(sky_ramp, y / sky_span + rng.normal(0, 0.13)), strokes(3, 7), 0)

    # Foliage band with an irregular top edge that climbs at the sides.
    xs = np.arange(W)
    top = (12 + 9 * smooth_noise(W, 18 * U) + 3 * smooth_noise(W, 5 * U) - 4) * V + 0.06 * H
    side = 42 * U
    top -= 26 * V * (np.clip(1 - xs / side, 0, 1) ** 1.4 + np.clip(1 - (W - 1 - xs) / side, 0, 1) ** 1.4)
    top = np.clip(top, 0, HORIZON - 6 * V).astype(int)
    mask = (Y >= top[None, :]) & (Y < HORIZON)

    fol_ramp = ["f0", "b0", "f1", "t0", "f2", "b1", "f3", "f4", "f5", "f6", "f7"]
    depth = np.clip((Y - top[None, :]) / np.maximum(HORIZON - top[None, :], 1), 0, 1)
    light = (1 - depth) * 0.9 + rng.normal(0, 0.12, (H, W))
    bg[mask] = ramp_pick(fol_ramp, light)[mask]

    pts = np.argwhere(mask)
    for _ in range(int(1700 * AREA)):
        y, x = pts[rng.integers(len(pts))]
        v = (1 - depth[y, x]) * 0.9 + rng.normal(0, 0.15)
        dab(bg, x, y, ramp_pick(fol_ramp, v), strokes(2, 5), rng.uniform(-math.pi, math.pi), mask)

    # Hanging wisteria-like blossoms in the upper foliage.
    upper = np.argwhere(mask & (depth < 0.55))
    for _ in range(int(110 * AREA)):
        y, x = upper[rng.integers(len(upper))]
        color = C[rng.choice(["v1", "v2", "fl", "v0"])]
        dab(bg, x, y, color, strokes(2, 4), math.pi / 2, mask)

    # Dark waterline under the far bank.
    rows = max(2, round(2 * V))
    for i, y in enumerate(range(HORIZON - rows, HORIZON)):
        row = rng.random(W) < 0.5 + 0.35 * (i + 1) / rows
        bg[y, row] = DARKEN[bg[y, row]]
    return bg


def paint_bridge(bg):
    x0, x1 = round(34 * U), round(206 * U)
    arch = 14.5 * V
    base = HORIZON - round(3 * V)
    deck = max(3, round(3 * V))  # deck thickness in rows
    rail_top, rail_mid = round(6.5 * V), round(3.3 * V)
    post_every, post_w = round(9 * U), max(1, round(U))
    for x in range(x0, x1 + 1):
        u = (x - x0) / (x1 - x0)
        yd = int(round(base - arch * math.sin(math.pi * u)))
        # shade the space under the arch
        for y in range(yd + deck, HORIZON):
            bg[y, x] = DARKEN[bg[y, x]]
        # deck: highlight edge, body, shadow
        bg[yd, x] = C["br3"] if rng.random() < 0.5 else C["br2"]
        for k in range(1, deck - 1):
            bg[yd + k, x] = C["br1"] if rng.random() < 0.85 else C["br2"]
        bg[yd + deck - 1, x] = C["br0"]
        # railings
        bg[yd - rail_top, x] = C["br3"] if rng.random() < 0.35 else C["br2"]
        bg[yd - rail_top + 1, x] = C["br1"]
        bg[yd - rail_mid, x] = C["br1"]
        # posts
        if (x - x0 - 3) % post_every < post_w:
            for y in range(yd - rail_top, yd):
                bg[y, x] = C["br2"]
        elif (x - x0 - 3) % post_every == post_w:
            for y in range(yd - rail_top + 2, yd):
                bg[y, x] = C["br0"]


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
    for _ in range(int(560 * AREA)):
        y = int(rng.integers(HORIZON, H))
        d = (y - HORIZON) / (H - HORIZON)
        if d < 0.33:
            pool = ["w0", "w1", "v0", "t0", "b0"]
        elif d < 0.66:
            pool = ["w1", "w2", "v0", "v1", "t1", "w3"]
        else:
            pool = ["w2", "w3", "w4", "v1", "w5", "t2"]
        x, length = int(rng.integers(0, W)), strokes(3, 10)
        tex[y, [(x + k) % W for k in range(length)]] = C[rng.choice(pool)]
    return tex


def make_glints():
    return [dict(
        x=int(rng.integers(0, W)), y=int(rng.integers(HORIZON + 1, H)),
        length=strokes(2, 4), phase=rng.uniform(0, 2 * math.pi),
        color=C[rng.choice(["w6", "w5", "sky3", "fl"])],
    ) for _ in range(int(18 * AREA))]


# Where the characters live (pixel coordinates on the real canvas).
BOAT_X, BOAT_Y = round(0.16 * W), HORIZON + round(10 * V)
FROG_PAD = (round(0.70 * W), H - round(8.5 * V), 10 * U)  # cx, cy, rx
KEEP_CLEAR = [  # (x0, y0, x1, y1) regions where no random lily pads are placed
    (BOAT_X - 6, BOAT_Y - 4, BOAT_X + 52, BOAT_Y + 16),
    (FROG_PAD[0] - 2 * FROG_PAD[2], FROG_PAD[1] - 10, FROG_PAD[0] + 2 * FROG_PAD[2], H),
]


def make_pads():
    pads = [Pad(FROG_PAD[0], FROG_PAD[1], FROG_PAD[2], FROG_PAD[2] * 0.38, 0.0, 0.6, None, 7)]
    for _ in range(round(10 * U)):
        ccx = rng.uniform(8 * U, W - 8 * U)
        ccy = HORIZON + 4 * V + (H - HORIZON - 6 * V) * rng.random() ** 0.7
        for _ in range(int(rng.integers(2, 7))):
            cy = int(np.clip(ccy + rng.normal(0, 2.5 * V), HORIZON + 3 * V, H - 2))
            cx = int(np.clip(ccx + rng.normal(0, 11 * U), 2, W - 3))
            rx = (1.6 + (cy - HORIZON) / V * 0.23 * rng.uniform(0.8, 1.15)) * U
            if any(x0 - rx < cx < x1 + rx and y0 - rx * 0.4 < cy < y1 + rx * 0.4
                   for x0, y0, x1, y1 in KEEP_CLEAR):
                continue
            flower = None
            if rx >= 3 * U and rng.random() < 0.45:
                flower = "white" if rng.random() < 0.3 else "pink"
            pads.append(Pad(cx, cy, rx, rx * 0.38, rng.uniform(0, 2 * math.pi),
                            rng.uniform(0, 2 * math.pi), flower, int(rng.integers(1 << 30))))
    pads.sort(key=lambda p: p.cy)
    return pads


FLOWER_SPRITES = [  # (minimum pad rx in pixels, sprite)
    (11, ["...w...",
          ".l.w.l.",
          "lwlwlwl",
          "plwywlp",
          ".prprp."]),
    (7, ["l.w.l",
         "plwlp",
         ".ryr."]),
    (4.5, [".l.",
           "pyp"]),
    (0, ["p"]),
]
FLOWER_COLORS = {
    "pink": {"w": "fl", "l": "fp", "p": "fr", "r": "fr", "y": "fy"},
    "white": {"w": "fw", "l": "fw", "p": "fl", "r": "fp", "y": "fy"},
}


def draw_pad(frame, pad, ox):
    cx, cy = pad.cx + ox, pad.cy
    rx, ry = pad.rx, max(pad.ry, 0.9)
    notch = rx >= 4
    shadow = max(1, round(V * 0.7))
    pix_rng = np.random.default_rng(pad.seed)
    speckle = pix_rng.random((int(2 * ry) + 6, int(2 * rx) + 6))
    for yy in range(int(cy - ry) - 1, int(cy + ry) + 2 + shadow):
        for xx in range(int(cx - rx) - 1, int(cx + rx) + 2):
            if not (0 <= xx < W and HORIZON <= yy < H):
                continue
            nx, ny = (xx - cx) / rx, (yy - cy) / ry
            r2 = nx * nx + ny * ny
            if r2 > 1:
                # soft shadow just under the pad
                ny_s = (yy - cy - shadow) / ry
                if nx * nx + ny_s * ny_s <= 1:
                    frame[yy, xx] = C["w0"]
                continue
            if notch and r2 > 0.04:
                ang = math.atan2(ny, nx)
                if abs((ang - pad.notch + math.pi) % (2 * math.pi) - math.pi) < 0.3:
                    continue
            s = -nx * 0.5 - ny * 0.85 + (speckle[yy - int(cy - ry) + 1, xx - int(cx - rx) + 1] - 0.5) * 0.6
            if r2 > 0.72 and ny > 0.2:
                color = "p0"
            elif s > 0.55:
                color = "p3"
            elif s > 0.0:
                color = "p2"
            else:
                color = "p1"
            frame[yy, xx] = C[color]

    if pad.flower:
        sprite = next(s for min_rx, s in FLOWER_SPRITES if rx >= min_rx)
        cmap = FLOWER_COLORS[pad.flower]
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
        for _ in range(int(170 * AREA)):
            x = rng.uniform(0, 40 * U)
            y = rng.uniform(-2 * V, 7 * V)
            if (x / (38 * U)) ** 2 + ((y + 2 * V) / (9 * V)) ** 2 > 1:
                continue
            xx = x if side == 0 else W - 1 - x
            dab(canopy, xx, y, C[rng.choice(["f1", "f2", "f3", "f4", "f5"])],
                strokes(2, 4), rng.uniform(0, math.pi))
        for _ in range(round(26 * U)):
            x0 = rng.uniform(0, 36 * U)
            length = int((14 + 34 * (1 - x0 / (36 * U)) * rng.uniform(0.55, 1.0)) * V)
            strands.append(dict(
                x0=x0 if side == 0 else W - 1 - x0, length=length,
                phase=x0 / U * 0.09 + rng.normal(0, 0.25),
                lean=(1 if side == 0 else -1) * rng.uniform(0, 0.05),
                seed=int(rng.integers(1 << 30)),
                sways=rng.random() < 0.25,  # only a few fronds move
            ))
    return canopy, strands


def draw_strand(frame, s, t=None):
    pr = np.random.default_rng(s["seed"])
    shades = pr.choice([C["f3"], C["f4"], C["f5"], C["f6"], C["f2"]], size=s["length"] + 1)
    leaf_every = max(2, round(3 * V / 1.5))
    for y in range(s["length"] + 1):
        k = y / s["length"]
        sway = 0.0 if t is None else 1.4 * U * k ** 1.8 * math.sin(t + s["phase"] - y / V * 0.05)
        x = int(round(s["x0"] + s["lean"] * y + sway))
        color = C["f6"] if y >= s["length"] - 1 else shades[y]
        if 0 <= x < W and y < H:
            frame[y, x] = color
            if y % leaf_every == 0:
                side = 1 if (y // leaf_every) % 2 else -1
                if 0 <= x + side < W:
                    frame[y, x + side] = shades[y]


def blit(frame, sprite, x, y, cmap, flip=False):
    for j, row in enumerate(sprite):
        for i, ch in enumerate(row[::-1] if flip else row):
            if ch != "." and 0 <= x + i < W and 0 <= y + j < H:
                frame[y + j, x + i] = C[cmap[ch]]


def line(frame, x0, y0, x1, y1, color):
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    for k in range(n + 1):
        xx, yy = round(x0 + (x1 - x0) * k / n), round(y0 + (y1 - y0) * k / n)
        if 0 <= xx < W and 0 <= yy < H:
            frame[yy, xx] = color


# ---- fisherman in a rowboat (static, except his float and line)

FISHER = [
    "......hHHh.....................",
    "....HHHHHHHH...................",
    ".....hssS......................",
    "......sS.......................",
    ".....wwwWW.....................",
    "....wwwwWWss...................",
    "....wwwwWW.....................",
    "....pppppppp...................",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.",
    ".BBBBBBBBBBBBBBBBBBBBBBBBBBBB..",
    "...DDDDDDDDDDDDDDDDDDDDDDDDD...",
]
FISHER_COLORS = {"H": "hat", "h": "hat2", "s": "skin", "S": "skin2", "w": "shirt", "W": "shirt2",
                 "p": "pants", "b": "wd2", "B": "wd1", "D": "wd0"}
HANDS = (BOAT_X + 11, BOAT_Y + 5)
ROD_TIP = (HANDS[0] + round(15 * U), HANDS[1] - round(8 * V))
FLOAT = (ROD_TIP[0] + 3, BOAT_Y + 9)
RIPPLE_PERIOD = FRAMES // 2  # two nibbles per loop
RIPPLE_LIFE = int(RIPPLE_PERIOD * 0.9)


def paint_fisherman(scene):
    # dark broken reflection of the hull
    hull_y = BOAT_Y + len(FISHER)
    for k in range(3):
        for x in range(BOAT_X + 1 + 2 * k, BOAT_X + 28 - 2 * k):
            if (x + k) % 2 == 0 and hull_y + k < H:
                scene[hull_y + k, x] = C["w0"]
    blit(scene, FISHER, BOAT_X, BOAT_Y, FISHER_COLORS)
    line(scene, *HANDS, *ROD_TIP, C["wd0"])


def draw_float(frame, f):
    age = f % RIPPLE_PERIOD
    for born in (0, 7):  # each nibble sends out two rings
        a = age - born
        if 0 <= a < RIPPLE_LIFE:
            r = (1.2 + a * 0.12) * U
            fade = a / RIPPLE_LIFE
            for yy in range(int(FLOAT[1] - r * 0.4) - 1, int(FLOAT[1] + r * 0.4) + 2):
                for xx in range(int(FLOAT[0] - r) - 1, int(FLOAT[0] + r) + 2):
                    if not (0 <= xx < W and HORIZON <= yy < H):
                        continue
                    d = math.hypot((xx - FLOAT[0]) / r, (yy - FLOAT[1]) / (r * 0.4))
                    if abs(d - 1) < 0.55 / r * U and (fade < 0.3 or (xx + yy) % 2 == 0):
                        frame[yy, xx] = C["w4"] if fade < 0.5 else C["w3"]
    dip = 1 if age < 4 else 0
    fx, fy = FLOAT[0], FLOAT[1] + dip
    line(frame, *ROD_TIP, fx, fy - 2, C["w5"])
    frame[fy - 1, fx] = C["float"]
    if not dip:
        frame[fy, fx] = C["fw"]


# ---- frog on a lily pad: blinks, croaks, and watches the dragonfly

FROG_OPEN = [
    "..GG...GG..",
    ".GLRG.GLRG.",
    ".GRRGgGRRG.",
    "ggggggggggg",
    "gGdddddddGg",
    "dgyyyyyyygd",
    "ddgyyyyygdd",
    "gggdg.gdggg",
]
FROG_BLINK = ["..GG...GG..", ".GGGG.GGGG.", ".GddGgGddG."] + FROG_OPEN[3:]
FROG_CROAK = FROG_OPEN[:6] + ["dyyyyyyyyyd", "ggyyyyyyygg"]


def draw_frog(frame, f, look_left):
    sprite = FROG_OPEN
    if f % 45 in (20, 21, 22):
        sprite = FROG_BLINK
    if 56 <= f < 62 or 66 <= f < 72:
        sprite = [*sprite[:6], *FROG_CROAK[6:]]
    cmap = {"G": "fg2", "g": "fg1", "d": "fg0", "y": "fgb", "R": "eyew",
            "L": "eye" if look_left else "eyew"}
    if not look_left:  # pupil on the right-hand side of each eye
        sprite = [row.replace("LR", "RL") for row in sprite]
        cmap["L"] = "eye"
    x = FROG_PAD[0] - len(FROG_OPEN[0]) // 2
    y = FROG_PAD[1] - len(FROG_OPEN) + 1
    blit(frame, sprite, x, y, cmap)


# ---- dragonfly: hovers, then darts to the next spot

HOVER = [(0.46 * W, 0.62 * H), (0.60 * W, 0.56 * H), (0.84 * W, 0.66 * H)]
DRAGON_UP = ["......w..w...",
             ".....ww.ww...",
             "aAaAaAaAtTthh",
             "............."]
DRAGON_DOWN = [".............",
               ".....wwwwww..",
               "aAaAaAaAtTthh",
               "......vvvv..."]
DRAGON_COLORS = {"v": "w5", "w": "fw", "a": "df1", "A": "df0", "t": "df1", "T": "df2", "h": "df0"}


def dragonfly_pos(f):
    seg = FRAMES / len(HOVER)
    i, local = int(f // seg), (f % seg) / seg
    (x0, y0), (x1, y1) = HOVER[i], HOVER[(i + 1) % len(HOVER)]
    hover = 0.7
    if local < hover:
        return x0, y0 + math.sin(local / hover * 2 * math.pi) * 1.2, x1 > x0
    e = (local - hover) / (1 - hover)
    e = e * e * (3 - 2 * e)  # smoothstep
    return x0 + (x1 - x0) * e, y0 + (y1 - y0) * e - math.sin(e * math.pi) * 6 * V, x1 > x0


def draw_dragonfly(frame, f):
    x, y, facing_right = dragonfly_pos(f)
    sprite = DRAGON_UP if f % 2 == 0 else DRAGON_DOWN
    blit(frame, sprite, round(x) - 6, round(y) - 2, DRAGON_COLORS, flip=not facing_right)
    return x


# --------------------------------------------------------------------------- frames

def render():
    bg = paint_background()
    paint_bridge(bg)
    tex = make_water_texture()
    glints = make_glints()
    pads = make_pads()
    canopy, strands = make_willows()

    # Everything that never moves is painted once into `scene`.
    scene = bg.copy()
    for y in range(HORIZON, H):
        src = max(0, 2 * HORIZON - 1 - y)
        dx = int(round(0.8 * U * math.sin(y / V * 0.9)))  # broken, painterly reflection
        scene[y] = WATER_TINT[bg[src, np.clip(np.arange(W) + dx, 0, W - 1)]]
        hit = tex[y] != EMPTY
        scene[y, hit] = tex[y, hit]
    for p in pads:
        draw_pad(scene, p, 0)
    paint_fisherman(scene)
    hit = canopy != EMPTY
    scene[hit] = canopy[hit]
    for s in strands:
        if not s["sways"]:
            draw_strand(scene, s)

    frames = []
    for f in range(FRAMES):
        t = 2 * math.pi * f / FRAMES
        frame = scene.copy()
        for g in glints:
            if math.sin(t + g["phase"]) > 0.75:
                for k in range(g["length"]):
                    if 0 <= g["x"] + k < W:
                        frame[g["y"], g["x"] + k] = g["color"]
        draw_float(frame, f)
        dragon_x = draw_dragonfly(frame, f)
        draw_frog(frame, f, look_left=dragon_x < FROG_PAD[0])
        for s in strands:
            if s["sways"]:
                draw_strand(frame, s, t)
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
    raw = gif.stat().st_size
    if shutil.which("gifsicle"):
        subprocess.run(["gifsicle", "-O3", "--batch", str(gif)], check=True)
    imgs[0].convert("RGB").save(OUT / "preview.png")
    print(f"{gif}  {raw / 1e6:.2f} MB raw -> {gif.stat().st_size / 1e6:.2f} MB  "
          f"({len(frames)} frames x {FRAME_MS} ms = {len(frames) * FRAME_MS / 1000:.1f} s, "
          f"{W}x{H} px art -> {W * SCALE}x{H * SCALE})")


if __name__ == "__main__":
    save(render())
