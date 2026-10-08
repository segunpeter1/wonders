"""Rich Friend podcast short, hand-drawn stick animation.

Cuts between a stick-figure podcast studio (two hosts, lip-synced to the
original audio) and stick-figure re-enactments of the stories being read out.
Everything is drawn in code; the audio is the original clip, unchanged.

Usage: python3 render.py <original_audio.wav> <output.mp4>
"""
import importlib.util
import math
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(HERE.parent / "human-lesson-video"))
stick = _load("stick", HERE.parent / "human-lesson-video" / "stick.py")
st = stick.st
Pen, clamp, lerp, ease_io, ease_out, kick = stick.Pen, stick.clamp, stick.lerp, stick.ease_io, stick.ease_out, stick.kick
SS, INK = stick.SS, stick.INK


def back_out(t):
    t = clamp(t, 0, 1)
    return 1 + 2.9 * (t - 1) ** 3 + 1.9 * (t - 1) ** 2

W, H, FPS = 1080, 1920, 30
END = 58.09
FONT_PATH = HERE.parent / "angel-demon-video" / "fonts" / "LuckiestGuy.ttf"

# ------------------------------------------------------------------ cast

HOST_A = dict(skin=(110, 72, 50), shirt=(250, 250, 248), pants=(225, 205, 160), hair="fade", beard=(30, 22, 18))
HOST_B = dict(skin=(196, 150, 112), shirt=(250, 250, 248), pants=(70, 70, 80), hair="fade", beard=(40, 30, 24))
NARRATOR = dict(skin=(150, 100, 66), shirt=(90, 170, 110), pants=(60, 70, 110), hair="short", beard=None)
RICH_GIRL = dict(skin=(165, 110, 75), shirt=(240, 120, 170), pants=(240, 120, 170), hair="puff", beard=None, dress=True)
CRYPTO_GUY = dict(skin=(185, 135, 95), shirt=(80, 120, 200), pants=(50, 50, 60), hair="short", beard=None)
DAD = dict(skin=(180, 130, 92), shirt=(150, 90, 160), pants=(60, 60, 70), hair="bald", beard=(200, 200, 205))

# --------------------------------------------------------------- drawing


def text(pen, x, y, s, size, fill=(255, 255, 255), stroke=INK, sw=6):
    f = ImageFont.truetype(str(FONT_PATH), max(4, int(size * pen.z * SS)))
    px, py = pen.p(x, y)
    tw = f.getlength(s)
    pen.d.text((px - tw / 2, py - size * pen.z * SS / 2), s, font=f, fill=fill,
               stroke_width=max(1, int(sw * pen.z * SS / 2)), stroke_fill=stroke)


def ik(sx, sy, tx, ty, l1, l2):
    dx, dy = tx - sx, ty - sy
    d = clamp(math.hypot(dx, dy), 1e-3, l1 + l2 - 1e-3)
    base = math.atan2(dy, dx)
    a = math.acos(clamp((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1, 1))
    cands = [(sx + l1 * math.cos(base + sg * a), sy + l1 * math.sin(base + sg * a)) for sg in (1, -1)]
    ex, ey = max(cands, key=lambda c: c[1])  # elbows hang down
    hx, hy = sx + math.cos(base) * min(math.hypot(dx, dy), l1 + l2), sy + math.sin(base) * min(math.hypot(dx, dy), l1 + l2)
    return (ex, ey), (hx, hy)


def face(pen, x, y, r, look, facing, eyes="open", mouth=0.0, brows=None, t=0.0):
    pen.ellipse(x, y, r, r, fill=look["skin"], width=8)
    hc = (32, 24, 20)
    if look["hair"] == "fade":
        pen.arc(x, y, r * 0.98, r * 0.98, 195, 345, width=r * 0.3, fill=hc)
    elif look["hair"] == "short":
        for i in range(6):
            a = math.radians(-165 + i * 30)
            pen.ellipse(x + math.cos(a) * r * 0.85, y + math.sin(a) * r * 0.85, r * 0.26, r * 0.26, fill=hc, outline=None)
    elif look["hair"] == "puff":
        pen.ellipse(x - facing * r * 0.2, y - r * 1.15, r * 0.6, r * 0.5, fill=hc, width=6)
        pen.arc(x, y, r * 0.98, r * 0.98, 190, 350, width=r * 0.22, fill=hc)
    if look.get("beard"):
        pts = [(x + r * 1.0 * math.cos(math.radians(a)), y + r * 1.0 * math.sin(math.radians(a))) for a in range(10, 171, 16)]
        pts += [(x - r * 0.55, y + r * 0.25), (x + r * 0.55, y + r * 0.25)][::-1]
        pen.poly(pts, fill=look["beard"], width=5)
    ex = x + facing * r * 0.3
    for dx in (-0.32, 0.32):
        cx, cy = ex + dx * r, y - r * 0.15
        if eyes in ("closed", "happy", "laughcry"):
            hp = eyes != "closed"
            pen.arc(cx, cy + (r * 0.06 if hp else 0), r * 0.15, r * 0.12, 200 if hp else 20,
                    340 if hp else 160, width=5)
        elif eyes == "huge":
            pen.ellipse(cx, cy - r * 0.04, r * 0.22, r * 0.28, fill=(255, 255, 255), width=5)
            pen.ellipse(cx + facing * r * 0.03, cy - r * 0.04, r * 0.06, r * 0.06, fill=INK, outline=None)
        elif eyes == "half":
            pen.ellipse(cx, cy, r * 0.13, r * 0.13, fill=(255, 255, 255), width=4)
            pen.ellipse(cx + facing * r * 0.04, cy + r * 0.03, r * 0.06, r * 0.06, fill=INK, outline=None)
            pen.line([(cx - r * 0.17, cy - r * 0.02), (cx + r * 0.17, cy - r * 0.02)], width=6)
        elif eyes == "up":
            pen.ellipse(cx, cy, r * 0.14, r * 0.16, fill=(255, 255, 255), width=4)
            pen.ellipse(cx + facing * r * 0.03, cy - r * 0.07, r * 0.07, r * 0.07, fill=INK, outline=None)
        else:
            pen.ellipse(cx, cy, r * 0.13, r * 0.15, fill=(255, 255, 255), width=4)
            pen.ellipse(cx + facing * r * 0.05, cy, r * 0.07, r * 0.08, fill=INK, outline=None)
            pen.ellipse(cx + facing * r * 0.02, cy - r * 0.04, r * 0.025, r * 0.025, fill=(255, 255, 255), outline=None)
        if eyes in ("cry", "laughcry"):
            for k in range(2):
                ty = cy + r * 0.25 + ((t * 300 + k * 40) % (r * 1.2))
                pen.ellipse(cx, ty, r * 0.06, r * 0.09, fill=(120, 180, 255), width=2)
    if brows == "up":
        for dx in (-0.32, 0.32):
            pen.arc(ex + dx * r, y - r * 0.52, r * 0.17, r * 0.1, 200, 340, width=5)
    elif brows == "sad":
        pen.line([(ex - r * 0.48, y - r * 0.38), (ex - r * 0.16, y - r * 0.5)], width=5)
        pen.line([(ex + r * 0.16, y - r * 0.5), (ex + r * 0.48, y - r * 0.38)], width=5)
    mx, my = x + facing * r * 0.28, y + r * 0.42
    if mouth == "laugh":
        o = 0.75 + 0.25 * abs(math.sin(t * 22))
        pen.poly([(mx - r * 0.3, my - r * 0.08), (mx + r * 0.3, my - r * 0.08), (mx + r * 0.18, my + r * 0.3 * o), (mx - r * 0.18, my + r * 0.3 * o)],
                 fill=(130, 30, 40), width=5)
        pen.line([(mx - r * 0.26, my - r * 0.03), (mx + r * 0.26, my - r * 0.03)], width=5, fill=(255, 255, 255))
    elif mouth == "flat":
        pen.line([(mx - r * 0.2, my), (mx + r * 0.2, my)], width=6)
    elif mouth == "frown":
        pen.arc(mx, my + r * 0.12, r * 0.22, r * 0.14, 200, 340, width=6)
    elif mouth == "O":
        pen.ellipse(mx, my, r * 0.14, r * 0.2, fill=(130, 30, 40), width=5)
    elif isinstance(mouth, str) or mouth < 0.12:
        pen.arc(mx, my - r * 0.1, r * 0.24, r * 0.15, 20, 160, width=6)
    else:
        h = r * (0.08 + 0.24 * mouth)
        pen.ellipse(mx, my, r * 0.22, h, fill=(130, 30, 40), width=5)
        pen.ellipse(mx, my + h * 0.45, r * 0.13, h * 0.38, fill=(230, 110, 120), outline=None)


# hand targets relative to the neck, in units of s (x is multiplied by facing)
POSES = {
    "rest": ((20, 110), (40, 108)),
    "phone": ((30, 62), (48, 58)),
    "point": ((10, 100), (120, 20)),
    "talk": ((10, 100), (70, 40)),
    "shrug": ((-70, 25), (75, 25)),
    "handshead": ((-30, -95), (35, -95)),
    "facepalm": ((0, -55), (35, -55)),
    "belly": ((5, 75), (35, 75)),
    "pray": ((32, 5), (40, 5)),
    "cry": ((12, -52), (40, -52)),
    "sip": ((10, 100), (42, -30)),
    "leash": ((10, 100), (75, 55)),
    "push": ((70, 55), (80, 50)),
    "walk": ((0, 100), (0, 100)),
    "think": ((10, 100), (38, -28)),
}


def person(pen, x, hip_y, s, look, facing=1, pose="rest", mouth=0.0, eyes="open", brows=None, t=0.0,
           sitting=False, lean=0.0, walk=False, prop=None, kick=0.0):
    r, T, ua, fa = 40 * s, 95 * s, 50 * s, 48 * s
    hc = look["pants"]
    if walk:
        ph = t * 2 * math.pi * 1.8
        hip_y -= abs(math.sin(ph)) * 6 * s
    nx, ny = x + facing * lean * s, hip_y - T
    hx, hy = nx, ny - r * 0.95
    shoulders = [(nx - facing * 13 * s, ny + 12 * s), (nx + facing * 13 * s, ny + 12 * s)]
    targets = POSES[pose]
    if walk:
        sw = math.sin(t * 2 * math.pi * 1.8) * 35 * s
        if pose == "walk":
            targets = ((-sw / s, 100), (sw / s, 100))

    def arm(i):
        sx, sy = shoulders[i]
        tx, ty = targets[i]
        tx, ty = nx + facing * tx * s, ny + ty * s
        (ex, ey), (px, py) = ik(sx, sy, tx, ty, ua, fa)
        pen.line([(sx, sy), (ex, ey), (px, py)], width=9)
        pen.ellipse(px, py, 8 * s, 8 * s, fill=look["skin"], width=4)
        return px, py

    arm(0)
    # legs
    if sitting:
        for k, off in enumerate((-8, 8)):
            kx, ky = x + facing * 52 * s, hip_y + off * s * 0.3 - kick * 28 * s * abs(math.sin(t * 18 + k * 1.6))
            fx, fy = kx + facing * 6 * s, hip_y + 58 * s
            pen.line([(x, hip_y), (kx, ky), (fx, fy)], width=12, fill=hc)
            pen.ellipse(fx + facing * 9 * s, fy, 15 * s, 8 * s, fill=(250, 250, 250), width=5)
    else:
        if walk:
            ph = t * 2 * math.pi * 1.8
            angs = (30 * math.sin(ph), -30 * math.sin(ph))
        else:
            angs = (10, -10)
        for a in angs:
            fx, fy = x + math.sin(math.radians(a)) * 100 * s * facing, hip_y + math.cos(math.radians(a)) * 100 * s
            pen.line([(x, hip_y), (fx, fy)], width=11, fill=INK if not look.get("dress") else INK)
            pen.ellipse(fx + facing * 9 * s, fy, 15 * s, 8 * s, fill=(250, 250, 250), width=5)
    # shirt / dress
    bot = hip_y + (45 * s if look.get("dress") else 8)
    wdt = 44 if look.get("dress") else 32
    pen.poly([(nx - 26 * s, ny + 6), (nx + 26 * s, ny + 6), (x + wdt * s, bot), (x - wdt * s, bot)], fill=look["shirt"], width=7)
    if look.get("dress") is None and not sitting:
        pass
    face(pen, hx, hy, r, look, facing, eyes=eyes, mouth=mouth, brows=brows, t=t)
    px, py = arm(1)
    if prop == "phone":
        pen.poly([(px - 14 * s, py - 30 * s), (px + 14 * s, py - 30 * s), (px + 14 * s, py + 6 * s), (px - 14 * s, py + 6 * s)],
                 fill=(40, 40, 50), width=4)
        pen.rect(px - 10 * s, py - 26 * s, px + 10 * s, py + 2 * s, fill=(140, 200, 255), outline=None)
    elif prop == "cup":
        pen.rect(px - 13 * s, py - 18 * s, px + 13 * s, py + 10 * s, fill=(250, 250, 250), width=5)
        for k in range(2):
            sx = px - 6 * s + k * 12 * s
            pen.line([(sx, py - 26 * s), (sx + 5 * s * math.sin(t * 6 + k), py - 40 * s), (sx, py - 54 * s)], width=3, fill=(160, 160, 160))
    return (hx, hy, r), (px, py)


def dog(pen, x, fy, s, facing, t):
    by = fy - 40 * s
    for k, dx in enumerate((-26, -12, 14, 28)):
        sw = 8 * math.sin(t * 11 + k * 1.7) * s
        pen.line([(x + dx * s * facing, by + 8 * s), (x + dx * s * facing + sw, fy)], width=7)
    pen.ellipse(x, by, 42 * s, 22 * s, fill=(225, 180, 120), width=6)
    pen.poly([(x - 30 * s, by - 18 * s), (x + 30 * s, by - 18 * s), (x + 34 * s, by + 14 * s), (x - 34 * s, by + 14 * s)],
             fill=(240, 120, 170), width=5)  # designer sweater
    ta = math.radians(-60 + 25 * math.sin(t * 16))
    tx = x - facing * 40 * s
    pen.line([(tx, by - 4 * s), (tx - facing * math.cos(ta) * 30 * s, by + math.sin(ta) * 30 * s)], width=6)
    hx, hy = x + facing * 48 * s, by - 26 * s
    pen.ellipse(hx, hy, 22 * s, 20 * s, fill=(225, 180, 120), width=6)
    pen.poly([(hx - facing * 14 * s, hy - 14 * s), (hx - facing * 4 * s, hy - 34 * s), (hx + facing * 4 * s, hy - 12 * s)],
             fill=(180, 130, 80), width=5)
    pen.ellipse(hx + facing * 22 * s, hy + 4 * s, 6 * s, 5 * s, fill=INK, outline=None)
    # sunglasses + gold chain
    pen.rect(hx - 6 * s, hy - 8 * s, hx + facing * 20 * s, hy + 1 * s, fill=(20, 20, 30), outline=None)
    pen.arc(hx, hy + 10 * s, 18 * s, 10 * s, 20, 160, width=4, fill=(230, 190, 50))


def money_bag(pen, x, y, s):
    pen.ellipse(x, y, 34 * s, 30 * s, fill=(200, 170, 110), width=6)
    pen.poly([(x - 12 * s, y - 28 * s), (x + 12 * s, y - 28 * s), (x + 18 * s, y - 42 * s), (x - 18 * s, y - 42 * s)],
             fill=(200, 170, 110), width=5)
    text(pen, x, y + 2 * s, "$", 34 * s, fill=(40, 120, 60), sw=3)


def coin(pen, x, y, s):
    pen.ellipse(x, y, 20 * s, 20 * s, fill=(245, 200, 60), width=5)
    text(pen, x, y + 1, "$", 22 * s, fill=(190, 140, 20), stroke=(190, 140, 20), sw=1)


def ha(pen, x, y, t, k=0):
    if (t * 3 + k) % 1 < 0.6:
        text(pen, x + 6 * math.sin(t * 20 + k), y, "HA", 54, fill=(255, 210, 60), sw=7)


def bubble(pen, x, y, rx, ry, tail_x, tail_y):
    for k in range(3):
        f = (k + 1) / 4
        pen.ellipse(lerp(tail_x, x, f * 0.6), lerp(tail_y, y + ry, f * 0.6), 10 + 8 * k, 10 + 8 * k, fill=(255, 255, 255), width=5)
    for a in range(0, 360, 40):
        pen.ellipse(x + rx * 0.82 * math.cos(math.radians(a)), y + ry * 0.82 * math.sin(math.radians(a)), rx * 0.32, ry * 0.32,
                    fill=(255, 255, 255), width=6)
    pen.ellipse(x, y, rx * 0.9, ry * 0.9, fill=(255, 255, 255), outline=None)


# ----------------------------------------------------------------- scenes

STUDIO = dict(A=(300, 1190), B=(780, 1190), s=1.7)


def studio_bg(pen, t):
    pen.rect(-800, -1000, 1900, 1500, fill=(238, 230, 214), outline=None)
    for x in range(-700, 1900, 55):  # curtain folds
        pen.line([(x, 300), (x + 8 * math.sin(x), 1060)], width=4, fill=(222, 210, 190))
    pen.rect(-800, 1500, 1900, 3000, fill=(196, 170, 140), outline=None)
    # neon sign
    pen.poly([(390, 560), (690, 560), (690, 660), (390, 660)], fill=(60, 40, 70), width=6)
    glow = (255, 120, 190) if (t * 2) % 1 < 0.92 else (200, 90, 150)
    text(pen, 540, 610, "THE POD", 58, fill=glow, stroke=(120, 40, 90), sw=4)
    # plant
    pen.rect(-20, 1300, 90, 1500, fill=(200, 120, 80), width=7)
    for k in range(6):
        a = math.radians(-150 + k * 25)
        pen.line([(35, 1300), (35 + math.cos(a) * 170, 1300 + math.sin(a) * 210)], width=10, fill=(70, 150, 80))
    # couch
    pen.poly([(80, 1020), (1000, 1020), (1000, 1210), (80, 1210)], fill=(150, 95, 60), width=8)
    pen.poly([(40, 1200), (1040, 1200), (1040, 1420), (40, 1420)], fill=(170, 110, 72), width=8)
    pen.hatch(60, 1360, 1020, 1410, step=30, fill=(120, 75, 50))


def mic(pen, x, y, facing):
    top = (x - facing * 330, 520)
    pen.line([top, (x - facing * 120, y - 230), (x + facing * 40, y - 20)], width=8, fill=(40, 40, 45))
    pen.poly([(x - 22, y - 34), (x + 22, y - 34), (x + 22, y + 34), (x - 22, y + 34)], fill=(35, 35, 40), width=5)
    for k in range(3):
        pen.line([(x - 16, y - 18 + k * 16), (x + 16, y - 18 + k * 16)], width=3, fill=(110, 110, 120))


def studio(pen, t, a, b):
    studio_bg(pen, t)
    s = STUDIO["s"]
    for who, look, facing, st_ in (("A", HOST_A, 1, a), ("B", HOST_B, -1, b)):
        x, hy = STUDIO[who]
        bob = st_.get("bob", 0) * math.sin(t * st_.get("bobf", 18))
        (hx, hy2, r), _ = person(pen, x, hy + bob, s, look, facing=facing, pose=st_.get("pose", "rest"), mouth=st_.get("mouth", 0.0),
                                 eyes=st_.get("eyes", "open"), brows=st_.get("brows"), t=t, sitting=True, lean=st_.get("lean", 0),
                                 prop=st_.get("prop"))
        mic(pen, hx + facing * r * 1.75, hy2 + r * 1.0, facing)


def market(pen, t, lt):
    pen.rect(-800, -1000, 1900, 3000, fill=(170, 215, 245), outline=None)
    pen.ellipse(880, 380, 70, 70, fill=(255, 220, 90), width=6)
    pen.rect(-800, 1350, 1900, 3000, fill=(215, 195, 160), outline=None)
    for i, (sx, c) in enumerate(((-120, (220, 70, 70)), (330, (60, 160, 90)), (780, (240, 160, 50)))):
        pen.rect(sx, 900, sx + 360, 1350, fill=(200, 150, 100), width=7)
        for k in range(6):
            pen.poly([(sx - 20 + k * 67, 820), (sx + 47 + k * 67, 820), (sx + 47 + k * 67, 920), (sx - 20 + k * 67, 920)],
                     fill=c if k % 2 == 0 else (250, 250, 245), width=5)
        for k in range(7):
            pen.ellipse(sx + 40 + k * 45, 1010 - (k % 2) * 25, 22, 22, fill=[(240, 90, 60), (250, 200, 60), (120, 190, 80)][k % 3], width=4)


def cart(pen, x, fy, s, items):
    pen.poly([(x - 110 * s, fy - 190 * s), (x + 110 * s, fy - 190 * s), (x + 85 * s, fy - 70 * s), (x - 85 * s, fy - 70 * s)],
             fill=(220, 225, 235), width=7)
    for k in range(5):
        pen.line([(x - 90 * s + k * 45 * s, fy - 185 * s), (x - 70 * s + k * 35 * s, fy - 75 * s)], width=3, fill=(150, 160, 175))
    pen.line([(x + 110 * s, fy - 190 * s), (x + 150 * s, fy - 230 * s)], width=8)
    for wx in (-60, 60):
        pen.ellipse(x + wx * s, fy - 25 * s, 20 * s, 20 * s, fill=(60, 60, 70), width=5)
    pen.line([(x - 85 * s, fy - 70 * s), (x - 60 * s, fy - 45 * s)], width=6)
    pen.line([(x + 85 * s, fy - 70 * s), (x + 60 * s, fy - 45 * s)], width=6)
    # dog shopping piled up, popping in one by one
    stuff = [("bag", -50, -240), ("bone", 40, -230), ("bowl", -10, -270), ("ball", 70, -280), ("bag2", 0, -320),
             ("bone", -70, -300), ("bed", 20, -370)]
    for k, (kind, dx, dy) in enumerate(stuff[:items]):
        ix, iy = x + dx * s, fy + dy * s
        if kind in ("bag", "bag2"):
            pen.rect(ix - 32 * s, iy - 45 * s, ix + 32 * s, iy + 45 * s, fill=(240, 200, 90) if kind == "bag" else (150, 200, 240), width=6)
            text(pen, ix, iy - 5 * s, "DOG", 22 * s, fill=(255, 255, 255), sw=4)
        elif kind == "bone":
            pen.line([(ix - 30 * s, iy), (ix + 30 * s, iy)], width=14 * s, fill=(250, 245, 230))
            for ex in (-30, 30):
                for ey in (-8, 8):
                    pen.ellipse(ix + ex * s, iy + ey * s, 11 * s, 11 * s, fill=(250, 245, 230), width=4)
        elif kind == "bowl":
            pen.poly([(ix - 40 * s, iy - 15 * s), (ix + 40 * s, iy - 15 * s), (ix + 28 * s, iy + 18 * s), (ix - 28 * s, iy + 18 * s)],
                     fill=(220, 70, 80), width=6)
        elif kind == "ball":
            pen.ellipse(ix, iy, 22 * s, 22 * s, fill=(170, 220, 60), width=5)
        else:
            pen.ellipse(ix, iy, 70 * s, 30 * s, fill=(200, 140, 230), width=6)
            text(pen, ix, iy, "VIP", 26 * s, fill=(255, 230, 120), sw=4)


def room_bg(pen, wall=(232, 222, 240), floor=(205, 180, 150)):
    pen.rect(-800, -1000, 1900, 1400, fill=wall, outline=None)
    for x in range(-700, 1900, 80):
        pen.line([(x, 300), (x, 1395)], width=3, fill=tuple(max(0, c - 14) for c in wall))
    pen.rect(-800, 1400, 1900, 3000, fill=floor, outline=None)
    pen.line([(-800, 1400), (1900, 1400)], width=7)


def crash_screen(pen, t, x, y, wdt, hgt):
    pen.rect(x - wdt / 2, y - hgt / 2, x + wdt / 2, y + hgt / 2, fill=(25, 30, 45), width=8)
    k = clamp((t - 12.3) / 1.2, 0, 1)
    pts = []
    for i in range(int(4 + 20 * k) + 1):
        f = i / 24
        px = x - wdt * 0.42 + f * wdt * 0.84
        py = y - hgt * 0.25 + (f ** 2) * hgt * 0.6 + 14 * math.sin(i * 1.7)
        pts.append((px, py))
    if len(pts) >= 2:
        pen.line(pts, width=7, fill=(240, 60, 60))
    text(pen, x, y - hgt * 0.38, "CRYPTO", 34, fill=(200, 200, 220), sw=3)


def living(pen):
    room_bg(pen, wall=(246, 232, 205))
    pen.rect(560, 640, 760, 790, fill=(250, 246, 235), width=7)  # family picture
    pen.ellipse(620, 715, 22, 22, fill=(180, 130, 92), width=4)
    pen.ellipse(695, 715, 22, 22, fill=(185, 135, 95), width=4)
    pen.poly([(60, 1060), (1020, 1060), (1020, 1230), (60, 1230)], fill=(110, 140, 190), width=8)
    pen.poly([(20, 1220), (1060, 1220), (1060, 1420), (20, 1420)], fill=(130, 160, 210), width=8)


def park(pen, t):
    pen.rect(-800, -1000, 1900, 3000, fill=(185, 225, 250), outline=None)
    pen.rect(-800, 1300, 1900, 3000, fill=(130, 200, 110), outline=None)
    for tx in (60, 980):
        pen.rect(tx - 25, 850, tx + 25, 1320, fill=(140, 95, 60), width=7)
        for k in range(4):
            pen.ellipse(tx + (k - 1.5) * 70, 780 - (k % 2) * 60, 120, 110, fill=(80, 160, 90), width=6)
    pen.poly([(330, 640), (750, 640), (750, 760), (330, 760)], fill=(250, 245, 230), width=6)
    pen.line([(540, 760), (540, 900)], width=8)
    text(pen, 540, 700, "UNI", 54, fill=(230, 90, 90), sw=5)
    pen.rect(200, 1190, 880, 1225, fill=(170, 110, 70), width=7)
    pen.line([(230, 1225), (230, 1330)], width=9)
    pen.line([(850, 1225), (850, 1330)], width=9)


def village(pen, cx, cy, sc):
    for k, hx in enumerate((-120, 60)):
        x = cx + hx * sc
        pen.rect(x - 60 * sc, cy - 10 * sc, x + 60 * sc, cy + 80 * sc, fill=(200, 150, 100), width=5)
        pen.poly([(x - 85 * sc, cy - 5 * sc), (x, cy - 80 * sc), (x + 85 * sc, cy - 5 * sc)], fill=(220, 190, 90), width=5)
        pen.rect(x - 15 * sc, cy + 30 * sc, x + 15 * sc, cy + 80 * sc, fill=(110, 70, 40), width=4)
    px = cx + 180 * sc
    pen.line([(px, cy + 80 * sc), (px - 10 * sc, cy - 90 * sc)], width=9, fill=(140, 95, 60))
    for a in (-160, -120, -60, -20):
        pen.line([(px - 10 * sc, cy - 90 * sc), (px - 10 * sc + math.cos(math.radians(a)) * 70 * sc, cy - 90 * sc + math.sin(math.radians(a)) * 40 * sc + 30 * sc)],
                 width=8, fill=(70, 150, 80))
    ch = cx - 30 * sc
    pen.ellipse(ch, cy + 100 * sc, 18 * sc, 14 * sc, fill=(255, 255, 255), width=4)
    pen.ellipse(ch + 16 * sc, cy + 86 * sc, 9 * sc, 9 * sc, fill=(255, 255, 255), width=4)
    pen.poly([(ch + 24 * sc, cy + 86 * sc), (ch + 34 * sc, cy + 90 * sc), (ch + 24 * sc, cy + 93 * sc)], fill=(250, 170, 40), width=2)


def rome(pen, cx, cy, sc, t):
    pen.ellipse(cx, cy + 10 * sc, 190 * sc, 95 * sc, fill=(230, 200, 150), width=6)
    pen.ellipse(cx, cy - 20 * sc, 150 * sc, 55 * sc, fill=(250, 245, 230), width=5)
    for row, (ry, n) in enumerate(((cy + 15 * sc, 7), (cy + 55 * sc, 7))):
        for k in range(n):
            ax = cx - 150 * sc + k * 50 * sc
            pen.arc(ax, ry, 15 * sc, 18 * sc, 180, 360, width=5)
            pen.line([(ax - 15 * sc, ry), (ax - 15 * sc, ry + 22 * sc)], width=4)
            pen.line([(ax + 15 * sc, ry), (ax + 15 * sc, ry + 22 * sc)], width=4)
    px = cx + 130 * sc + 40 * sc * math.sin(t * 2)
    pen.poly([(px - 30 * sc, cy - 120 * sc), (px + 30 * sc, cy - 112 * sc), (px - 30 * sc, cy - 104 * sc)], fill=(250, 250, 255), width=4)
    for k, c in enumerate(((60, 150, 80), (250, 250, 250), (220, 60, 60))):
        pen.rect(cx - 210 * sc + k * 22 * sc, cy - 140 * sc, cx - 188 * sc + k * 22 * sc, cy - 100 * sc, fill=c, width=3)


# ----------------------------------------------------------------- audio


def load_audio(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return a.reshape(-1, w.getnchannels()).mean(1), sr


def talk_track(a, sr):
    hop = sr // FPS
    n = int(END * FPS) + 2
    rms = np.array([np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2)) if i * hop < len(a) else 0 for i in range(n)])
    lo, hi = np.percentile(rms, 30), np.percentile(rms, 92)
    o = np.clip((rms - lo) / (hi - lo + 1e-6), 0, 1)
    out = np.where(o > 0.45, 1.0, np.where(o > 0.12, 0.5, 0.0))
    for i in range(3, n):  # flap on long sounds
        if out[i] == 1.0 and out[i - 1] == 1.0 and out[i - 2] == 1.0 and i % 4 == 0:
            out[i] = 0.5
    return out


# -------------------------------------------------------------- captions

CAPTIONS = [
    [("HOW", 0.00), ("DID", 0.40), ("YOU", 0.60), ("FIND", 0.80), ("OUT", 1.10)],
    [("THAT", 1.56), ("YOUR", 1.86), ("FRIEND", 2.14), ("IS", 2.50)],
    [("FROM", 2.76), ("A", 2.96), ("RICH", 3.14), ("FAMILY?", 3.36)],
    [("I", 3.96), ("ESCORTED", 4.02), ("HER", 4.44), ("TO", 4.72), ("THE", 4.88), ("MARKET", 5.08)],
    [("AND", 5.48), ("EVERYTHING", 6.22), ("WE", 6.60), ("BOUGHT", 6.94)],
    [("WAS", 7.14), ("JUST", 7.40), ("FOR", 7.66), ("HER", 7.92), ("DOG.", 8.18)],
    [("JESUS", 8.92), ("CHRIST.", 9.36)],
    [("THAT'S", 9.92), ("MONEY.", 10.06)],
    [("THAT'S", 10.56), ("JUICY", 10.68), ("MONEY.", 10.92)],
    [("THAT'S", 11.36), ("JUICY", 11.46), ("MONEY.", 11.58)],
    [("MY", 12.00), ("GUY", 12.14), ("LOST", 12.36), ("30K", 12.66), ("IN", 13.16), ("CRYPTO", 13.40)],
    [("AND", 13.76), ("HE", 14.68), ("WAS", 14.82), ("STILL", 15.00), ("NORMAL,", 15.28)],
    [("LIKE", 15.88), ("COMPLETELY", 16.18), ("NORMAL.", 16.66)],
    [("HE", 17.14), ("CASUALLY", 17.44), ("DISCUSSED", 17.80), ("IT", 18.20)],
    [("WITH", 18.34), ("HIS", 18.48), ("DAD", 18.70)],
    [("AND", 18.98), ("THEY", 19.54), ("JUST", 19.72), ("LAUGHED", 19.94), ("IT", 20.14), ("OFF.", 20.36)],
    [("I", 20.68), ("LOST", 20.86), ("1K", 21.10)],
    [("IN", 21.76), ("THAT", 21.92), ("SAME", 22.16), ("PERIOD.", 22.38)],
    [("THE", 22.88), ("REASON", 23.10), ("I", 23.36), ("SURVIVED", 23.60), ("TILL", 23.94), ("TODAY", 24.16)],
    [("IS", 24.48), ("BY", 24.70), ("THE", 24.84), ("SPECIAL", 25.06), ("GRACE", 25.28), ("OF", 25.48), ("GOD.", 25.70)],
    [("BRO,", 30.70)],
    [("30", 32.58), ("BAGS", 33.04)],
    [("AND", 34.14), ("HIM", 34.56), ("AND", 34.72), ("HIS", 34.84), ("DAD", 35.00)],
    [("JUST", 35.38), ("LAUGHED", 35.70), ("IT", 35.96), ("OFF.", 36.30)],
    [("WHAT", 36.96), ("KIND", 37.06), ("OF", 37.18), ("MONEY", 37.28)],
    [("ARE", 37.40), ("WE", 37.48), ("TALKING", 37.70), ("ABOUT?", 37.98)],
    [("BILLIONS.", 38.64)],
    [("I", 39.44), ("NEED", 39.94), ("THAT", 40.70)],
    [("GEN", 41.86), ("MONEY.", 42.12)],
    [("THAT'S", 42.62), ("ENERGY.", 42.92)],
    [("WE", 43.68), ("WERE", 44.24), ("DISCUSSING", 44.50)],
    [("ABOUT", 45.06), ("WHERE", 45.40), ("TO", 45.62), ("SPEND", 45.88)],
    [("THE", 46.06), ("SEMESTER", 46.36), ("BREAK.", 46.72)],
    [("I", 47.22), ("MENTIONED", 47.46), ("MY", 47.68), ("COUSINS", 47.96)],
    [("IN", 48.26), ("THE", 48.44), ("VILLAGE.", 48.62)],
    [("SHE", 49.34), ("MENTIONED", 49.58), ("ROME", 49.92), ("IN", 50.20), ("ITALY.", 50.42)],
]
CAP_FONT = ImageFont.truetype(str(FONT_PATH), 92)
_cap = {}


def caption_img(ci, active):
    key = (ci, active)
    if key not in _cap:
        words = [w for w, _ in CAPTIONS[ci]]
        sp = CAP_FONT.getlength(" ")
        tw = sum(CAP_FONT.getlength(w) for w in words) + sp * (len(words) - 1)
        f = CAP_FONT.font_variant(size=int(92 * min(1.0, 990 / tw)))
        sp = f.getlength(" ")
        tw = sum(f.getlength(w) for w in words) + sp * (len(words) - 1)
        img = Image.new("RGBA", (W, 170), (0, 0, 0, 0))
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d, ds = ImageDraw.Draw(img), ImageDraw.Draw(sh)
        x = (W - tw) / 2
        for i, w in enumerate(words):
            ds.text((x + 5, 40), w, font=f, fill=(0, 0, 0, 160), stroke_width=10, stroke_fill=(0, 0, 0, 160))
            d.text((x, 30), w, font=f, fill=(255, 214, 0) if i == active else (255, 255, 255), stroke_width=9, stroke_fill=(30, 20, 40))
            x += f.getlength(w) + sp
        sh = sh.filter(ImageFilter.GaussianBlur(6))
        sh.alpha_composite(img)
        _cap[key] = sh
    return _cap[key]


def draw_caption(layer, t):
    for ci, words in enumerate(CAPTIONS):
        start = words[0][1]
        nxt = CAPTIONS[ci + 1][0][1] if ci + 1 < len(CAPTIONS) else 99
        end = min(nxt, words[-1][1] + 1.0)
        if start <= t < end:
            active = max(i for i, (_, s) in enumerate(words) if t >= s)
            img = caption_img(ci, active)
            sc = lerp(0.6, 1.0, back_out((t - start) / 0.14))
            sc *= 1 + 0.05 * kick(t, words[active][1], 0.08)
            if abs(sc - 1) > 0.002:
                img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.BILINEAR)
            layer.alpha_composite(img, (int(W / 2 - img.width / 2), int(1620 - img.height / 2)))
            return


# ---------------------------------------------------------------- shots


def speaker(t):
    for a, b, who in ((0, 3.9, "A"), (8.8, 11.3, "B"), (11.3, 11.95, "A"), (30.7, 38.55, "B"), (38.55, 39.4, "A"),
                      (39.4, 42.55, "B"), (42.55, 43.65, "A")):
        if a <= t < b:
            return who
    return None


def frame(t, m, seed):
    img = Image.new("RGB", (W * SS, H * SS), (238, 230, 214))
    wide, closeA, closeB = (540, 1090, 1.35), (330, 1020, 1.85), (760, 1020, 1.85)

    def P(c, push=0.0, sx=0.0):
        return Pen(img, c[0], c[1], c[2] * (1 + push), sx=sx, seed=seed)

    sp = speaker(t)
    mA = m if sp == "A" else 0.0
    mB = m if sp == "B" else 0.0

    if t < 1.8:  # wide intro: A reads the question
        pen = P(wide, 0.06 * ease_io(t / 1.8))
        studio(pen, t, dict(pose="phone", prop="phone", mouth=mA, eyes="half"), dict(eyes="open"))
    elif t < 3.9:
        pen = P(closeA, 0.08 * ease_io((t - 1.8) / 2.1))
        studio(pen, t, dict(pose="phone", prop="phone", mouth=mA, eyes="up" if t > 3.1 else "half", brows="up" if t > 3.1 else None),
               dict())
    elif t < 6.1:  # market walk
        lt = t - 3.9
        pen = P((lerp(380, 600, ease_io(lt / 2.2)), 1120, 1.5))
        market(pen, t, lt)
        x = lerp(150, 520, lt / 2.2)
        person(pen, x + 260, 1240, 1.1, RICH_GIRL, facing=1, pose="leash", walk=True, t=t, eyes="happy")
        dog(pen, x + 470, 1345, 0.9, 1, t)
        pen.line([(x + 340, 1180), (x + 455, 1290)], width=4, fill=(220, 60, 120))
        person(pen, x, 1240, 1.1, NARRATOR, facing=1, pose="push", walk=True, t=t, eyes="open")
        cart(pen, x + 120, 1345, 0.85, 0)
    elif t < 8.8:  # cart full of dog stuff, then the dog
        lt = t - 6.1
        items = int(clamp((t - 6.2) / 0.25, 0, 7))
        z = lerp(1.6, 2.2, ease_io((t - 7.4) / 1.2)) if t > 7.4 else 1.6
        cx = lerp(420, 640, ease_io((t - 7.4) / 1.2)) if t > 7.4 else 420
        pen = P((cx, lerp(1080, 1170, ease_io((t - 7.4) / 1.2)) if t > 7.4 else 1080, z))
        market(pen, t, lt)
        cart(pen, 420, 1345, 1.0, items)
        dog(pen, 640, 1345, 1.0, -1, t)
        if t > 8.18:
            for k in range(5):
                a = math.radians(k * 72 + t * 90)
                stick.stars_burst(pen, 640, 1250, t - 8.18) if k == 0 else None
    elif t < 9.9:  # B: Jesus Christ
        pen = P(closeB, 0.12 * kick(t, 8.92, 0.2))
        studio(pen, t, dict(), dict(pose="handshead", mouth=mB, eyes="huge", brows="up"))
    elif t < 11.3:
        pen = P(wide, 0.05 * kick(t, 10.56, 0.15))
        studio(pen, t, dict(eyes="happy", mouth=0.0), dict(pose="point", mouth=mB, eyes="open"))
    elif t < 12.0:
        pen = P(closeA, 0.06 * kick(t, 11.36, 0.12))
        studio(pen, t, dict(pose="talk", mouth=mA, eyes="happy"), dict())
    elif t < 17.1:  # crypto loss, still normal
        pen = P((560, 1040, 1.45 + 0.03 * (t - 12)))
        room_bg(pen)
        crash_screen(pen, t, 760, 760, 380, 300)
        calm = t > 14.6
        person(pen, 380, 1280, 1.6, CRYPTO_GUY, facing=1, pose="sip" if calm else "phone", prop="cup" if calm else "phone",
               eyes="happy" if calm else "open", mouth=0.0, t=t)
        if t > 12.66:
            k = ease_out((t - 12.66) / 0.25)
            text(pen, 760, 1000 + 20 * (1 - k), "-$30,000", 70 * (0.6 + 0.4 * k), fill=(240, 60, 60), sw=7)
        if t > 15.28:
            text(pen, 360, 760, "UNBOTHERED", 46, fill=(120, 200, 255), sw=6)
    elif t < 20.6 or (34.1 <= t < 36.7):  # chatting with dad, then laughing it off
        laugh = t > 19.9 or t >= 34.1
        pen = P((540, 1110, 1.5 + (0.15 * ease_io((t - 19.9) / 0.6) if 17 < t < 21 and laugh else 0)))
        living(pen)
        bob = 8 * math.sin(t * 20) if laugh else 0
        talking = int(t * 3) % 2
        person(pen, 360, 1240 + bob, 1.35, CRYPTO_GUY, facing=1, pose="belly" if laugh else "talk", sitting=True, t=t,
               mouth="laugh" if laugh else (0.6 * abs(math.sin(t * 14)) if talking else 0.0), eyes="happy" if laugh else "open")
        person(pen, 720, 1240 - bob, 1.35, DAD, facing=-1, pose="belly" if laugh else "rest", sitting=True, t=t,
               mouth="laugh" if laugh else (0.6 * abs(math.sin(t * 14)) if not talking else 0.0), eyes="happy" if laugh else "open")
        if laugh:
            ha(pen, 300, 800, t, 0)
            ha(pen, 790, 760, t, 0.5)
        if t >= 34.1:
            for k in range(4):
                a = (t - 34.1) * 0.6 + k * 0.25
                money_bag(pen, 180 + k * 230 + 30 * math.sin(t * 3 + k), 1150 - (a % 1) * 900, 1.0)
            text(pen, 540, 560, "30 BAGS!", 70, fill=(120, 220, 120), sw=8)
    elif t < 22.85:  # I lost 1k
        pen = P((540, 1060, 1.5))
        room_bg(pen, wall=(200, 205, 225), floor=(170, 160, 150))
        for k in range(5):  # rain cloud
            pen.ellipse(440 + k * 50, 640 - (k % 2) * 30, 70, 55, fill=(150, 155, 170), width=6)
        for k in range(10):
            ry = 720 + ((t * 600 + k * 70) % 320)
            pen.line([(410 + k * 30, ry), (402 + k * 30, ry + 26)], width=4, fill=(110, 160, 230))
        person(pen, 540, 1300, 1.4, NARRATOR, facing=1, pose="phone", prop="phone", eyes="cry", brows="sad", mouth="frown", t=t, sitting=True)
        if t > 21.1:
            text(pen, 540, 1500, "-$1,000", 66, fill=(240, 60, 60), sw=7)
    elif t < 26.0:  # grace of God
        pen = P((540, 1080, lerp(1.35, 1.55, ease_io((t - 22.85) / 3.1))))
        room_bg(pen, wall=(225, 218, 240), floor=(185, 170, 160))
        g = clamp((t - 25.0) / 0.4, 0, 1)
        beam = tuple(int(lerp(c, 255, 0.55 + 0.35 * g)) for c in (225, 218, 240))
        pen.poly([(470, -200), (610, -200), (740, 1400), (340, 1400)], fill=(255, 245, 190), outline=None)
        for k in range(8):
            a = t * 2 + k
            stick.stars_burst(pen, 540 + 160 * math.cos(a), 900 + 120 * math.sin(a * 1.3), ((t * 1.5 + k * 0.13) % 1) * 0.6)
        person(pen, 540, 1290, 1.4, NARRATOR, facing=1, pose="pray", eyes="closed", mouth=0.0, t=t, sitting=True)
        pen.ellipse(552, 1290 - 95 * 1.4 - 40 * 1.4 * 2.1, 46, 12, fill=None, outline=(250, 210, 60), width=7)
    elif t < 30.7:  # both lose it
        pen = P(wide, 0.04 * math.sin(t * 2))
        studio(pen, t, dict(pose="belly", mouth="laugh", eyes="happy", lean=18, bob=6, prop=None),
               dict(pose="facepalm", mouth="laugh", eyes="happy", lean=-10, bob=8, bobf=14))
        ha(pen, 300, 760, t, 0)
        ha(pen, 780, 720, t, 0.4)
    elif t < 32.5:
        pen = P(closeB, 0.05 * math.sin(t * 3))
        studio(pen, t, dict(), dict(pose="facepalm", mouth=mB if t < 31.2 else "laugh", eyes="happy", bob=6, bobf=14))
    elif t < 34.1:
        pen = P(closeB, 0.1 * kick(t, 32.58, 0.18))
        studio(pen, t, dict(), dict(pose="shrug", mouth=mB, eyes="huge", brows="up"))
    elif t < 38.55:
        pen = P(wide if t < 37.6 else closeB)
        studio(pen, t, dict(eyes="half", mouth=0.0, prop="phone", pose="phone"), dict(pose="talk", mouth=mB, eyes="open", lean=-10))
    elif t < 39.4:  # Billions
        pen = P(closeA, 0.1 * kick(t, 38.64, 0.2))
        studio(pen, t, dict(pose="phone", prop="phone", mouth=mA if t < 39.0 else "flat", eyes="half"), dict())
        for k in range(9):
            cy = 300 + ((t - 38.55) * 900 + k * 130) % 1200
            coin(pen, 80 + k * 70 + 20 * math.sin(k), cy, 1.0)
    elif t < 42.55:  # I need that gen money
        pen = P(closeB, 0.06 * ease_io((t - 39.4) / 3))
        studio(pen, t, dict(), dict(pose="pray", mouth=mB, eyes="up", lean=-6))
        stick.stars_burst(pen, 700, 650, ((t - 39.4) * 0.9) % 0.6)
        for k in range(3):
            a = (t * 0.5 + k / 3) % 1
            money_bag(pen, 900 - k * 60, 760 - a * 300, 0.6 + 0.2 * a)
    elif t < 43.65:  # That's energy
        pen = P(wide, 0.05 * kick(t, 42.62, 0.15))
        studio(pen, t, dict(pose="point", mouth=mA, eyes="happy"), dict(mouth=0.4 * abs(math.sin(t * 12)), eyes="happy"))
    elif t < 47.15:  # semester break chat
        pen = P((540, 1080, 1.45))
        park(pen, t)
        talking = int(t * 2.5) % 2
        person(pen, 400, 1185, 1.25, NARRATOR, facing=1, pose="talk" if talking else "rest", sitting=True, t=t,
               mouth=0.6 * abs(math.sin(t * 14)) if talking else 0.0)
        person(pen, 680, 1185, 1.25, RICH_GIRL, facing=-1, pose="rest" if talking else "talk", sitting=True, t=t,
               mouth=0.6 * abs(math.sin(t * 14)) if not talking else 0.0)
    elif t < 49.25:  # cousins in the village
        lt = t - 47.15
        pen = P((470, 1000, 1.55))
        park(pen, t)
        person(pen, 400, 1185, 1.25, NARRATOR, facing=1, pose="think", sitting=True, t=t, eyes="happy")
        person(pen, 680, 1185, 1.25, RICH_GIRL, facing=-1, pose="rest", sitting=True, t=t, eyes="open")
        k = ease_out(lt / 0.3)
        if k > 0:
            bubble(pen, 420, 640, 230 * k, 160 * k, 420, 900)
            if k > 0.9:
                village(pen, 420, 640, 0.95)
    elif t < 50.8:  # Rome in Italy
        lt = t - 49.25
        pen = P((620, 1000, 1.55))
        park(pen, t)
        person(pen, 400, 1185, 1.25, NARRATOR, facing=1, pose="rest", sitting=True, t=t, eyes="huge", mouth="O")
        person(pen, 680, 1185, 1.25, RICH_GIRL, facing=-1, pose="think", sitting=True, t=t, eyes="happy")
        k = ease_out(lt / 0.3)
        if k > 0:
            bubble(pen, 690, 640, 240 * k, 170 * k, 690, 900)
            if k > 0.9:
                rome(pen, 690, 650, 0.9, t)
    else:  # final laugh
        lt = t - 50.8
        pen = P(wide if lt < 3.5 else (closeA if lt < 5.2 else wide), 0.03 * math.sin(t * 2))
        studio(pen, t, dict(pose="belly", mouth="laugh", eyes="happy", lean=22, bob=7),
               dict(pose="facepalm", mouth="laugh", eyes="happy", lean=-14, bob=8, bobf=13))
        ha(pen, 300, 760, t, 0)
        ha(pen, 780, 720, t, 0.4)
    return img.resize((W, H), Image.LANCZOS)


def main(audio_path, out):
    a, sr = load_audio(audio_path)
    track = talk_track(a, sr)
    tmp = out.with_suffix(".video.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-tune", "animation",
         "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    n = int(round(END * FPS))
    maps, paper = st.make_boil_maps(n=3, amp=1.6), st.make_paper()
    for di in range((n + 1) // 2):
        i = 2 * di
        t = i / FPS
        m = max(track[i], track[i + 1])
        f = np.asarray(frame(t, m, seed=di), np.float32)
        mx, my = maps[di % len(maps)]
        f = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * paper
        img = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
        for k in range(2 if i + 1 < n else 1):
            tt = (i + k) / FPS
            cap = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw_caption(cap, tt)
            comp = img.copy()
            comp.alpha_composite(cap)
            px = np.asarray(comp.convert("RGB"), np.float32)
            if tt > END - 0.4:
                px *= clamp((END - tt) / 0.4, 0, 1)
            ff.stdin.write(px.astype(np.uint8).tobytes())
        if di % 150 == 0:
            print(f"drawing {di}/{n // 2}", flush=True)
    ff.stdin.close()
    ff.wait()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-i", str(audio_path), "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-t", str(END), "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink()
    print("done", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
