"""When Her Smile Takes Away Yours - chalkboard stick animation.

Classic stick figures drawn in chalk on a dark green board: chalky grain,
coloured-chalk accents, handwritten captions. Two hosts read relationship
tweets; each tweet is acted out, then the hosts fall apart laughing.
Audio is the original clip, unchanged.

Usage: python3 render.py <original_audio.wav> <output.mp4>
"""
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("tw", HERE.parent / "tweets-podcast" / "render.py")
tw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tw)
rf = tw.rf
Pen, RotPen, SS = tw.Pen, tw.RotPen, tw.SS
clamp, lerp, ease_io, ease_out, kick = tw.clamp, tw.lerp, tw.ease_io, tw.ease_out, tw.kick

W, H, FPS = 1080, 1920, 30
END = 61.19

BOARD = (34, 52, 44)
CHALK = (236, 236, 226)
DIM = (150, 165, 155)
YEL, PINK, BLUE, ORANGE, GREEN, RED = (246, 226, 120), (242, 150, 175), (140, 200, 240), (245, 170, 90), (150, 220, 140), (240, 110, 100)

FONT_PATH = HERE.parent / "smooth-talker-video" / "fonts" / "PermanentMarker.ttf"
FLOOR, HIP = 1350, 1235
SEAT = {"A": (320, 1), "B": (760, -1)}
S = 1.7

HOST_A = dict(acc=BLUE, hair="fade", beard=True)
HOST_B = dict(acc=ORANGE, hair="fade", beard=True)
YOU = dict(acc=GREEN, hair="fade")
GF = dict(acc=PINK, hair="long", dress=True)


def tint(c, k):
    """Chalk colour smudged into the board (k = coverage)."""
    return tuple(int(b + (a - b) * k) for a, b in zip(c, BOARD))


# ------------------------------------------------------------------ text

_fonts = {}


def ctext(pen, x, y, s, size, fill=CHALK):
    px = max(6, int(size * pen.z * SS))
    f = _fonts.get(px) or _fonts.setdefault(px, ImageFont.truetype(str(FONT_PATH), px))
    cx, cy = pen.p(x, y)
    tw_ = f.getlength(s)
    pen.d.text((cx - tw_ / 2, cy - px * 0.6), s, font=f, fill=fill)


def bubble(pen, x, y, w, h, tx, ty, col=CHALK):
    pen.poly([(x - w / 2, y - h / 2), (x + w / 2, y - h / 2), (x + w / 2, y + h / 2), (x - w / 2, y + h / 2)], fill=BOARD, outline=col, width=6)
    pen.poly([(x - 20, y + h / 2 - 3), (x + 20, y + h / 2 - 3), (tx, ty)], fill=BOARD, outline=None)
    pen.line([(x - 20, y + h / 2), (tx, ty), (x + 20, y + h / 2)], width=6, fill=col)


# ------------------------------------------------------------------ figure

POSES = dict(rf.POSES)
POSES.update({"wave": ((10, 100), (55, -95)), "hold": ((10, 100), (75, 62)), "thumb": ((10, 100), (70, -10)),
              "cross": ((40, 55), (45, 55)), "kneeslap": ((60, 95), (70, 95))})


def fig(pen, x, hip, s, look, facing=1, pose="rest", mouth=0.0, eyes="open", brows=None, t=0.0, sitting=False,
        lean=0.0, walk=False, prop=None, kick_=0.0, shake=0.0):
    r, T, ua, fa = 34 * s, 100 * s, 48 * s, 46 * s
    lw = 5.2 * s
    ph = (x * 0.0137 + (1.7 if facing < 0 else 0)) % 6.28
    talk = float(mouth) if isinstance(mouth, (int, float)) else (0.8 if mouth == "laugh" else 0.0)
    if walk:
        hip -= abs(math.sin(t * 2 * math.pi * 1.8)) * 6 * s
    lean = lean + 2 * math.sin(t * 0.9 + ph) + (3 * math.sin(t * 7 + ph) if talk > 0.05 else 0)
    jig = shake * 6 * s * math.sin(t * 38 + ph)
    breathe = 2.5 * s * math.sin(t * 2.3 + ph)
    nx, ny = x + facing * lean * s + jig, hip - T + breathe
    nod = 4 * s * math.sin(t * 11 + ph) * min(1.0, talk * 1.4)
    hx, hy = nx + facing * 2 * s * math.sin(t * 1.3 + ph), ny - r - 6 * s + nod
    if eyes in ("open", "up") and ((t + ph) % 3.4) < 0.13:
        eyes = "closed"
    targets = [list(p) for p in POSES[pose]]
    if walk and pose == "walk":
        sw = math.sin(t * 2 * math.pi * 1.8) * 35
        targets = [[-sw, 100], [sw, 100]]
    if talk > 0.05 and pose in ("talk", "point", "shrug", "rest", "think"):
        g = min(1.0, talk * 1.5)
        targets[1][0] += (14 * math.sin(t * 5.3 + ph) + (25 if pose == "rest" else 0)) * g
        targets[1][1] += (-22 * abs(math.sin(t * 4.1 + ph)) - (45 if pose == "rest" else 0)) * g
    if pose == "belly":  # laughing: hands bounce on belly
        targets[0][1] += 8 * math.sin(t * 20)
        targets[1][1] += 8 * math.sin(t * 20 + 1)
    if pose == "kneeslap":
        targets[1][1] -= 30 * abs(math.sin(t * 9))
    shoulders = [(nx - facing * 4 * s, ny + 10 * s), (nx + facing * 4 * s, ny + 10 * s)]

    def arm(i):
        sx, sy = shoulders[i]
        tx, ty = nx + facing * targets[i][0] * s, ny + targets[i][1] * s
        (ex, ey), (px, py) = rf.ik(sx, sy, tx, ty, ua, fa)
        pen.line([(sx, sy), (ex, ey), (px, py)], width=lw, fill=CHALK)
        pen.ellipse(px, py, 6 * s, 6 * s, fill=CHALK, outline=None)
        return px, py

    arm(0)
    if sitting:
        for k in range(2):
            kx = x + facing * 50 * s + (k * 8 * s * facing)
            ky = hip - 4 * s - kick_ * 30 * s * abs(math.sin(t * 18 + k * 1.6))
            fx, fy = kx + facing * 4 * s, hip + 60 * s
            pen.line([(x, hip), (kx, ky), (fx, fy)], width=lw, fill=CHALK)
            pen.line([(fx, fy), (fx + facing * 16 * s, fy)], width=lw, fill=CHALK)
    else:
        if walk:
            w = t * 2 * math.pi * 1.8
            angs = (30 * math.sin(w), -30 * math.sin(w))
        else:
            angs = (9, -9)
        for a in angs:
            fx, fy = x + math.sin(math.radians(a)) * 100 * s * facing, hip + math.cos(math.radians(a)) * 100 * s
            pen.line([(x, hip), (fx, fy)], width=lw, fill=CHALK)
            pen.line([(fx, fy), (fx + facing * 16 * s, fy)], width=lw, fill=CHALK)
    # torso + coloured-chalk shirt scribble
    pen.line([(nx, ny), (x, hip)], width=lw, fill=CHALK)
    if look.get("dress"):
        pen.poly([(nx, ny + 8 * s), (x + 40 * s, hip + 48 * s), (x - 40 * s, hip + 48 * s)], fill=tint(look["acc"], 0.35),
                 outline=look["acc"], width=lw * 0.8)
    else:
        zz = []
        for k in range(7):
            f = (k + 0.5) / 7
            cx, cy = lerp(nx, x, f), lerp(ny + 8 * s, hip - 4 * s, f)
            zz.append((cx + (14 * s if k % 2 else -14 * s), cy))
        pen.line(zz, width=lw * 0.7, fill=look["acc"])
    # head
    pen.ellipse(hx, hy, r, r, fill=BOARD, outline=CHALK, width=lw)
    hair = look.get("hair")
    if hair == "fade":
        pen.arc(hx, hy, r * 0.93, r * 0.93, 200, 340, width=lw * 1.5, fill=CHALK)
    elif hair == "long":
        for d in (-1, 1):
            pen.line([(hx + d * r * 0.95, hy - r * 0.2), (hx + d * r * 1.1, hy + r * 1.3)], width=lw, fill=CHALK)
        pen.arc(hx, hy, r * 1.02, r * 1.02, 180, 360, width=lw * 1.3, fill=CHALK)
        pen.ellipse(hx - facing * r * 0.9, hy - r * 0.75, 8 * s, 8 * s, fill=look["acc"], outline=None)  # bow
    if look.get("beard"):
        pen.arc(hx, hy + r * 0.05, r * 0.9, r * 0.9, 25, 155, width=lw * 1.5, fill=DIM)
    ex = hx + facing * r * 0.28
    for dx in (-0.3, 0.3):
        cx, cy = ex + dx * r, hy - r * 0.12
        if eyes in ("closed",):
            pen.line([(cx - r * 0.12, cy), (cx + r * 0.12, cy)], width=lw * 0.7, fill=CHALK)
        elif eyes in ("happy", "laughcry"):
            pen.line([(cx - r * 0.13, cy + r * 0.06), (cx, cy - r * 0.08), (cx + r * 0.13, cy + r * 0.06)], width=lw * 0.7, fill=CHALK)
        elif eyes == "huge":
            pen.ellipse(cx, cy - r * 0.04, r * 0.18, r * 0.22, fill=BOARD, outline=CHALK, width=lw * 0.6)
            pen.ellipse(cx, cy - r * 0.04, r * 0.06, r * 0.06, fill=CHALK, outline=None)
        elif eyes == "flat":
            pen.line([(cx - r * 0.13, cy), (cx + r * 0.13, cy)], width=lw * 0.6, fill=CHALK)
            pen.ellipse(cx, cy + r * 0.05, r * 0.05, r * 0.05, fill=CHALK, outline=None)
        else:
            pen.ellipse(cx, cy, r * 0.07, r * 0.09, fill=CHALK, outline=None)
        if eyes in ("cry", "laughcry"):
            for k in range(2):
                ty = cy + r * 0.2 + ((t * 260 + k * 37) % (r * 1.3))
                pen.ellipse(cx, ty, r * 0.06, r * 0.09, fill=BLUE, outline=None)
    if brows == "sad":
        pen.line([(ex - r * 0.45, hy - r * 0.38), (ex - r * 0.15, hy - r * 0.48)], width=lw * 0.6, fill=CHALK)
        pen.line([(ex + r * 0.15, hy - r * 0.48), (ex + r * 0.45, hy - r * 0.38)], width=lw * 0.6, fill=CHALK)
    elif brows == "up":
        for dx in (-0.3, 0.3):
            pen.arc(ex + dx * r, hy - r * 0.5, r * 0.15, r * 0.09, 200, 340, width=lw * 0.6, fill=CHALK)
    elif brows == "angry":
        pen.line([(ex - r * 0.45, hy - r * 0.48), (ex - r * 0.12, hy - r * 0.36)], width=lw * 0.6, fill=CHALK)
        pen.line([(ex + r * 0.12, hy - r * 0.36), (ex + r * 0.45, hy - r * 0.48)], width=lw * 0.6, fill=CHALK)
    mx, my = hx + facing * r * 0.22, hy + r * 0.42
    if mouth == "laugh":
        o = 0.75 + 0.25 * abs(math.sin(t * 22))
        pen.poly([(mx - r * 0.32, my - r * 0.1), (mx + r * 0.32, my - r * 0.1), (mx + r * 0.16, my + r * 0.34 * o),
                  (mx - r * 0.16, my + r * 0.34 * o)], fill=tint(RED, 0.5), outline=CHALK, width=lw * 0.6)
    elif mouth == "frown":
        pen.arc(mx, my + r * 0.12, r * 0.2, r * 0.12, 200, 340, width=lw * 0.6, fill=CHALK)
    elif mouth == "flat":
        pen.line([(mx - r * 0.18, my), (mx + r * 0.18, my)], width=lw * 0.6, fill=CHALK)
    elif mouth == "O":
        pen.ellipse(mx, my, r * 0.12, r * 0.17, fill=tint(RED, 0.4), outline=CHALK, width=lw * 0.5)
    elif mouth == "zip":
        pen.line([(mx - r * 0.25, my), (mx + r * 0.25, my)], width=lw * 0.6, fill=CHALK)
        for k in range(5):
            zx = mx - r * 0.2 + k * r * 0.1
            pen.line([(zx, my - r * 0.06), (zx, my + r * 0.06)], width=lw * 0.4, fill=YEL)
    elif mouth == "grin":
        pen.arc(mx, my - r * 0.18, r * 0.32, r * 0.3, 15, 165, width=lw * 0.7, fill=CHALK)
    elif isinstance(mouth, str) or mouth < 0.12:
        pen.arc(mx, my - r * 0.1, r * 0.22, r * 0.14, 20, 160, width=lw * 0.6, fill=CHALK)
    else:
        h = r * (0.07 + 0.22 * mouth)
        pen.ellipse(mx, my, r * 0.2, h, fill=tint(RED, 0.5), outline=CHALK, width=lw * 0.5)
    px, py = arm(1)
    if prop == "phone":
        pen.rect(px - 13 * s, py - 30 * s, px + 13 * s, py + 6 * s, fill=tint(BLUE, 0.35), outline=CHALK, width=lw * 0.6)
    elif prop == "cup":
        pen.rect(px - 12 * s, py - 18 * s, px + 12 * s, py + 8 * s, fill=BOARD, outline=CHALK, width=lw * 0.6)
        for k in range(2):
            sx = px - 5 * s + k * 10 * s
            pen.line([(sx, py - 24 * s), (sx + 5 * s * math.sin(t * 6 + k), py - 38 * s), (sx, py - 52 * s)], width=3, fill=DIM)
    elif prop == "thumb":
        pen.line([(px, py), (px, py - 22 * s)], width=lw, fill=CHALK)
    return (hx, hy, r), (px, py)


# ------------------------------------------------------------------ studio


def studio_bg(pen, t):
    pen.line([(-900, FLOOR), (2000, FLOOR)], width=6, fill=CHALK)
    for k in range(9):  # floorboard ticks
        x = -60 + k * 140
        pen.line([(x, FLOOR + 40), (x + 70, FLOOR + 40)], width=3, fill=DIM)
    # posters
    for (x0, y0, x1, y1), c in (((60, 380, 300, 640), YEL), ((790, 380, 1030, 640), PINK)):
        pen.rect(x0, y0, x1, y1, fill=None, outline=CHALK, width=5)
        for k in range(4):
            pen.line([(x0 + 25, y0 + 40 + k * 55), (x1 - 25, y0 + 70 + k * 55)], width=4, fill=c)
    pen.rect(380, 470, 700, 570, fill=None, outline=YEL, width=6)
    ctext(pen, 540, 520, "THE POD", 58, fill=YEL)
    # lamp
    pen.line([(540, 640), (540, 760)], width=4, fill=DIM)
    pen.poly([(500, 760), (580, 760), (560, 720), (520, 720)], fill=tint(YEL, 0.4), outline=CHALK, width=4)


def chair(pen, x, facing, dx=0):
    x += dx
    pen.rect(x - 55, HIP + 6, x + 55, HIP + 24, fill=BOARD, outline=CHALK, width=5)
    bx = x - facing * 62
    pen.rect(bx - 12, HIP - 170, bx + 12, HIP + 10, fill=BOARD, outline=CHALK, width=5)
    pen.line([(x, HIP + 24), (x, FLOOR - 22)], width=6, fill=CHALK)
    pen.line([(x - 55, FLOOR - 10), (x + 55, FLOOR - 10)], width=6, fill=CHALK)
    for wx in (-55, 55):
        pen.ellipse(x + wx, FLOOR - 6, 8, 8, fill=CHALK, outline=None)


def mic(pen, x, facing):
    mx, my = x + facing * 118, HIP - 215
    pen.line([(x + facing * 330, 300), (x + facing * 230, my - 150), (mx, my - 20)], width=5, fill=DIM)
    pen.rect(mx - 20, my - 34, mx + 20, my + 34, fill=BOARD, outline=CHALK, width=5)
    for k in range(3):
        pen.line([(mx - 14, my - 16 + k * 16), (mx + 14, my - 16 + k * 16)], width=3, fill=DIM)


def host(pen, who, t, st):
    x, facing = SEAT[who]
    look = HOST_A if who == "A" else HOST_B
    if st.get("spin"):
        facing = facing if int(t * 7) % 2 == 0 else -facing
    if st.get("floor"):  # lying on the floor, rolling and kicking
        ph = 0 if who == "A" else 1.3
        roll = math.pi / 2 + 0.35 * math.sin(t * 6 + ph)
        rp = RotPen(pen, x - facing * 70, FLOOR, -facing * roll, facing * 250 + 25 * math.sin(t * 3 + ph), -15)
        fig(rp, x, HIP, S, look, facing=facing, pose="belly", mouth="laugh", eyes="laughcry", t=t, sitting=True, kick_=1.0)
        chair(pen, x, SEAT[who][1], dx=-facing * 30)
        return
    ang = -facing * st.get("ang", 0.0)
    rp = RotPen(pen, x - facing * 70, FLOOR, ang, facing * 250 * clamp(st.get("ang", 0) / (math.pi / 2), 0, 1), st.get("dy", 0))
    if st.get("chair", True):
        chair(rp, x, facing)
    fig(rp, x, HIP + st.get("bob", 0) * math.sin(t * 19), S, look, facing=facing, pose=st.get("pose", "rest"),
        mouth=st.get("mouth", 0.0), eyes=st.get("eyes", "open"), brows=st.get("brows"), t=t, sitting=True,
        lean=st.get("lean", 0), prop=st.get("prop"), kick_=st.get("kick", 0.0), shake=st.get("shake", 0.0))
    if st.get("ang", 0) < 0.2 and not st.get("nomic"):
        mic(pen, x, SEAT[who][1])


def studio(pen, t, a, b):
    studio_bg(pen, t)
    host(pen, "A", t, a)
    host(pen, "B", t, b)


def has(pen, t, x, y, k=0.0, size=60):
    if (t * 3 + k) % 1 < 0.6:
        ctext(pen, x + 6 * math.sin(t * 20 + k), y, "HA", size, fill=YEL)


LAUGH = dict(mouth="laugh", eyes="laughcry", pose="belly", bob=6, shake=1.0)

# ------------------------------------------------------------------ stories


def ground(pen, y=1400):
    pen.line([(-900, y), (2000, y)], width=6, fill=CHALK)


def heart(pen, x, y, r, fill):
    pts = []
    for i in range(24):
        a = 2 * math.pi * i / 24
        pts.append((x + r * math.sin(a) ** 3, y - r * (13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)) / 16))
    pen.poly(pts, fill=fill, outline=CHALK, width=5)


def story_hating(pen, t):
    ground(pen)
    pen.rect(300, 1240, 780, 1270, fill=BOARD, outline=CHALK, width=6)  # bench
    for bx in (330, 750):
        pen.line([(bx, 1270), (bx, 1400)], width=6, fill=CHALK)
    k = ease_io((t - 1.6) / 2.4)
    fig(pen, 430, 1245, 1.35, YOU, facing=1, sitting=True, t=t, eyes="happy" if k < 0.4 else "flat",
        mouth="grin" if k < 0.4 else "flat", lean=-8 * k)
    fig(pen, 650, 1245, 1.35, GF, facing=-1, sitting=True, t=t, eyes="happy", mouth="grin", pose="talk")
    # love meter draining
    pen.rect(330, 560, 750, 640, fill=BOARD, outline=CHALK, width=6)
    fillw = 400 * (1 - k)
    if fillw > 4:
        pen.rect(340, 570, 340 + fillw, 630, fill=tint(PINK if k < 0.6 else RED, 0.7), outline=None)
    ctext(pen, 540, 520, "LOVE METER", 46, fill=PINK)
    heart(pen, 540, 760, 50 * (1 - 0.5 * k), tint(PINK, 0.6))
    if k > 0.7:
        pen.line([(510, 740), (545, 770), (525, 800), (560, 830)], width=6, fill=BOARD)  # crack


def story_hands(pen, t):
    ground(pen)
    cx = 540 + (t - 5.2) * -40
    fig(pen, cx - 110, 1300, 1.4, YOU, facing=1, pose="hold", walk=True, t=t, eyes="flat" if t > 6.24 else "open",
        mouth="flat")
    fig(pen, cx + 110, 1300, 1.4, GF, facing=-1, pose="hold", walk=True, t=t + 0.3, eyes="happy", mouth="grin")
    for k in range(4):  # little hearts from her
        a = ((t - 5.2) * 0.7 + k / 4) % 1
        heart(pen, cx + 110 + 40 * math.sin(a * 6 + k), 1000 - a * 300, 18, tint(PINK, 0.6))
    if t > 6.6:  # thought bubble: dotted CUT HERE line across the joined hands
        k = ease_out((t - 6.6) / 0.25)
        bubble(pen, 300, 700, 380 * k, 230 * k, 380, 950)
        if k > 0.9:
            pen.line([(170, 700), (300, 700)], width=8, fill=CHALK)  # arm
            pen.line([(300, 700), (430, 700)], width=8, fill=PINK)
            for i in range(6):
                pen.line([(300, 640 + i * 22), (300, 652 + i * 22)], width=5, fill=YEL)
            ctext(pen, 300, 615, "CUT HERE", 34, fill=YEL)
            sx = 330 + 10 * math.sin(t * 14)  # scissors
            pen.line([(sx, 760), (sx + 60, 800)], width=6, fill=CHALK)
            pen.line([(sx, 800), (sx + 60, 760)], width=6, fill=CHALK)
            pen.ellipse(sx + 75, 750, 14, 14, fill=None, outline=CHALK, width=5)
            pen.ellipse(sx + 75, 810, 14, 14, fill=None, outline=CHALK, width=5)


def story_problem(pen, t):
    ground(pen)
    fig(pen, 680, 1300, 1.45, GF, facing=-1, pose="cross", t=t, eyes="open", brows="angry", mouth=0.5 * abs(math.sin(t * 12)) if t < 14.7 else "flat")
    fig(pen, 330, 1300, 1.45, YOU, facing=1, pose="point" if t > 14.2 else "rest", t=t, eyes="flat", mouth="flat")
    bubble(pen, 700, 720, 420, 150, 690, 930)
    ctext(pen, 700, 720, "WHAT'S THE PROBLEM?", 40)
    if t > 14.22:
        k = ease_out((t - 14.22) / 0.25)
        ctext(pen, 640, 470, "THE PROBLEM", 60 * k + 1, fill=YEL)
        pen.line([(640, 520), (660, 560 + 220 * k)], width=8, fill=YEL)
        pen.poly([(635, 560 + 220 * k), (690, 560 + 220 * k), (665, 600 + 220 * k)], fill=YEL, outline=None)


def story_battery(pen, t):
    if t < 20.2:  # blissful hammock
        ground(pen)
        for tx in (170, 910):
            pen.line([(tx, 1400), (tx + 20, 820)], width=10, fill=CHALK)
            for a in (-150, -110, -70, -30):
                pen.line([(tx + 20, 820), (tx + 20 + 150 * math.cos(math.radians(a)), 820 + 80 * math.sin(math.radians(a)) + 60)],
                         width=7, fill=GREEN)
        pen.ellipse(840, 420, 70, 70, fill=tint(YEL, 0.5), outline=YEL, width=6)
        for k in range(8):
            a = k * math.pi / 4 + t * 0.4
            pen.line([(840 + 90 * math.cos(a), 420 + 90 * math.sin(a)), (840 + 120 * math.cos(a), 420 + 120 * math.sin(a))], width=5, fill=YEL)
        sway = 15 * math.sin(t * 2)
        pen.arc(560, 1000 + sway, 360, 140, 10, 170, width=7, fill=CHALK)
        rp = RotPen(pen, 560, 1080 + sway, -math.pi / 2 + 0.25)
        fig(rp, 560, 1080 + sway, 1.25, YOU, facing=1, pose="handshead", t=t, sitting=True, eyes="closed", mouth="grin")
        for k in range(3):
            a = ((t - 17.3) * 0.7 + k / 3) % 1
            ctext(pen, 600 + 80 * a + 30 * math.sin(a * 8), 960 - a * 300, "~", 50, fill=YEL)
    else:  # her dead phone
        pen.rect(360, 520, 720, 1220, fill=BOARD, outline=CHALK, width=8)
        pen.rect(470, 760, 610, 900, fill=BOARD, outline=RED, width=8)
        pen.rect(610, 805, 630, 855, fill=RED, outline=None)
        if int(t * 4) % 2:
            pen.rect(480, 770, 500, 890, fill=RED, outline=None)
        ctext(pen, 540, 980, "0%", 80, fill=RED)
        ctext(pen, 540, 1100, "BATTERY DEAD", 40, fill=CHALK)
        fig(pen, 860, 1380, 1.2, GF, facing=-1, pose="phone", prop="phone", t=t, eyes="huge", mouth="O")


def story_silent(pen, t):
    ground(pen)
    pen.ellipse(150, 400, 60, 60, fill=tint(YEL, 0.5), outline=YEL, width=6)
    for k in range(3):  # birds
        bx = 300 + k * 160 + 30 * math.sin(t + k)
        pen.line([(bx - 25, 500 + k * 30), (bx, 515 + k * 30), (bx + 25, 500 + k * 30)], width=5, fill=CHALK)
    fig(pen, 330, 1240, 1.4, YOU, facing=1, sitting=True, pose="sip", prop="cup", t=t, eyes="happy", mouth="grin")
    pen.rect(240, 1240, 420, 1265, fill=BOARD, outline=CHALK, width=5)
    for k in range(2):
        a = ((t - 23.2) * 0.6 + k / 2) % 1
        ctext(pen, 440 + 40 * a, 900 - a * 250, "~", 46, fill=YEL)
    if t > 26.7:  # reveal: she's there, arms crossed, mouth zipped
        k = ease_out((t - 26.7) / 0.3)
        fig(pen, 760, 1300, 1.45, GF, facing=-1, pose="cross", t=t, eyes="flat", brows="angry", mouth="zip", lean=-20 * (1 - k))
        ctext(pen, 760, 760, "...", 90, fill=CHALK)
        if t > 27.7:
            k2 = ease_out((t - 27.7) / 0.25)
            ctext(pen, 540, 470, "SILENT TREATMENT", 58 * k2 + 1, fill=PINK)


def story_talk(pen, t):
    ground(pen)
    fig(pen, 700, 1300, 1.45, GF, facing=-1, pose="pray", t=t, eyes="open", brows="sad", mouth="frown")
    bubble(pen, 720, 720, 360, 140, 700, 930)
    ctext(pen, 720, 720, "TALK TO ME...", 42, fill=PINK)
    done = t > 34.84
    fig(pen, 330, 1300, 1.45, YOU, facing=1, pose="thumb" if done else "rest", prop="thumb" if done else None, t=t,
        eyes="flat", mouth="grin" if done else "flat")
    # effort meter
    pen.rect(140, 880, 200, 1120, fill=BOARD, outline=CHALK, width=5)
    lvl = 0.15 if t > 33.7 else 0.6
    pen.rect(148, 1112 - 224 * lvl, 192, 1112, fill=tint(GREEN if lvl > 0.3 else RED, 0.7), outline=None)
    ctext(pen, 170, 840, "EFFORT", 34, fill=YEL)
    if done:
        bubble(pen, 330, 760, 260, 120, 340, 950)
        ctext(pen, 330, 760, "IT'S FINE.", 42)


def story_call(pen, t):
    ground(pen)
    if t < 39.6:
        x = lerp(560, 840, ease_io((t - 37.4) / 2.2))
        fig(pen, x, 1300, 1.45, GF, facing=1, pose="wave", walk=True, t=t, eyes="happy", mouth=0.5 * abs(math.sin(t * 12)))
        bubble(pen, 520, 700, 520, 150, 640, 940)
        ctext(pen, 520, 700, "I'LL CALL YOU WHEN", 36, fill=PINK)
        ctext(pen, 520, 745, "I GET HOME!", 36, fill=PINK)
        fig(pen, 260, 1300, 1.45, YOU, facing=1, pose="wave", t=t, eyes="happy", mouth="grin")
    else:
        lt = t - 39.6
        fig(pen, 380, 1300, 1.55, YOU, facing=1, pose="hold", t=t, eyes="half" if False else "flat", mouth="grin")
        px, py = 520, 1130
        pen.rect(px - 50, py - 90, px + 50, py + 90, fill=BOARD, outline=CHALK, width=6)
        ctext(pen, px, py - 40, "OFF" if lt > 0.4 else "ON", 30, fill=RED if lt > 0.4 else GREEN)
        if lt > 1.0:  # battery yanked out and thrown away
            k = ease_out((lt - 1.0) / 0.9)
            bx, by = px + 600 * k, py - 700 * k + 900 * k * k
            rp = RotPen(pen, bx, by, k * 9)
            rp.rect(bx - 30, by - 50, bx + 30, by + 50, fill=tint(YEL, 0.5), outline=CHALK, width=5)
            rp.line([(bx - 10, by - 15), (bx + 10, by - 15)], width=4, fill=CHALK)
            if k > 0.95:
                ctext(pen, 860, 900, "YEET", 60, fill=YEL)


def story_smile(pen, t):
    ground(pen)
    k = ease_io((t - 46.0) / 0.9)
    fig(pen, 330, 1300, 1.5, YOU, facing=1, t=t, eyes="open" if k < 0.6 else "flat", mouth="grin" if k < 0.15 else "flat", brows="sad" if k > 0.6 else None)
    fig(pen, 760, 1300, 1.5, GF, facing=-1, t=t, eyes="happy", mouth="grin")
    if 0.1 < k < 0.98:  # his smile floats across to her
        sx, sy = lerp(370, 720, k), lerp(1060, 1060, k) - 140 * math.sin(math.pi * k)
        pen.arc(sx, sy - 20, 40, 30, 15, 165, width=8, fill=YEL)
        for d in (-1, 1):
            ctext(pen, sx + d * 50, sy - 40, "*", 30, fill=YEL)
    if k >= 0.98:
        hx, hy = 760 - 1.5 * 2, 1300 - 100 * 1.5 - 34 * 1.5 - 6 * 1.5
        pen.arc(hx - 10, hy + 10, 34 * 1.5 * 0.45, 34 * 1.5 * 0.4, 15, 165, width=8, fill=YEL)


def dog(pen, x, y, t, f=1):
    pen.ellipse(x, y, 70, 40, fill=BOARD, outline=CHALK, width=6)
    for dx in (-45, -20, 25, 50):
        pen.line([(x + f * dx, y + 30), (x + f * dx + 6 * math.sin(t * 9 + dx), y + 95)], width=6, fill=CHALK)
    pen.line([(x - f * 70, y - 10), (x - f * 110, y - 40 + 15 * math.sin(t * 20))], width=6, fill=CHALK)
    hx, hy = x + f * 70, y - 55
    pen.ellipse(hx, hy, 42, 36, fill=BOARD, outline=CHALK, width=6)
    pen.poly([(hx - f * 30, hy - 20), (hx - f * 45, hy + 30), (hx - f * 15, hy)], fill=tint(ORANGE, 0.5), outline=CHALK, width=4)
    pen.ellipse(hx + f * 12, hy - 8, 6, 6, fill=CHALK, outline=None)
    pen.ellipse(hx + f * 40, hy + 4, 8, 6, fill=CHALK, outline=None)
    tl = 30 + 20 * abs(math.sin(t * 16))
    pen.poly([(hx + f * 30, hy + 18), (hx + f * 50, hy + 18), (hx + f * 55, hy + 18 + tl), (hx + f * 30, hy + 18 + tl)], fill=tint(PINK, 0.8), outline=CHALK, width=4)


def story_kiss(pen, t):
    ground(pen)
    swap = t > 51.75
    fig(pen, 360, 1300, 1.5, YOU, facing=1, t=t, eyes="closed" if not swap else "flat", mouth="grin" if not swap else "frown",
        brows=None if not swap else "sad", pose="rest" if not swap else "facepalm")
    if not swap:
        fig(pen, 520, 1300, 1.5, GF, facing=-1, t=t, eyes="closed", mouth="O", lean=12)
        for k in range(3):
            a = ((t - 50.1) * 0.8 + k / 3) % 1
            heart(pen, 440 + 40 * math.sin(a * 6 + k), 1000 - a * 250, 20, tint(PINK, 0.6))
    else:
        dog(pen, 540, 1060 + 10 * math.sin(t * 12), t, f=-1)
        for k in range(5):  # slobber drops
            dy = ((t * 400 + k * 60) % 300)
            pen.ellipse(450 + k * 12, 1000 + dy, 7, 10, fill=BLUE, outline=None)
        ctext(pen, 540, 640, "SLURP", 80 + 10 * math.sin(t * 20), fill=PINK)


def story_weekend(pen, t):
    ground(pen)
    pen.rect(110, 480, 410, 720, fill=BOARD, outline=CHALK, width=6)
    pen.rect(110, 480, 410, 540, fill=tint(RED, 0.6), outline=CHALK, width=6)
    ctext(pen, 260, 630, "WEEKEND", 44, fill=YEL)
    pen.rect(620, 470, 960, 900, fill=BOARD, outline=CHALK, width=6)
    k = ease_out((t - 58.2) / 0.3)
    if k > 0:
        pen.rect(650, 560, 650 + 280 * k, 700, fill=tint(PINK, 0.4), outline=PINK, width=5)
        if k > 0.9:
            ctext(pen, 790, 610, "I'M COMING", 34, fill=CHALK)
            ctext(pen, 790, 660, "THIS WEEKEND <3", 30, fill=CHALK)
    cry = t > 59.5
    fig(pen, 400, 1300, 1.55, YOU, facing=1, pose="cry" if cry else "phone", prop=None if cry else "phone", t=t,
        eyes="cry" if cry else "huge", brows="sad", mouth="frown" if cry else "O")
    if cry:  # river of tears
        pool = 60 + 300 * ease_out((t - 59.5) / 1.2)
        pen.ellipse(400, 1405, pool, 22, fill=tint(BLUE, 0.6), outline=BLUE, width=4)
        ctext(pen, 540, 360, "I CRIED", 80, fill=BLUE)


# ------------------------------------------------------------------ captions

CAPTIONS = [
    [("THAT", 0.00), ("PHASE", 0.42), ("WHERE", 0.82), ("YOU", 1.06), ("SLOWLY", 1.42)],
    [("START", 1.82), ("HATING", 2.30), ("YOUR", 2.72), ("GIRLFRIEND", 3.04)],
    [("IS", 3.38), ("CRAZY...", 3.78)],
    [("WHEN", 4.38), ("SHE", 4.44), ("HOLDS", 4.62), ("YOUR", 4.94), ("HAND", 5.18), ("IN", 5.32), ("PUBLIC", 5.58)],
    [("AND", 5.94), ("YOU", 6.24), ("FEEL", 6.42), ("LIKE", 6.58), ("CUTTING", 6.84), ("IT", 6.98), ("OFF", 7.24)],
    [("WHEN", 12.18), ("THE", 12.70), ("PROBLEM", 13.04)],
    [("ASKS", 13.28), ("YOU", 13.66), ("WHAT", 13.80), ("THE", 13.94), ("PROBLEM", 14.22), ("IS", 14.60)],
    [("YOU'LL", 17.34), ("BE", 17.56), ("WONDERING", 17.68), ("WHY", 17.98)],
    [("YOUR", 18.22), ("DAY", 18.46), ("IS", 18.64), ("GOING", 18.80), ("SO", 19.02), ("WELL...", 19.54)],
    [("HER", 20.26), ("BATTERY", 20.50), ("DIED.", 20.86)],
    [("WHEN", 23.18), ("YOU'RE", 23.70), ("WONDERING", 23.94), ("WHY", 24.28)],
    [("YOU'RE", 24.48), ("HAVING", 24.70), ("SUCH", 24.86), ("A", 25.12), ("PEACEFUL", 25.36), ("DAY", 25.68)],
    [("AND", 26.10), ("IT", 26.32), ("TURNS", 26.48), ("OUT", 26.64), ("SHE", 26.80), ("WAS", 26.94)],
    [("GIVING", 27.18), ("YOU", 27.40), ("THE", 27.62), ("SILENT", 27.78), ("TREATMENT", 28.38)],
    [("WHEN", 31.30), ("SHE", 31.74), ("SAYS", 31.92), ("TALK", 32.12), ("TO", 32.46), ("ME", 32.68)],
    [("AND", 32.84), ("YOU", 33.02), ("KNOW", 33.14), ("SHE'S", 33.24), ("NOT", 33.48), ("WORTH", 33.68), ("THE", 33.86), ("EFFORT", 34.06)],
    [("SO", 34.38), ("YOU", 34.58), ("JUST", 34.66), ("TELL", 34.84), ("HER", 34.98), ("IT'S", 35.12), ("FINE", 35.32)],
    [("OHHHH!", 35.80)],
    [("WHEN", 37.44), ("SHE", 37.88), ("SAYS", 38.18), ("I'LL", 38.40), ("CALL", 38.70), ("YOU", 38.78)],
    [("WHEN", 38.92), ("I", 39.04), ("GET", 39.20), ("HOME", 39.42)],
    [("THEN", 39.64), ("YOU", 39.80), ("SWITCH", 39.92), ("OFF", 40.10), ("YOUR", 40.26), ("PHONE", 40.44)],
    [("AND", 40.58), ("REMOVE", 40.80), ("THE", 40.98), ("BATTERY", 41.60)],
    [("WHEN", 45.50), ("HER", 45.70), ("SMILE", 46.02), ("TAKES", 46.24), ("AWAY", 46.56), ("YOURS", 46.82)],
    [("WHEN", 50.08), ("HER", 50.52), ("KISSES", 50.82), ("START", 51.10), ("TO", 51.46), ("FEEL", 51.70)],
    [("LIKE", 51.80), ("BEING", 51.98), ("LICKED", 52.24), ("BY", 52.50), ("A", 52.76), ("DOG", 53.02)],
    [("WOW!", 54.54)],
    [("BRO!", 55.40)],
    [("I", 56.08), ("REMEMBER", 56.24), ("THIS", 56.52), ("ONE", 56.80), ("TIME", 57.04)],
    [("ON", 57.20), ("A", 57.40), ("WEEKEND", 57.52), ("SHE", 57.80), ("SAID", 58.10)],
    [("SHE", 58.24), ("WAS", 58.38), ("COMING...", 58.60)],
    [("YO,", 58.92), ("I", 59.64), ("CRIED.", 59.66)],
]
CAP_FONT = ImageFont.truetype(str(FONT_PATH), 84)
_cap = {}


def caption_img(ci, active):
    key = (ci, active)
    if key not in _cap:
        words = [w for w, _ in CAPTIONS[ci]]
        sp = CAP_FONT.getlength(" ")
        tw_ = sum(CAP_FONT.getlength(w) for w in words) + sp * (len(words) - 1)
        f = CAP_FONT.font_variant(size=int(84 * min(1.0, 980 / tw_)))
        sp = f.getlength(" ")
        tw_ = sum(f.getlength(w) for w in words) + sp * (len(words) - 1)
        img = Image.new("RGBA", (W, 170), (0, 0, 0, 0))
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d, ds = ImageDraw.Draw(img), ImageDraw.Draw(sh)
        x = (W - tw_) / 2
        for i, w in enumerate(words):
            ds.text((x + 4, 38), w, font=f, fill=(0, 0, 0, 170))
            d.text((x, 30), w, font=f, fill=YEL if i == active else CHALK)
            if i == active:
                d.line([(x, 30 + f.size * 1.15), (x + f.getlength(w), 30 + f.size * 1.12)], fill=YEL, width=6)
            x += f.getlength(w) + sp
        sh = sh.filter(ImageFilter.GaussianBlur(6))
        sh.alpha_composite(img)
        _cap[key] = sh
    return _cap[key]


def draw_caption(layer, t):
    for ci, words in enumerate(CAPTIONS):
        start = words[0][1]
        nxt = CAPTIONS[ci + 1][0][1] if ci + 1 < len(CAPTIONS) else 99
        if start <= t < min(nxt, words[-1][1] + 1.0):
            active = max(i for i, (_, s) in enumerate(words) if t >= s)
            img = caption_img(ci, active)
            sc = lerp(0.6, 1.0, rf.back_out((t - start) / 0.14)) * (1 + 0.05 * kick(t, words[active][1], 0.08))
            if abs(sc - 1) > 0.002:
                img = img.resize((max(1, int(img.width * sc)), max(1, int(img.height * sc))), Image.BILINEAR)
            layer.alpha_composite(img, (int(W / 2 - img.width / 2), int(1640 - img.height / 2)))
            return


# ------------------------------------------------------------------ timeline

SPEAK = [(0, 4.3, "A"), (4.3, 7.5, "B"), (12.1, 15.3, "A"), (17.3, 21.3, "B"), (23.1, 28.8, "A"), (31.3, 35.6, "B"),
         (35.6, 37.4, "A"), (37.4, 42.7, "A"), (43.2, 45.0, "B"), (45.5, 47.4, "B"), (50.0, 53.4, "A"), (54.4, 60.1, "B")]


def speaker(t):
    for a, b, w in SPEAK:
        if a <= t < b:
            return w
    return ""


def frame(img, t, m, seed):
    wide, cA, cB = (540, 1110, 1.38), (390, 1030, 1.75), (690, 1030, 1.75)
    story = (540, 1040, 1.3)

    def P(c, push=0.0, sx=0.0):
        return Pen(img, c[0], c[1], c[2] * (1 + push), sx=sx, seed=seed)

    sp = speaker(t)
    mA = m if sp == "A" else 0.0
    mB = m if sp == "B" else 0.0
    reader = dict(pose="phone", prop="phone", eyes="open")
    fall = lambda t0, d=0.45: (math.pi / 2) * ease_out((t - t0) / d) if t >= t0 else 0.0  # noqa: E731
    shk = lambda t0: 30 * kick(t, t0 + 0.4, 0.2) * math.sin(t * 70)  # noqa: E731

    if t < 1.2:
        studio(P(wide, 0.05 * ease_io(t / 1.2)), t, dict(reader, mouth=mA), dict())
    elif t < 4.3:
        story_hating(P(story), t)
    elif t < 5.2:
        studio(P(cB), t, dict(), dict(reader, mouth=mB))
    elif t < 7.5:
        story_hands(P(story), t)
    elif t < 12.1:  # laugh 1: A tips over backwards, B slaps his knee
        pen = P(wide, sx=shk(8.0))
        studio(pen, t, dict(LAUGH, ang=fall(8.0), kick=1.0 if t > 8.5 else 0, nomic=t > 8.0),
               dict(LAUGH, pose="kneeslap", lean=-12))
        has(pen, t, 760, 760, 0.4)
        if 8.35 < t < 8.9:
            ctext(pen, 260, 960, "CRASH!", 80, fill=YEL)
    elif t < 15.3:
        story_problem(P(story), t)
    elif t < 17.3:  # laugh 2: B spins in his chair
        pen = P(wide)
        studio(pen, t, dict(LAUGH, pose="facepalm"), dict(LAUGH, spin=True, nomic=True))
        has(pen, t, 320, 760, 0.0)
    elif t < 21.3:
        if t < 18.0:
            studio(P(cB), t, dict(), dict(reader, mouth=mB))
        else:
            story_battery(P(story), t)
    elif t < 23.1:  # laugh 3: B falls sideways off his chair
        pen = P(wide, sx=shk(21.5))
        studio(pen, t, dict(LAUGH, pose="facepalm", lean=12), dict(LAUGH, ang=fall(21.5), kick=1.0 if t > 22 else 0, nomic=True))
    elif t < 28.8:
        if t < 23.9:
            studio(P(cA), t, dict(reader, mouth=mA), dict())
        else:
            story_silent(P(story), t)
    elif t < 31.3:  # laugh 4: both end up on the floor
        pen = P(wide)
        studio(pen, t, dict(floor=True), dict(floor=True))
        if int(t * 4) % 2:
            ctext(pen, 540, 820, "HAHAHAHA", 80, fill=YEL)
    elif t < 35.6:
        if t < 32.0:
            studio(P(cB), t, dict(), dict(reader, mouth=mB))
        else:
            story_talk(P(story), t)
    elif t < 37.4:  # OHHHH
        pen = P(cA, 0.1 * kick(t, 35.8, 0.2))
        studio(pen, t, dict(pose="handshead", mouth=mA if mA else "O", eyes="huge", brows="up"), dict(LAUGH))
    elif t < 42.7:
        story_call(P(story), t)
    elif t < 45.5:  # laugh 5: B goes over backwards, A points
        pen = P(wide, sx=shk(43.4))
        studio(pen, t, dict(LAUGH, pose="point", lean=10), dict(LAUGH, mouth=mB if mB else "laugh", ang=fall(43.4),
                                                                kick=1.0 if t > 43.9 else 0, nomic=True))
    elif t < 47.4:
        if t < 45.95:
            studio(P(cB), t, dict(), dict(reader, mouth=mB))
        else:
            story_smile(P(story), t)
    elif t < 50.0:  # laugh 6: rolling on the floor
        pen = P(wide, 0.03 * math.sin(t * 3))
        studio(pen, t, dict(floor=True), dict(floor=True))
        if int(t * 4) % 2:
            ctext(pen, 540, 820, "HAHAHAHA", 80, fill=YEL)
    elif t < 53.4:
        if t < 50.6:
            studio(P(cA), t, dict(reader, mouth=mA), dict())
        else:
            story_kiss(P(story), t)
    elif t < 56.0:  # WOW / BRO
        pen = P(cB, 0.1 * kick(t, 54.54, 0.2) + 0.1 * kick(t, 55.4, 0.2))
        studio(pen, t, dict(LAUGH), dict(pose="handshead" if t < 55.4 else "point", mouth=mB if mB else "O", eyes="huge", brows="up", lean=-8))
    elif t < 60.1:
        if t < 56.9:
            studio(P(cB), t, dict(), dict(reader, mouth=mB))
        else:
            story_weekend(P(story), t)
    else:  # final: everybody down
        pen = P(wide)
        studio(pen, t, dict(floor=True), dict(floor=True))
        ctext(pen, 540, 820, "HAHAHAHA", 80, fill=YEL)


# ------------------------------------------------------------------ chalk board


def make_board():
    rng = np.random.default_rng(7)
    base = np.ones((H, W, 3), np.float32) * np.array(BOARD, np.float32)
    smudge = cv2.resize(rng.normal(0, 1, (48, 27)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    base += smudge[..., None] * 5
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img, "RGBA")
    for _ in range(9):  # faint eraser swirls
        x, y, rx, ry = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(120, 320), rng.uniform(60, 160)
        d.arc((x - rx, y - ry, x + rx, y + ry), rng.uniform(0, 180), rng.uniform(200, 360), fill=(255, 255, 255, 14), width=int(rng.uniform(30, 70)))
    img = img.filter(ImageFilter.GaussianBlur(3))
    a = np.asarray(img, np.float32)
    # wooden frame
    a[:22] = a[-22:] = (120, 85, 55)
    a[:, :22] = a[:, -22:] = (120, 85, 55)
    return a


def make_grain(n=3):
    rng = np.random.default_rng(11)
    out = []
    for _ in range(n):
        fine = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 0.7)
        streak = cv2.resize(rng.random((H // 3, W // 24)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        g = np.clip(0.35 + 0.55 * fine + 0.35 * streak, 0.25, 1.0)
        out.append(g[..., None].astype(np.float32))
    return out


def main(audio_path, out):
    a, sr = rf.load_audio(audio_path)
    rf.END = END
    track = rf.talk_track(a, sr)
    board = make_board()
    board_ss = Image.fromarray(np.clip(board, 0, 255).astype(np.uint8)).resize((W * SS, H * SS), Image.BILINEAR)
    grains = make_grain()
    tmp = out.with_suffix(".video.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-tune", "animation",
         "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    n = int(round(END * FPS))
    for di in range((n + 1) // 2):
        i = 2 * di
        t = i / FPS
        img = board_ss.copy()
        frame(img, t, max(track[i], track[min(i + 1, len(track) - 1)]), di)
        f = np.asarray(img.resize((W, H), Image.LANCZOS), np.float32)
        f = board + (f - board) * grains[di % len(grains)]  # chalk grain
        base = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
        for k in range(2 if i + 1 < n else 1):
            tt = (i + k) / FPS
            cap = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            draw_caption(cap, tt)
            comp = base.copy()
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
