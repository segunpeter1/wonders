"""When Her Smile Takes Away Yours - clean cartoon stickman style.

White round heads, bold outlines, pastel flat sets, red-box word captions and
a POV title card. Same timeline and audio as ../smile-takes-yours.

Usage: python3 smile_clean.py <original_audio.wav> <output.mp4>
"""
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import style as st  # noqa: E402

_spec = importlib.util.spec_from_file_location("fl", HERE.parent / "smile-takes-yours" / "render_flow.py")
fl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fl)
chalk, rf = fl.chalk, fl.rf
clamp, lerp, ease_io, ease_out, kick = fl.clamp, fl.lerp, fl.ease_io, fl.ease_out, fl.kick

W, H, SS, FPS = st.W, st.H, st.SS, 30
END = chalk.END
INK, WHITE = st.INK, st.WHITE
YEL, PINK, RED, BLUE, GREEN = (255, 214, 0), (255, 120, 170), (230, 50, 60), (90, 160, 240), (90, 190, 110)
FLOOR = 1250

# ------------------------------------------------------------------ cast

HOST_A = dict(hair="short", hc=(40, 30, 25), shirt=(40, 40, 48), beard=(40, 30, 25))
HOST_B = dict(hair="short", hc=(110, 70, 40), shirt=(240, 232, 210), beard=(110, 70, 40))
BF = dict(hair="spiky", hc=(110, 70, 40), shirt=(80, 160, 90))
GF = dict(hair="pony", hc=(70, 40, 30), shirt=(245, 130, 175), bow=(245, 80, 140))

POSES = {"rest": ((-30, 150), (40, 150)), "phone": ((30, 90), (60, 85)), "point": ((-30, 150), (160, 20)),
         "talk": ((-30, 150), (110, 50)), "shrug": ((-120, 10), (120, 10)), "handshead": ((-55, -100), (55, -100)),
         "facepalm": ((-20, 150), (25, -60)), "belly": ((-10, 110), (45, 110)), "pray": ((35, 30), (45, 30)),
         "cry": ((-10, -60), (30, -60)), "sip": ((-30, 150), (55, -20)), "cross": ((55, 80), (60, 70)),
         "wave": ((-30, 150), (90, -120)), "hold": ((-30, 150), (110, 120)), "thumb": ((-30, 150), (100, -20)),
         "kneeslap": ((60, 150), (95, 150)), "chin": ((-30, 150), (30, -10)), "headback": ((-10, 110), (45, 110))}


class RotCam(st.Cam):
    def __init__(self, base, px, py, ang, dx=0.0, dy=0.0):
        self.__dict__.update(base.__dict__)
        self.px, self.py, self.ca, self.sa, self.dx, self.dy = px, py, math.cos(ang), math.sin(ang), dx, dy

    def p(self, x, y):
        x, y = x - self.px, y - self.py
        x, y = x * self.ca - y * self.sa + self.px + self.dx, x * self.sa + y * self.ca + self.py + self.dy
        return st.Cam.p(self, x, y)

    def _swap(self):
        return abs(self.sa) > abs(self.ca)

    def rect(self, x0, y0, x1, y1, fill, outline=INK, width=5, r=0):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill, outline=outline, width=width)

    def ellipse(self, x, y, rx, ry, fill=None, outline=INK, width=6):
        cx, cy = self.p(x, y)
        if self._swap():
            rx, ry = ry, rx
        rx, ry = rx * self.z * SS, ry * self.z * SS
        self.d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fill, outline=outline, width=self.w(width) if outline else 0)

    def arc(self, x, y, rx, ry, a0, a1, width=5, fill=INK):
        n = max(6, int(abs(a1 - a0) / 10))
        pts = [(x + rx * math.cos(math.radians(a)), y + ry * math.sin(math.radians(a))) for a in np.linspace(a0, a1, n)]
        self.line(pts, width=width, fill=fill)


def person(c, x, hip, look, facing=1, pose="rest", mouth=0.0, eyes="open", brows="flat", t=0.0, sit=True, s=1.0,
           lean=0.0, walk=False, kick_=0.0, bounce=0.0, prop=None):
    r = 70 * s
    ph = (x * 0.013 + (1.9 if facing < 0 else 0)) % 6.28
    talk = float(mouth) if isinstance(mouth, (int, float)) else (0.8 if mouth == "laugh" else 0)
    hip -= bounce * abs(math.sin(t * 18)) * 10 * s
    lean = lean + 1.5 * math.sin(t * 0.9 + ph) + (3 * math.sin(t * 7 + ph) if talk > 0.05 else 0)
    nx, ny = x + facing * lean * s, hip - 160 * s + 2 * s * math.sin(t * 2.3 + ph)
    nod = 4 * s * math.sin(t * 11 + ph) * min(1.0, talk * 1.4)
    hx, hy = nx, ny - r * 0.95 + nod
    if eyes == "open" and ((t + ph) % 3.4) < 0.12:
        eyes = "closed"
    tg = [list(p) for p in POSES[pose]]
    if talk > 0.05 and pose in ("talk", "point", "rest", "chin", "shrug"):
        tg[1][0] += 14 * math.sin(t * 5.3 + ph)
        tg[1][1] += -22 * abs(math.sin(t * 4.1 + ph)) - (50 if pose == "rest" else 0)
    if pose in ("belly", "kneeslap"):
        tg[1][1] += 10 * math.sin(t * 20)
    lw = 7 * s

    def arm(i):
        sx, sy = nx, ny + 22 * s
        tx, ty = nx + facing * tg[i][0] * s, ny + tg[i][1] * s
        (ex, ey), (px, py) = rf.ik(sx, sy, tx, ty, 72 * s, 70 * s)
        c.line([(sx, sy), (ex, ey), (px, py)], width=lw)
        c.ellipse(px, py, 10 * s, 10 * s, WHITE, width=4 * s)
        return px, py

    arm(0)
    # legs
    if sit:
        for k, dx in enumerate((-8, 8)):
            kx = x + facing * 100 * s + dx * s
            ky = hip - kick_ * 40 * s * abs(math.sin(t * 16 + k * 1.7))
            fx, fy = kx + facing * 6 * s, hip + 150 * s - kick_ * 60 * s * abs(math.sin(t * 16 + k * 1.7))
            c.line([(x, hip), (kx, ky), (fx, fy)], width=lw)
            c.line([(fx, fy), (fx + facing * 22 * s, fy)], width=lw)
    else:
        sw = 28 * math.sin(t * 2 * math.pi * 1.8) if walk else 8
        for a in (sw, -sw):
            fx, fy = x + math.sin(math.radians(a)) * 175 * s * facing, hip + math.cos(math.radians(a)) * 175 * s
            c.line([(x, hip), (fx, fy)], width=lw)
            c.line([(fx, fy), (fx + facing * 22 * s, fy)], width=lw)
    # torso (shirt)
    if look is GF or look.get("bow"):
        c.poly([(nx - 24 * s, ny - 4 * s), (nx + 24 * s, ny - 4 * s), (x + 52 * s, hip + 40 * s), (x - 52 * s, hip + 40 * s)], look["shirt"], width=6 * s)
    else:
        c.rect(nx - 28 * s, ny - 6 * s, x + 28 * s, hip + 8 * s, look["shirt"], width=6 * s, r=12 * s) if abs(nx - x) < 1 else \
            c.poly([(nx - 28 * s, ny - 6 * s), (nx + 28 * s, ny - 6 * s), (x + 28 * s, hip + 8 * s), (x - 28 * s, hip + 8 * s)], look["shirt"], width=6 * s)
    # head
    hc = look["hc"]
    if look["hair"] == "pony":
        c.ellipse(hx - facing * r * 1.05, hy - r * 0.2, r * 0.32, r * 0.55, hc, width=5 * s)
    c.ellipse(hx, hy, r, r * 0.95, WHITE, width=7 * s)
    if look.get("beard"):
        c.arc(hx, hy + r * 0.04, r * 0.9, r * 0.86, 25, 155, width=r * 0.2, fill=look["beard"])
    if look["hair"] == "short":
        c.poly([(hx + r * math.cos(math.radians(a)), hy + r * 0.95 * math.sin(math.radians(a))) for a in range(195, 346, 15)]
               + [(hx + r * 0.75, hy - r * 0.55), (hx - r * 0.75, hy - r * 0.55)], hc, width=5 * s)
    elif look["hair"] == "spiky":
        for k in range(5):
            a = math.radians(-150 + k * 25)
            px, py = hx + math.cos(a) * r * 0.82, hy + math.sin(a) * r * 0.78
            c.poly([(px - 20 * s, py + 12 * s), (px + 20 * s, py + 12 * s), (px + 10 * s * facing, py - 34 * s)], hc, width=4 * s)
    elif look["hair"] == "pony":
        c.poly([(hx + r * math.cos(math.radians(a)), hy + r * 0.95 * math.sin(math.radians(a))) for a in range(190, 351, 16)]
               + [(hx + r * 0.6, hy - r * 0.5), (hx - r * 0.6, hy - r * 0.5)], hc, width=5 * s)
        c.ellipse(hx - facing * r * 0.85, hy - r * 0.6, 14 * s, 14 * s, look["bow"], width=4 * s)
    ex = hx + facing * r * 0.18
    for k, dx in enumerate((-0.32, 0.32)):
        cx, cy = ex + dx * r, hy - r * 0.05
        if eyes == "closed":
            c.line([(cx - r * 0.11, cy), (cx + r * 0.11, cy)], width=5 * s)
        elif eyes in ("happy", "laughcry"):
            c.line([(cx - r * 0.13, cy + r * 0.05), (cx, cy - r * 0.08), (cx + r * 0.13, cy + r * 0.05)], width=5 * s)
        elif eyes == "huge":
            c.ellipse(cx, cy, r * 0.17, r * 0.2, WHITE, width=4 * s)
            c.ellipse(cx, cy, r * 0.07, r * 0.08, INK, outline=None)
        elif eyes == "half":
            c.ellipse(cx, cy + r * 0.02, r * 0.09, r * 0.09, INK, outline=None)
            c.line([(cx - r * 0.14, cy - r * 0.04), (cx + r * 0.14, cy - r * 0.04)], width=5 * s)
        else:
            c.ellipse(cx, cy, r * 0.085, r * 0.105, INK, outline=None)
        if eyes in ("cry", "laughcry"):
            for j in range(2):
                ty = cy + r * 0.2 + ((t * 260 + j * 40) % (r * 1.2))
                c.ellipse(cx, ty, r * 0.06, r * 0.09, (120, 190, 255), width=2)
        by = cy - r * 0.3
        if brows == "flat":
            c.line([(cx - r * 0.15, by), (cx + r * 0.15, by)], width=6 * s)
        elif brows == "skeptic":
            tilt = r * 0.12 if k == (0 if facing > 0 else 1) else -r * 0.05
            c.line([(cx - r * 0.15, by + tilt), (cx + r * 0.15, by - tilt * 0.3)], width=6 * s)
        elif brows == "up":
            c.arc(cx, by + r * 0.05, r * 0.15, r * 0.1, 200, 340, width=6 * s)
        elif brows == "sad":
            d = 1 if dx < 0 else -1
            c.line([(cx - r * 0.15, by + d * r * 0.06), (cx + r * 0.15, by - d * r * 0.06)], width=6 * s)
        elif brows == "angry":
            d = -1 if dx < 0 else 1
            c.line([(cx - r * 0.15, by + d * r * 0.07), (cx + r * 0.15, by - d * r * 0.07)], width=6 * s)
    mx, my = hx + facing * r * 0.15, hy + r * 0.45
    if mouth == "laugh":
        o = 0.75 + 0.25 * abs(math.sin(t * 22))
        c.poly([(mx - r * 0.3, my - r * 0.1), (mx + r * 0.3, my - r * 0.1), (mx + r * 0.15, my + r * 0.32 * o), (mx - r * 0.15, my + r * 0.32 * o)],
               (110, 40, 45), width=5 * s)
        c.ellipse(mx, my + r * 0.2 * o, r * 0.12, r * 0.06, (235, 110, 120), outline=None)
    elif mouth == "frown":
        c.arc(mx, my + r * 0.12, r * 0.18, r * 0.12, 200, 340, width=5 * s)
    elif mouth == "smile":
        c.arc(mx, my - r * 0.12, r * 0.22, r * 0.16, 20, 160, width=5 * s)
    elif mouth == "grin":
        c.poly([(mx - r * 0.26, my - r * 0.06), (mx + r * 0.26, my - r * 0.06), (mx, my + r * 0.2)], WHITE, width=5 * s)
    elif mouth == "flat":
        c.line([(mx - r * 0.13, my), (mx + r * 0.13, my)], width=5 * s)
    elif mouth == "O":
        c.ellipse(mx, my, r * 0.11, r * 0.16, (110, 40, 45), width=5 * s)
    elif mouth == "zip":
        c.line([(mx - r * 0.2, my), (mx + r * 0.2, my)], width=5 * s)
        for j in range(5):
            zx = mx - r * 0.16 + j * r * 0.08
            c.line([(zx, my - r * 0.05), (zx, my + r * 0.05)], width=3 * s, fill=(160, 160, 170))
    elif isinstance(mouth, (int, float)) and mouth > 0.1:
        c.ellipse(mx, my, r * 0.18, r * (0.05 + 0.18 * mouth), (110, 40, 45), width=5 * s)
        c.ellipse(mx, my + r * 0.07 * mouth, r * 0.1, r * 0.05 * mouth, (235, 110, 120), outline=None)
    else:
        c.line([(mx - r * 0.11, my), (mx + r * 0.11, my - r * 0.02)], width=5 * s)
    px, py = arm(1)
    if prop == "phone":
        c.rect(px - 16 * s, py - 40 * s, px + 16 * s, py + 8 * s, (50, 50, 60), width=4 * s, r=5 * s)
    elif prop == "cup":
        c.rect(px - 16 * s, py - 24 * s, px + 16 * s, py + 10 * s, WHITE, width=4 * s, r=4 * s)
    elif prop == "thumb":
        c.line([(px, py), (px, py - 26 * s)], width=lw)
    return (hx, hy, r), (px, py)


# ------------------------------------------------------------------ sets


def studio(c):
    c.rect(-700, -600, 1800, FLOOR, (226, 214, 238), outline=None)
    c.rect(-700, FLOOR, 1800, 2700, (200, 186, 214), outline=None)
    c.rect(-700, FLOOR - 8, 1800, FLOOR + 8, (240, 232, 246), outline=None)
    for x in (-140, 960):  # acoustic panels
        for k in range(3):
            c.rect(x, 560 + k * 130, x + 220, 670 + k * 130, (70, 64, 84), width=4, r=10)
    for k, (x, col) in enumerate(((130, (255, 170, 120)), (330, (120, 190, 240)), (560, (250, 140, 190)), (760, (150, 210, 140)))):
        c.rect(x, 420, x + 170, 650, (250, 248, 240), width=6)
        c.rect(x + 14, 434, x + 156, 636, col, outline=None)
        c.ellipse(x + 85, 520, 34, 34, WHITE, width=4)
        c.line([(x + 50, 600), (x + 120, 600)], width=5)
    c.rect(330, 700, 750, 730, (200, 160, 120), width=5)  # shelf
    for k in range(6):
        c.rect(350 + k * 66, 650, 390 + k * 66, 700, [(240, 90, 80), (90, 150, 240), (250, 200, 60)][k % 3], width=4, r=6)
    for x in (40, 1040):  # ring lights
        c.ellipse(x, 760, 80, 80, None, outline=(255, 248, 210), width=18)
        c.line([(x, 840), (x, FLOOR)], width=6)
    c.rect(-300, 140, 1400, 200, (255, 236, 160), outline=None)  # light strip glow
    c.rect(440, 520, 640, 580, (60, 40, 80), width=4, r=10)
    fnt = ImageFont.truetype(str(st.CAP_FONT), int(42 * c.z * SS))
    px, py = c.p(540, 550)
    c.d.text((px - fnt.getlength("THE POD") / 2, py - 26 * c.z * SS), "THE POD", font=fnt, fill=(255, 130, 200))


def chair(c, x, facing, hip=1110):
    bx = x - facing * 70
    c.rect(bx - 22, hip - 230, bx + 22, hip + 20, (60, 60, 70), width=5, r=14)
    c.rect(x - 70, hip + 6, x + 70, hip + 34, (60, 60, 70), width=5, r=10)
    c.line([(x, hip + 34), (x, FLOOR - 30)], width=8)
    c.line([(x - 70, FLOOR - 14), (x + 70, FLOOR - 14)], width=8)
    for wx in (-70, 70):
        c.ellipse(x + wx, FLOOR - 8, 10, 10, (40, 40, 45), width=3)


def mic(c, x, facing, hip=1110):
    mx, my = x + facing * 150, hip - 290
    c.line([(mx + facing * 20, FLOOR - 4), (mx + facing * 20, my + 40), (mx, my + 30)], width=7, fill=(50, 50, 58))
    c.line([(mx - 30 + facing * 20, FLOOR - 4), (mx + 50 + facing * 20, FLOOR - 4)], width=8, fill=(50, 50, 58))
    c.rect(mx - 24, my - 44, mx + 24, my + 44, (40, 40, 48), width=5, r=20)


SEAT = {"A": 320, "B": 760}


def host(c, who, t, stt):
    x = SEAT[who]
    facing = 1 if who == "A" else -1
    look = HOST_A if who == "A" else HOST_B
    if stt.get("spin"):
        facing = facing if int(t * 7) % 2 == 0 else -facing
    if stt.get("floor"):
        ph = 0 if who == "A" else 1.3
        roll = math.pi / 2 + 0.3 * math.sin(t * 6 + ph)
        rc = RotCam(c, x - facing * 70, FLOOR, -facing * roll, facing * 280 + 20 * math.sin(t * 3 + ph), -20)
        person(rc, x, 1110, look, facing, "belly", "laugh", "laughcry", "up", t, kick_=1.0)
        chair(c, x - facing * 40, facing)
        return
    ang = -facing * stt.get("ang", 0.0)
    rc = RotCam(c, x - facing * 70, FLOOR, ang, facing * 280 * clamp(stt.get("ang", 0) / (math.pi / 2), 0, 1), 0)
    chair(rc, x, facing)
    person(rc, x, 1110, look, facing, stt.get("pose", "rest"), stt.get("mouth", 0.0), stt.get("eyes", "open"),
           stt.get("brows", "flat"), t, lean=stt.get("lean", 0), kick_=stt.get("kick", 0), bounce=stt.get("bounce", 0),
           prop=stt.get("prop"))
    if stt.get("ang", 0) < 0.2 and not stt.get("nomic"):
        mic(c, x, facing)


def pod(c, t, a, b):
    studio(c)
    host(c, "A", t, a)
    host(c, "B", t, b)


LAUGH = dict(mouth="laugh", eyes="laughcry", brows="up", pose="belly", bounce=1.0)


def ground(c, wall, floor, y=1250):
    c.rect(-700, -600, 1800, y, wall, outline=None)
    c.rect(-700, y, 1800, 2700, floor, outline=None)
    c.rect(-700, y - 6, 1800, y + 6, tuple(min(255, v + 20) for v in floor), outline=None)


def tree(c, x, y, s=1.0):
    c.rect(x - 25 * s, y - 330 * s, x + 25 * s, y, (150, 110, 80), width=5)
    for k in range(4):
        c.ellipse(x + (k - 1.5) * 70 * s, y - 380 * s - (k % 2) * 60 * s, 100 * s, 90 * s, (130, 190, 120), width=5)


def park(c):
    ground(c, (200, 232, 245), (170, 215, 150))
    tree(c, 60, 1250)
    tree(c, 1000, 1250, 0.9)
    c.ellipse(860, 360, 60, 60, (255, 225, 120), outline=None)


def bench(c, x0=300, x1=800, y=1110):
    c.rect(x0, y - 120, x1, y - 90, (190, 140, 100), width=5, r=6)
    c.rect(x0, y + 4, x1, y + 34, (190, 140, 100), width=5, r=6)
    for bx in (x0 + 30, x1 - 30):
        c.line([(bx, y + 34), (bx, FLOOR)], width=8)


def street(c, t):
    ground(c, (210, 228, 240), (190, 190, 200))
    for k, (bx, col) in enumerate(((-200, (240, 180, 160)), (140, (180, 200, 230)), (500, (240, 220, 160)), (860, (190, 220, 190)))):
        c.rect(bx, 520 - (k % 2) * 80, bx + 330, 1250, col, width=6)
        for r in range(4):
            for q in range(2):
                c.rect(bx + 50 + q * 140, 600 - (k % 2) * 80 + r * 150, bx + 130 + q * 140, 690 - (k % 2) * 80 + r * 150, (250, 248, 235), width=4)


def room(c, wall=(246, 232, 214), floor=(220, 196, 168)):
    ground(c, wall, floor)
    c.rect(700, 520, 980, 760, (250, 246, 236), width=6)
    c.poly([(720, 740), (800, 620), (880, 720), (960, 740)], (160, 200, 170), outline=None)
    c.rect(-60, 560, 260, 900, (200, 225, 240), width=7)
    c.line([(100, 560), (100, 900)], width=5, fill=(250, 248, 240))


def beach(c):
    ground(c, (170, 220, 250), (245, 225, 170), y=1180)
    c.rect(-700, 1060, 1800, 1180, (90, 190, 220), outline=None)
    c.ellipse(860, 380, 70, 70, (255, 225, 120), outline=None)
    for tx in (120, 960):
        c.line([(tx, 1250), (tx + 30, 640)], width=18, fill=(170, 120, 80))
        for a in (-160, -120, -60, -20):
            c.line([(tx + 30, 640), (tx + 30 + 160 * math.cos(math.radians(a)), 640 + 90 * math.sin(math.radians(a)) + 50)], width=12, fill=(90, 170, 90))


def garden(c):
    ground(c, (215, 238, 230), (160, 210, 140))
    for k in range(9):
        c.ellipse(-40 + k * 140, 1150, 80, 60, (120, 180, 110), width=4)
        c.ellipse(-40 + k * 140 + 20, 1120, 14, 14, [(250, 140, 170), (255, 220, 110), (250, 250, 250)][k % 3], outline=None)
    c.ellipse(140, 380, 60, 60, (255, 225, 120), outline=None)


def porch(c):
    ground(c, (60, 70, 120), (130, 120, 130))
    c.rect(160, 500, 920, 1250, (200, 180, 160), width=7)
    c.rect(560, 760, 800, 1250, (255, 220, 140), width=7)
    c.rect(560, 760, 620, 1250, (170, 110, 70), width=5)
    for k in range(25):
        c.ellipse((k * 97) % 1080, 100 + (k * 53) % 300, 3, 3, (255, 255, 230), outline=None)


def bedroom(c):
    room(c, wall=(232, 226, 246), floor=(200, 186, 170))
    c.rect(500, 1000, 1060, 1180, (200, 200, 230), width=6, r=20)
    c.rect(960, 880, 1060, 1180, (170, 130, 100), width=6)


# ------------------------------------------------------------------ overlays (screen space)


def drawn_heart(c, x, y, r, fill):
    pts = []
    for i in range(28):
        a = 2 * math.pi * i / 28
        pts.append((x + r * math.sin(a) ** 3, y - r * (13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)) / 16))
    c.poly(pts, fill, width=4)


def dog(c, x, y, t, f=-1):
    c.ellipse(x, y, 110, 60, (235, 190, 120), width=6)
    for dx in (-70, -30, 40, 80):
        c.line([(x + f * dx, y + 45), (x + f * dx + 6 * math.sin(t * 9 + dx), y + 140)], width=10)
    c.line([(x - f * 110, y - 10), (x - f * 160, y - 60 + 20 * math.sin(t * 20))], width=10)
    hx, hy = x + f * 110, y - 90
    c.ellipse(hx, hy, 68, 60, (235, 190, 120), width=6)
    c.poly([(hx - f * 50, hy - 30), (hx - f * 75, hy + 50), (hx - f * 25, hy)], (200, 150, 90), width=5)
    c.ellipse(hx + f * 20, hy - 14, 9, 10, INK, outline=None)
    c.ellipse(hx + f * 66, hy + 6, 13, 10, INK, outline=None)
    tl = 40 + 30 * abs(math.sin(t * 16))
    c.poly([(hx + f * 45, hy + 30), (hx + f * 75, hy + 30), (hx + f * 80, hy + 30 + tl), (hx + f * 45, hy + 30 + tl)], (240, 120, 140), width=4)


def frame(t, m):
    img = Image.new("RGB", (W * SS, H * SS), (230, 230, 230))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    sp = chalk.speaker(t)
    mA = m if sp == "A" else 0.0
    mB = m if sp == "B" else 0.0
    wide, cA, cB = (540, 1010, 1.32), (360, 900, 1.7), (720, 900, 1.7)
    reader = dict(pose="phone", prop="phone", eyes="half", brows="flat")
    fall = lambda t0, du=0.45: (math.pi / 2) * ease_out((t - t0) / du) if t >= t0 else 0.0  # noqa: E731

    def C(v, push=0.0, sx=0.0, sy=0.0):
        return st.Cam(img, v[0] - sx / v[2], v[1] - sy / v[2], v[2] * (1 + push))

    def shk(t0, a=34):
        k = kick(t, t0, 0.2) * a
        return k * math.sin(t * 71), k * math.cos(t * 53)

    if t < 1.2:
        pod(C(wide, 0.05 * ease_io(t / 1.2)), t, dict(reader, mouth=mA), dict(eyes="open", brows="flat"))
    elif t < 4.3:  # hating her slowly
        c = C((540, 1000, 1.25))
        park(c)
        bench(c)
        k = ease_io((t - 1.6) / 2.4)
        person(c, 430, 1110, BF, 1, "rest", "flat" if k > 0.4 else "smile", "half" if k > 0.4 else "open", "flat", t, lean=-6 * k)
        person(c, 640, 1110, GF, -1, "hold", "smile", "happy", "flat", t, lean=10)
        d.rounded_rectangle((240, 330, 840, 400), radius=18, fill=(255, 255, 255), outline=INK, width=6)
        fw = 580 * (1 - k)
        if fw > 6:
            d.rounded_rectangle((250, 340, 250 + fw, 390), radius=12, fill=PINK if k < 0.6 else RED)
        fl.txt(d, 540, 290, "LOVE METER", 54, PINK)
    elif t < 5.2:
        pod(C(cB), t, dict(), dict(reader, mouth=mB))
    elif t < 7.5:  # holding hands
        c = C((540, 1000, 1.2), sx=-(t - 5.2) * 30)
        street(c, t)
        person(c, 440, 1060, BF, 1, "hold", "flat" if t > 6.24 else 0.0, "half" if t > 6.24 else "open", "sad", t, sit=False, walk=True)
        person(c, 640, 1060, GF, -1, "hold", "smile", "happy", "flat", t + 0.3, sit=False, walk=True)
        if t > 6.6:
            hx, hy = c.p(540, 1060)
            hx, hy = hx / SS, hy / SS
            for k in range(6):
                d.line([(hx, hy - 120 + k * 45), (hx, hy - 95 + k * 45)], fill=YEL, width=10)
            fl.txt(d, hx, hy - 190, "CUT HERE", 70 * fl.pop(t, 6.6), YEL)
    elif t < 12.1:  # laugh 1
        sx, sy = shk(8.0)
        pod(C(wide, sx=sx, sy=sy), t, dict(LAUGH, ang=fall(8.0), kick=1.0 if t > 8.5 else 0, nomic=t > 8.0),
            dict(LAUGH, pose="kneeslap", lean=-10))
        if 8.3 < t < 9.0:
            fl.txt(d, 300, 640, "CRASH!", 140 * fl.pop(t, 8.3), YEL)
        fl.has(d, t, 800, 560, 0.4)
    elif t < 15.3:  # the problem
        if t < 13.0:
            pod(C(cA), t, dict(reader, mouth=mA), dict())
        else:
            c = C((540, 1000, 1.25))
            room(c)
            person(c, 700, 1090, GF, -1, "cross", 0.5 * abs(math.sin(t * 12)) if t < 14.7 else "flat", "open", "angry", t, sit=False)
            person(c, 360, 1090, BF, 1, "point" if t > 14.2 else "rest", "flat", "half", "flat", t, sit=False)
            fl.bubble(d, 640, 420, 600, 130, 640, 600, ["WHAT'S THE PROBLEM?"], 48)
            if t > 14.22:
                fl.txt(d, 330, 600, "THE PROBLEM", 76 * fl.pop(t, 14.22), YEL)
                d.line([(420, 650), (690, 820)], fill=YEL, width=14)
                d.polygon([(700, 790), (720, 850), (660, 840)], fill=YEL)
    elif t < 17.3:  # laugh 2: B spins
        pod(C(wide), t, dict(LAUGH, pose="facepalm"), dict(LAUGH, spin=True, nomic=True))
        fl.has(d, t, 320, 560)
    elif t < 21.3:  # battery
        if t < 18.0:
            pod(C(cB), t, dict(), dict(reader, mouth=mB))
        elif t < 20.2:
            c = C((540, 980, 1.2))
            beach(c)
            sway = 12 * math.sin(t * 2)
            c.arc(540, 900 + sway, 380, 150, 10, 170, width=7, fill=(170, 130, 90))
            rc = RotCam(c, 540, 1000 + sway, -math.pi / 2 + 0.3)
            person(rc, 540, 1000 + sway, BF, 1, "handshead", "smile", "closed", "flat", t)
            for k in range(3):
                a = ((t - 18) * 0.6 + k / 3) % 1
                fl.txt(d, 620 + 120 * a, 640 - 300 * a, "~", 70, YEL)
        else:
            c = C((540, 900, 1.6))
            bedroom(c)
            person(c, 540, 1090, GF, 1, "phone", "O", "huge", "up", t, sit=False, prop="phone")
            fl.txt(d, W / 2, 360, "0%", 160 * fl.pop(t, 20.4), RED)
            fl.txt(d, W / 2, 490, "BATTERY DEAD", 90 * fl.pop(t, 20.86), RED)
    elif t < 23.1:  # laugh 3: B falls sideways
        sx, sy = shk(21.5)
        pod(C(wide, sx=sx, sy=sy), t, dict(LAUGH, pose="facepalm", lean=12), dict(LAUGH, ang=fall(21.5), kick=1.0 if t > 22 else 0, nomic=True))
    elif t < 28.8:  # silent treatment
        if t < 23.9:
            pod(C(cA), t, dict(reader, mouth=mA), dict())
        else:
            k = ease_io((t - 26.6) / 0.8)
            c = C((lerp(420, 560, k), 1000, lerp(1.4, 1.2, k)))
            garden(c)
            c.rect(260, 1000, 460, 1030, (190, 140, 100), width=5)
            person(c, 360, 1000, BF, 1, "sip", "smile", "closed", "flat", t, prop="cup")
            if t > 26.7:
                person(c, 760, 1090, GF, -1, "cross", "zip", "half", "angry", t, sit=False)
                fl.txt(d, 760, 560, "...", 110 * fl.pop(t, 26.8))
            if t > 27.6:
                fl.txt(d, W / 2, 330, "SILENT TREATMENT", 92 * fl.pop(t, 27.6), PINK)
    elif t < 31.3:  # laugh 4: both on the floor
        pod(C(wide), t, dict(floor=True), dict(floor=True))
        fl.hahaha(d, t, 420)
    elif t < 35.6:  # talk to me
        if t < 32.0:
            pod(C(cB), t, dict(), dict(reader, mouth=mB))
        else:
            c = C((540, 1000, 1.25))
            room(c)
            done = t > 34.84
            person(c, 700, 1090, GF, -1, "pray", "frown", "open", "sad", t, sit=False)
            person(c, 360, 1090, BF, 1, "thumb" if done else "rest", "grin" if done else "flat", "half", "flat", t, sit=False,
                   prop="thumb" if done else None)
            fl.bubble(d, 700, 430, 420, 120, 720, 640, ["TALK TO ME..."], 52, (200, 60, 110))
            d.rounded_rectangle((60, 520, 130, 900), radius=14, fill=(255, 255, 255), outline=INK, width=6)
            lvl = 0.12 if t > 33.68 else 0.65
            d.rounded_rectangle((70, 890 - 360 * lvl, 120, 890), radius=10, fill=GREEN if lvl > 0.3 else RED)
            fl.txt(d, 100, 480, "EFFORT", 40, YEL)
            if done:
                fl.bubble(d, 340, 560, 330, 120, 360, 700, ["IT'S FINE."], 56)
    elif t < 37.4:  # OHHHH
        pod(C(cA, 0.1 * kick(t, 35.8, 0.2)), t, dict(pose="handshead", mouth=mA if mA else "O", eyes="huge", brows="up"), dict(LAUGH))
        fl.txt(d, W / 2, 360, "OHHHH!", 150 * fl.pop(t, 35.8), YEL)
    elif t < 42.7:  # call you / battery
        if t < 39.6:
            c = C((540, 1000, 1.2))
            porch(c)
            x = lerp(560, 300, ease_io((t - 37.4) / 2.2))
            person(c, x, 1090, GF, -1, "wave", 0.5 * abs(math.sin(t * 12)), "happy", "flat", t, sit=False, walk=True, prop=None)
            person(c, 700, 1090, BF, -1, "wave", "smile", "half", "flat", t, sit=False)
            fl.bubble(d, 420, 380, 680, 180, 330, 600, ["I'LL CALL YOU WHEN", "I GET HOME!"], 50, (200, 60, 110))
        else:
            c = C((520, 900, 1.6))
            room(c)
            person(c, 480, 1100, BF, 1, "hold", "grin", "half", "skeptic", t, sit=False)
            px, py = c.p(620, 880)
            px, py = px / SS, py / SS
            d.rounded_rectangle((px - 50, py - 90, px + 50, py + 90), radius=14, fill=(60, 60, 70), outline=INK, width=6)
            fl.txt(d, px, py, "OFF" if t > 40.1 else "ON", 46, RED if t > 40.1 else GREEN)
            if t > 41.0:
                k = ease_out((t - 41.0) / 0.9)
                bx, by = px + 500 * k, py - 600 * k + 800 * k * k
                d.rounded_rectangle((bx - 28, by - 46, bx + 28, by + 46), radius=6, fill=(250, 210, 80), outline=INK, width=5)
                fl.txt(d, W / 2, 330 + 40, "BATTERY: GONE", 90 * fl.pop(t, 41.0), YEL)
    elif t < 45.5:  # laugh 5: B tips back, A points
        sx, sy = shk(43.4)
        pod(C(wide, sx=sx, sy=sy), t, dict(LAUGH, pose="point", lean=10), dict(LAUGH, mouth=mB if mB else "laugh", ang=fall(43.4),
                                                                           kick=1.0 if t > 43.9 else 0, nomic=True))
        if 43.7 < t < 44.4:
            fl.txt(d, 780, 640, "CRASH!", 140 * fl.pop(t, 43.7), YEL)
    elif t < 47.4:  # smile takes yours
        if t < 46.0:
            pod(C(cB), t, dict(), dict(reader, mouth=mB))
        else:
            c = C((540, 1000, 1.3))
            park(c)
            k = ease_io((t - 46.1) / 0.8)
            person(c, 360, 1090, BF, 1, "rest", "smile" if k < 0.15 else "frown", "open" if k < 0.6 else "half", "flat" if k < 0.6 else "sad", t, sit=False)
            person(c, 720, 1090, GF, -1, "rest", "grin" if k > 0.9 else "smile", "happy", "flat", t, sit=False)
            a0, a1 = c.p(380, 870), c.p(700, 870)
            sx, sy = lerp(a0[0], a1[0], k) / SS, lerp(a0[1], a1[1], k) / SS - 220 * math.sin(math.pi * k)
            if 0.1 < k < 0.95:
                d.arc((sx - 60, sy - 70, sx + 60, sy + 30), 20, 160, fill=YEL, width=14)
            if t > 46.6:
                fl.txt(d, 300, 420, "-1 SMILE", 70 * fl.pop(t, 46.6), RED)
                fl.txt(d, 780, 420, "+1 SMILE", 70 * fl.pop(t, 46.8), GREEN)
    elif t < 50.0:  # laugh 6
        pod(C(wide, 0.03 * math.sin(t * 3)), t, dict(floor=True), dict(floor=True))
        fl.hahaha(d, t, 420)
    elif t < 53.4:  # kisses / dog
        if t < 50.6:
            pod(C(cA), t, dict(reader, mouth=mA), dict())
        else:
            c = C((520, 960, 1.4))
            room(c)
            swap = t > 51.75
            person(c, 400, 1090, BF, 1, "facepalm" if swap else "rest", "frown" if swap else "smile", "closed" if not swap else "half",
                   "sad" if swap else "flat", t, sit=False)
            if not swap:
                person(c, 560, 1090, GF, -1, "rest", "O", "closed", "flat", t, sit=False, lean=14)
                for k in range(3):
                    a = ((t - 50.6) * 0.8 + k / 3) % 1
                    drawn_heart(c, 480 + 40 * math.sin(a * 6 + k), 760 - a * 250, 26, PINK)
            else:
                dog(c, 600, 980 + 10 * math.sin(t * 12), t, f=-1)
                fl.txt(d, W / 2, 360, "SLURP!", (130 + 10 * math.sin(t * 20)) * fl.pop(t, 51.8), PINK)
    elif t < 56.0:  # WOW / BRO
        pod(C(cB, 0.12 * kick(t, 54.54, 0.2) + 0.12 * kick(t, 55.4, 0.2)), t, dict(LAUGH),
            dict(pose="handshead" if t < 55.4 else "point", mouth=mB if mB else "O", eyes="huge", brows="up", lean=-6))
        if t > 54.54:
            fl.txt(d, W / 2, 360, "WOW!" if t < 55.4 else "BRO!", 160 * fl.pop(t, 54.54 if t < 55.4 else 55.4), YEL)
    elif t < 60.6:  # coming / cried
        if t < 56.9:
            pod(C(cB), t, dict(), dict(reader, mouth=mB))
        else:
            c = C((540, 940, 1.45))
            bedroom(c)
            cry = t > 58.9
            person(c, 540, 1110, BF, 1, "cry" if cry else "phone", "frown" if cry else "O", "cry" if cry else "huge", "sad", t,
                   prop=None if cry else "phone")
            if not cry and t > 58.2:
                fl.bubble(d, W / 2, 400, 680, 180, 560, 600, ["I'M COMING", "THIS WEEKEND <3"], 58, (200, 60, 110))
            if cry:
                c.ellipse(540, 1270, 120 + 260 * ease_out((t - 58.9) / 1.2), 24, (150, 200, 255), width=4)
                fl.txt(d, W / 2, 360, "I CRIED", 150 * fl.pop(t, 59.6), BLUE)
    else:
        pod(C(wide), t, dict(floor=True), dict(floor=True))
        fl.hahaha(d, t, 420)
    st.pov_title(img, "POV: When her smile", "takes away yours")
    return img.resize((W, H), Image.LANCZOS), layer


CAPS = [[w for w, _ in c] for c in chalk.CAPTIONS]


def caption(img, t):
    for ci, words in enumerate(chalk.CAPTIONS):
        start = words[0][1]
        nxt = chalk.CAPTIONS[ci + 1][0][1] if ci + 1 < len(chalk.CAPTIONS) else 99
        if start <= t < min(nxt, words[-1][1] + 1.0):
            active = max(i for i, (_, s) in enumerate(words) if t >= s)
            big = Image.new("RGBA", (W * SS, 140 * SS), (0, 0, 0, 0))
            dd = ImageDraw.Draw(big)
            f = ImageFont.truetype(str(st.CAP_FONT), 88 * SS)
            sp = f.getlength(" ")
            ws = CAPS[ci]
            tw = sum(f.getlength(w) for w in ws) + sp * (len(ws) - 1)
            scale = min(1.0, (W - 60) * SS / tw)
            f = ImageFont.truetype(str(st.CAP_FONT), int(88 * SS * scale))
            sp = f.getlength(" ")
            tw = sum(f.getlength(w) for w in ws) + sp * (len(ws) - 1)
            x = (W * SS - tw) / 2
            for i, w in enumerate(ws):
                ww = f.getlength(w)
                if i == active:
                    dd.rounded_rectangle((x - 10 * SS, 14 * SS, x + ww + 10 * SS, 14 * SS + f.size * 1.08), radius=10 * SS, fill=(200, 20, 50))
                dd.text((x, 16 * SS), w, font=f, fill=WHITE, stroke_width=5 * SS, stroke_fill=INK)
                x += ww + sp
            small = big.resize((W, 140), Image.LANCZOS)
            img.alpha_composite(small, (0, 1520))
            return


def main(audio_path, out):
    a, sr = rf.load_audio(audio_path)
    rf.END = END
    track = rf.talk_track(a, sr)
    tmp = out.with_suffix(".video.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "animation",
         "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    n = int(round(END * FPS))
    for di in range((n + 1) // 2):  # drawn on twos, like the reference
        i = 2 * di
        t = i / FPS
        base, layer = frame(t, max(track[i], track[min(i + 1, len(track) - 1)]))
        base = base.convert("RGBA")
        base.alpha_composite(layer)
        for k in range(2 if i + 1 < n else 1):
            tt = (i + k) / FPS
            comp = base.copy()
            caption(comp, tt)
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
