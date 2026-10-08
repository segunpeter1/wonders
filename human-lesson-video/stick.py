"""Stick-figure version of The Human Lesson, drawn and animated entirely in code.

Every frame is drawn from scratch: characters have jointed limbs, a run cycle,
lip-sync from the voice loudness, and the camera moves between shots.
Audio is the original voiceover only.

Usage: python3 stick.py <voiceover.wav> <output.mp4>
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import importlib.util

import render as hl  # captions + mouth track from the 3D version

_st = importlib.util.spec_from_file_location("st_render", Path(__file__).parent.parent / "smooth-talker-video" / "render.py")
st = importlib.util.module_from_spec(_st)
_st.loader.exec_module(st)
import cv2

W, H, FPS, END = hl.W, hl.H, hl.FPS, hl.END
SS = 2  # supersampling for smooth lines

INK = (58, 34, 26)
SKIN = (196, 132, 82)
WALL = (236, 230, 216)
FLOOR_C = (214, 198, 176)
FLOOR = 1250


def clamp(x, a, b):
    return max(a, min(b, x))


def lerp(a, b, t):
    return a + (b - a) * t


def ease_io(t):
    t = clamp(t, 0, 1)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp(t, 0, 1)
    return 1 - (1 - t) ** 3


def kick(t, t0, d):
    return 0.0 if t < t0 else math.exp(-(t - t0) / d)


# ------------------------------------------------------------------- pen


class Pen:
    """Draws in world coordinates through a camera, with a sketchy hand-drawn hand.

    Every stroke is drawn twice with a little jitter, fills are slightly off
    the outline like marker colouring, and circles are wobbly. The jitter is
    seeded per drawing so the lines "boil" from drawing to drawing.
    """

    def __init__(self, img, cx, cy, z, sx=0.0, sy=0.0, seed=0):
        self.d = ImageDraw.Draw(img)
        self.cx, self.cy, self.z = cx - sx / z, cy - sy / z, z
        self.rng = np.random.default_rng(seed)

    def p(self, x, y):
        return ((x - self.cx) * self.z + W / 2) * SS, ((y - self.cy) * self.z + H / 2) * SS

    def w(self, v):
        return max(1, int(v * self.z * SS))

    def j(self, a=2.0):
        return self.rng.uniform(-a, a) * SS

    def _stroke(self, q, fill, width, closed=False):
        for k, (amt, wf) in enumerate(((1.6, 1.0), (2.8, 0.45))):
            qq = [(x + self.j(amt), y + self.j(amt)) for x, y in q]
            if closed:
                qq.append(qq[0])
            wd = max(1, int(width * wf))
            self.d.line(qq, fill=fill, width=wd, joint="curve")
            if k == 0 and not closed:
                r = wd / 2
                for x, y in (qq[0], qq[-1]):
                    self.d.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def line(self, pts, width=9, fill=INK):
        q = [self.p(*pt) for pt in pts]
        if len(q) == 2:  # add a midpoint so straight lines bend a little
            (x0, y0), (x1, y1) = q
            q = [q[0], ((x0 + x1) / 2, (y0 + y1) / 2), q[1]]
        self._stroke(q, fill, self.w(width))

    def _loop(self, x, y, rx, ry, n=28):
        ph = self.rng.uniform(0, 6.28)
        pts = []
        for i in range(n):
            a = 2 * math.pi * i / n
            k = 1 + 0.025 * math.sin(3 * a + ph)
            pts.append(self.p(x + rx * k * math.cos(a), y + ry * k * math.sin(a)))
        return pts

    def ellipse(self, x, y, rx, ry, fill=None, outline=INK, width=8):
        q = self._loop(x, y, rx, ry)
        if fill is not None:
            ox, oy = self.j(3), self.j(3)
            self.d.polygon([(a + ox, b + oy) for a, b in q], fill=fill)
        if outline:
            self._stroke(q, outline, self.w(width), closed=True)

    def poly(self, pts, fill, outline=INK, width=8):
        q = [self.p(*pt) for pt in pts]
        if fill is not None:
            ox, oy = self.j(3.5), self.j(3.5)
            self.d.polygon([(a + ox, b + oy) for a, b in q], fill=fill)
        if outline:
            self._stroke(q, outline, self.w(width), closed=True)

    def arc(self, x, y, rx, ry, a0, a1, width=7, fill=INK):
        n = max(6, int(abs(a1 - a0) / 12))
        q = [self.p(x + rx * math.cos(math.radians(a)), y + ry * math.sin(math.radians(a)))
             for a in np.linspace(a0, a1, n)]
        self._stroke(q, fill, self.w(width))

    def hatch(self, x0, y0, x1, y1, step=22, width=3, fill=(120, 95, 80)):
        """Diagonal pencil hatching inside a box."""
        x = x0
        while x < x1 + (y1 - y0):
            a = (max(x0, x - (y1 - y0)), min(y1, y0 + (x - x0)))
            b = (min(x, x1), y0 + max(0, x - x1))
            self._stroke([self.p(*a), self.p(*b)], fill, self.w(width))
            x += step

    def rect(self, x0, y0, x1, y1, fill, outline=INK, width=8):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill, outline, width)


# --------------------------------------------------------------- figures


def limb(x, y, length, ang, facing):
    a = math.radians(ang)
    return x + math.sin(a) * length * facing, y + math.cos(a) * length


def head(pen, x, y, r, facing, eyes="open", mouth=0.0, brows=None, hair="kid", look=0.0):
    pen.ellipse(x, y, r, r, fill=SKIN, width=8)
    if hair == "kid":
        for i in range(7):
            a = math.radians(-160 + i * 23)
            pen.ellipse(x + math.cos(a) * r * 0.82, y + math.sin(a) * r * 0.82, r * 0.3, r * 0.3, fill=(40, 26, 20), outline=None)
    elif hair == "dad":
        pen.arc(x, y, r, r, 190, 350, width=r * 0.35, fill=(40, 26, 20))
        pen.arc(x, y + 2, r * 0.95, r * 0.95, 20, 160, width=r * 0.22, fill=(40, 26, 20))  # beard
    elif hair == "mom":
        pen.ellipse(x - facing * r * 0.25, y - r * 0.55, r * 1.05, r * 0.7, fill=(150, 80, 190), width=7)
    ex = x + facing * r * (0.28 + look)
    for k, dx in enumerate((-0.32, 0.32)):
        cx, cy = ex + dx * r, y - r * 0.12
        if eyes == "closed":
            pen.arc(cx, cy + r * 0.05, r * 0.16, r * 0.12, 200, 340, width=5)
        elif eyes == "sleep":
            pen.line([(cx - r * 0.14, cy), (cx + r * 0.14, cy)], width=5)
        elif eyes == "huge":
            pen.ellipse(cx, cy - r * 0.05, r * 0.24, r * 0.3, fill=(255, 255, 255), width=5)
            pen.ellipse(cx + facing * r * 0.04, cy - r * 0.05, r * 0.05, r * 0.05, fill=INK, outline=None)
        else:
            pen.ellipse(cx, cy, r * 0.13, r * 0.15, fill=(255, 255, 255), width=4)
            pen.ellipse(cx + facing * r * 0.05, cy, r * 0.07, r * 0.08, fill=INK, outline=None)
            pen.ellipse(cx + facing * r * 0.02, cy - r * 0.04, r * 0.025, r * 0.025, fill=(255, 255, 255), outline=None)
    if hair == "kid":  # rosy cheeks
        for dx in (-0.55, 0.55):
            pen.ellipse(ex + dx * r * 0.9, y + r * 0.22, r * 0.14, r * 0.08, fill=(230, 120, 110), outline=None)
    if brows == "smug":
        bx = ex
        pen.line([(bx - r * 0.45, y - r * 0.38), (bx - r * 0.15, y - r * 0.45)], width=5)
        pen.line([(bx + r * 0.15, y - r * 0.55), (bx + r * 0.45, y - r * 0.42)], width=5)
    elif brows == "shock":
        for dx in (-0.32, 0.32):
            pen.arc(ex + dx * r, y - r * 0.5, r * 0.18, r * 0.12, 200, 340, width=5)
    mx, my = x + facing * r * 0.3, y + r * 0.42
    if mouth == "O":
        pen.ellipse(mx, my, r * 0.16, r * 0.24, fill=(120, 30, 40), width=5)
    elif mouth < 0.12:
        pen.arc(mx, my - r * 0.12, r * 0.25, r * 0.16, 20, 160, width=6)
    else:
        h = r * (0.08 + 0.26 * mouth)
        pen.ellipse(mx, my, r * 0.24, h, fill=(120, 30, 40), width=5)
        pen.ellipse(mx, my + h * 0.45, r * 0.14, h * 0.4, fill=(230, 110, 120), outline=None)


def kid(pen, x, hip_y, t, pose, facing=-1, mouth=0.0, eyes="open", brows=None, s=1.0):
    """Jointed stick kid in orange pyjamas and green dino slippers."""
    L, T, r, ua, fa = 92 * s, 100 * s, 46 * s, 52 * s, 48 * s
    if pose == "run":
        ph = t * 2 * math.pi * 3.2
        hip_y -= abs(math.sin(ph)) * 12 * s
        legs = (40 * math.sin(ph), -40 * math.sin(ph))
        arms = ((-50 * math.sin(ph), 70 - 50 * math.sin(ph)), (50 * math.sin(ph), 70 + 50 * math.sin(ph)))
        lean = 14 * s
    else:
        legs = (14, -14)
        lean = 0
        arms = {"finger": ((-45, -20), (150, 168)), "point": ((-45, -20), (95, 92)),
                "salute": ((-45, -20), (135, 245)), "stand": ((-45, -20), (40, 20))}[pose]
    gy = FLOOR + 4
    for k in range(4):
        pen.line([(x - 50 * s + k * 8, gy + k * 4), (x + 50 * s - k * 8, gy + k * 4)], width=3, fill=(170, 150, 125))
    # legs + slippers
    for a in legs:
        fx, fy = limb(x, hip_y, L, a, facing)
        pen.line([(x, hip_y), (fx, fy)], width=11)
        pen.ellipse(fx + facing * 12 * s, fy + 2, 24 * s, 13 * s, fill=(70, 160, 70), width=6)
    nx, ny = x + facing * lean, hip_y - T
    # pyjama top
    pen.poly([(nx - 30 * s, ny + 8), (nx + 30 * s, ny + 8), (x + 36 * s, hip_y + 6), (x - 36 * s, hip_y + 6)],
             fill=(250, 130, 40), width=7)
    for k in range(3):  # dino spots
        pen.ellipse(x + (k - 1) * 18 * s, ny + 45 * s + (k % 2) * 25 * s, 7 * s, 6 * s, fill=(60, 150, 70), outline=None)
    sx, sy = nx, ny + 14 * s
    hands = []
    for k, (a1, a2) in enumerate(arms):
        sxx = sx + facing * (16 * s if k == 1 else -16 * s)
        ex, ey = limb(sxx, sy, ua, a1, facing)
        hx, hy = limb(ex, ey, fa, a2, facing)
        pen.line([(sxx, sy), (ex, ey), (hx, hy)], width=9)
        pen.ellipse(hx, hy, 9 * s, 9 * s, fill=SKIN, width=4)
        hands.append((hx, hy, a2))
    if pose in ("finger", "point"):
        hx, hy, a2 = hands[1]
        fx, fy = limb(hx, hy, 20 * s, a2, facing)
        pen.line([(hx, hy), (fx, fy)], width=6)
    head(pen, nx, ny - r * 0.9, r, facing, eyes=eyes, mouth=mouth, brows=brows)


# ------------------------------------------------------------------ room


def room(pen, t, door_open, parents_awake=False, show_parents=True, window_flap=0.0):
    pen.rect(-1500, -2000, 2600, FLOOR, fill=WALL, outline=None)
    pen.rect(-1500, FLOOR, 2600, 3500, fill=FLOOR_C, outline=None)
    pen.line([(-1500, FLOOR), (2600, FLOOR)], width=7)
    # wallpaper stripes, floorboards
    for x in range(-600, 1500, 70):
        pen.line([(x, 560), (x, FLOOR - 6)], width=3, fill=(224, 214, 196))
    for y in (FLOOR + 70, FLOOR + 170, FLOOR + 310):
        for x0 in range(-600, 1500, 320):
            pen.line([(x0 + (y % 90), y), (x0 + (y % 90) + 250, y)], width=3, fill=(185, 165, 140))
    # framed dinosaur picture
    pen.rect(250, 820, 400, 940, fill=(250, 246, 235), width=7)
    pen.line([(275, 915), (320, 870), (345, 885), (375, 860)], width=4, fill=(90, 150, 90))
    pen.ellipse(372, 855, 10, 8, fill=(90, 150, 90), width=3)
    # rug
    pen.ellipse(640, FLOOR + 55, 250, 38, fill=(205, 95, 85), width=6)
    for k in range(9):
        x = 430 + k * 50
        pen.line([(x, FLOOR + 48), (x + 25, FLOOR + 62), (x + 50, FLOOR + 48)], width=3, fill=(250, 220, 160))
    # toy blocks by the door
    for k, (bx, col) in enumerate(((905, (90, 140, 220)), (945, (240, 190, 60)))):
        pen.rect(bx, FLOOR - 36, bx + 36, FLOOR, fill=col, width=5)
    pen.rect(922, FLOOR - 70, 958, FLOOR - 36, fill=(220, 80, 80), width=5)
    # bedside lamp (left of the headboard)
    pen.rect(-60, 1100, 40, FLOOR, fill=(170, 120, 80), width=7)
    pen.line([(-10, 1100), (-10, 1035)], width=6)
    pen.poly([(-45, 1035), (25, 1035), (10, 985), (-30, 985)], fill=(255, 220, 130), width=6)
    for a in (-150, -120, -60, -30):
        r0, r1 = 70, 95
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        pen.line([(-10 + r0 * ca, 1010 + r0 * sa), (-10 + r1 * ca, 1010 + r1 * sa)], width=4, fill=(230, 170, 60))
    # window with moon
    pen.rect(560, 700, 690, 860, fill=(40, 52, 110), width=9)
    pen.ellipse(655, 738, 24, 24, fill=(250, 245, 215), outline=None)
    for sx, sy in ((585, 760), (605, 825), (665, 820)):
        pen.ellipse(sx, sy, 4, 4, fill=(255, 255, 230), outline=None)
    pen.line([(625, 700), (625, 860)], width=6)
    pen.line([(560, 780), (690, 780)], width=6)
    if window_flap > 0:  # curtain swish after the jump
        pen.poly([(560, 700), (560 + 40 * window_flap, 780), (560, 860)], fill=(120, 150, 210), width=5)
    # door + hallway light
    if door_open > 0:
        pen.rect(760, 870, 890, FLOOR, fill=(255, 225, 140), width=8)
        pen.poly([(760, FLOOR), (890, FLOOR), (860, FLOOR + 400), (480, FLOOR + 400)],
                 fill=(250, 220, 160), outline=None)
        dw = 130 * (1 - door_open)
        pen.rect(890 - max(dw, 16), 870, 890, FLOOR, fill=(150, 95, 60), width=8)
    else:
        pen.rect(760, 870, 890, FLOOR, fill=(150, 95, 60), width=8)
        pen.ellipse(778, 1070, 8, 8, fill=(230, 200, 90), width=4)
    # bed
    pen.rect(60, 980, 100, FLOOR, fill=(70, 80, 130), width=8)
    pen.rect(100, 1120, 500, 1175, fill=(160, 105, 70), width=8)
    pen.hatch(110, 1145, 490, 1170, step=26, fill=(120, 75, 50))
    pen.line([(480, 1175), (480, FLOOR)], width=10)
    pen.ellipse(195, 1100, 95, 28, fill=(255, 255, 255), width=7)
    if show_parents and not parents_awake == "up":
        eyes = "open" if parents_awake else "sleep"
        head(pen, 155, 1072, 40, 1, eyes=eyes, mouth=0.0, hair="dad", look=-0.1)
        head(pen, 245, 1068, 36, 1, eyes=eyes, mouth=0.0, hair="mom", look=-0.1)
    pen.poly([(205, 1085), (300, 1070), (400, 1078), (500, 1085), (500, 1130), (190, 1130)], fill=(245, 180, 205), width=8)


def zzz(pen, t):
    for k in range(3):
        a = (t * 0.8 + k / 3) % 1
        x, y = 210 + 50 * a + 15 * math.sin(a * 6), 1030 - 160 * a
        sz = 14 + 16 * a
        pen.line([(x - sz, y - sz), (x + sz, y - sz), (x - sz, y + sz), (x + sz, y + sz)], width=5)


def parents_up(pen, t):
    """Both parents bolt upright, frozen in shock."""
    for i, (x, hair, r) in enumerate(((170, "dad", 44), (300, "mom", 40))):
        jit = 3 * math.sin(t * 60 + i)
        pen.line([(x, 1100), (x + jit, 1000)], width=11)
        pen.poly([(x - 34, 1005), (x + 34, 1005), (x + 40, 1100), (x - 40, 1100)],
                 fill=(90, 120, 200) if i == 0 else (235, 140, 190), width=7)
        for side in (-1, 1):
            pen.line([(x + side * 30, 1015), (x + side * 48, 1060), (x + side * 18, 1088)], width=9)
            pen.ellipse(x + side * 18, 1088, 9, 9, fill=SKIN, width=4)
        head(pen, x + jit, 1000 - r * 0.95, r, 1, eyes="huge", mouth="O", brows="shock", hair=hair, look=-0.28)
        for a in (-140, -110, -70, -40):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            hy = 1000 - r * 0.95
            pen.line([(x + ca * r * 1.3, hy + sa * r * 1.3), (x + ca * r * 1.75, hy + sa * r * 1.75)], width=5)
        # sweat drop
        dy = (t * 120) % 60
        pen.ellipse(x + r * 0.9, 1000 - r * 1.2 + dy, 7, 11, fill=(150, 200, 255), width=3)
    pen.poly([(90, 1095), (500, 1095), (500, 1130), (90, 1130)], fill=(245, 180, 205), width=8)


def stars_burst(pen, x, y, t):
    if t < 0 or t > 0.6:
        return
    for k in range(8):
        a = k * math.pi / 4
        rr = 40 + 160 * ease_out(t / 0.6)
        sx, sy = x + math.cos(a) * rr, y + math.sin(a) * rr
        sz = 16 * (1 - t / 0.6)
        pen.poly([(sx, sy - sz), (sx + sz * 0.3, sy - sz * 0.3), (sx + sz, sy), (sx + sz * 0.3, sy + sz * 0.3),
                  (sx, sy + sz), (sx - sz * 0.3, sy + sz * 0.3), (sx - sz, sy), (sx - sz * 0.3, sy - sz * 0.3)],
                 fill=(255, 210, 60), width=3)


def speed_lines(pen, x, y, t):
    for k in range(4):
        yy = y - 120 + k * 50
        pen.line([(x + 60, yy), (x + 150 + 40 * math.sin(t * 30 + k), yy)], width=5)


# ----------------------------------------------------------------- frames


def frame(gt, m, seed=0):
    img = Image.new("RGB", (W * SS, H * SS), WALL)
    if gt < 1.0:  # bursting in
        z = lerp(1.9, 1.3, ease_io((gt - 0.45) / 0.55))
        pen = Pen(img, lerp(780, 480, ease_io((gt - 0.45) / 0.55)), lerp(1060, 1060, 0), z, seed=seed)
        room(pen, gt, door_open=ease_out(gt / 0.12))
        if gt < 0.15:
            zzz(pen, gt)
        run = clamp((gt - 0.55) / 0.45, 0, 1)
        if run == 0:
            kid(pen, 825, FLOOR - 106, gt, "finger", mouth=m, eyes="closed", s=1.15)
        else:
            kid(pen, lerp(825, 600, ease_io(run)), FLOOR - 106, gt, "run", mouth=0.0, s=1.15)
            speed_lines(pen, lerp(825, 600, ease_io(run)), FLOOR - 200, gt)
    elif gt < 2.5:  # the professor
        k = ease_io((gt - 1.0) / 1.5)
        z = lerp(2.1, 2.5, k) + 0.08 * kick(gt, 1.24, 0.12) + 0.08 * kick(gt, 2.32, 0.12)
        pen = Pen(img, 590, 990, z, seed=seed)
        room(pen, gt, 1.0)
        sway = 6 * math.sin(gt * 5)
        kid(pen, 600 + sway, FLOOR - 106, gt, "finger", mouth=m, eyes="closed", s=1.15)
    elif gt < 4.3:  # pointing at the parents
        punch = ease_out((gt - 3.30) / 0.12) if gt >= 3.30 else 0
        z = lerp(1.75, 1.85, (gt - 2.5) / 1.8) + 0.4 * punch
        cx = lerp(390, 470, punch)
        sx = 18 * kick(gt, 3.30, 0.15) * math.sin(gt * 70)
        pen = Pen(img, cx, 1040, z, sx=sx, seed=seed)
        room(pen, gt, 1.0, parents_awake=True)
        kid(pen, 590, FLOOR - 106, gt, "point", mouth=m, eyes="open", brows="smug", s=1.15)
    elif gt < 5.65:  # frozen parents
        k = ease_io((gt - 4.3) / 1.35)
        sx = 26 * kick(gt, 4.3, 0.14) * math.sin(gt * 80)
        pen = Pen(img, 240, lerp(1010, 990, k), lerp(2.4, 2.85, k), sx=sx, seed=seed)
        room(pen, gt, 1.0, show_parents=False)
        parents_up(pen, gt)
    else:  # until we meet again
        pen = Pen(img, 420, 1030, 1.6, seed=seed)
        room(pen, gt, 1.0, show_parents=False, window_flap=clamp((gt - 7.15) / 0.15, 0, 1))
        parents_up(pen, gt)
        go = clamp((gt - 6.95) / 0.2, 0, 1)
        if gt < 6.95:
            kid(pen, 690, FLOOR - 106, gt, "salute", mouth=m, eyes="open", s=1.15)
        elif gt < 7.15:
            x = lerp(690, 625, go)
            kid(pen, x, FLOOR - 106 - 340 * math.sin(math.pi * go * 0.5), gt, "run", facing=-1, mouth=0.0, s=1.15)
        stars_burst(pen, 625, 780, gt - 7.15)
    out = img.resize((W, H), Image.LANCZOS)
    return out


def main(voice_path, out):
    voice, sr = hl.read_wav(voice_path)
    track = hl.mouth_track(voice, sr)
    tmp = out.with_suffix(".video.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "animation",
         "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    n = int(round(END * FPS))
    maps, paper = st.make_boil_maps(n=3, amp=1.6), st.make_paper()
    # draw on twos: a new drawing every 2 frames, held for both
    for di in range((n + 1) // 2):
        i = 2 * di
        gt = i / FPS
        m = max(track[i], track[min(i + 1, len(track) - 1)])
        f = np.asarray(frame(gt, m, seed=di), np.float32)
        mx, my = maps[di % len(maps)]
        f = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * paper
        img = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
        for k in range(2 if i + 1 < n else 1):
            t = (i + k) / FPS
            cap = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            hl.draw_caption(cap, t)
            out = img.copy()
            out.alpha_composite(cap)
            a = np.asarray(out.convert("RGB"), np.float32)
            if t > END - 0.3:
                a *= clamp((END - t) / 0.3, 0, 1)
            ff.stdin.write(a.astype(np.uint8).tobytes())
    ff.stdin.close()
    ff.wait()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-i", str(voice_path), "-c:v", "copy",
                    "-af", f"afade=t=out:st={END - 0.3}:d=0.3", "-c:a", "aac", "-b:a", "192k", "-t", str(END),
                    "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink()
    print("done", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
