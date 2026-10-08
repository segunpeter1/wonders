"""Tweets podcast short, hand-drawn stick animation.

Hosts read tweets (each re-enacted as a little stick story) and then lose it:
tipping off their chairs, rolling on the floor, spinning in their chairs.
Audio is the original clip, unchanged; profanity is censored in the captions only.

Usage: python3 render.py <original_audio.wav> <output.mp4>
"""
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("rf", HERE.parent / "rich-friend-podcast" / "render.py")
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)

stick, st = rf.stick, rf.st
Pen, clamp, lerp, ease_io, ease_out, kick = rf.Pen, rf.clamp, rf.lerp, rf.ease_io, rf.ease_out, rf.kick
person, text, ha, INK, SS = rf.person, rf.text, rf.ha, rf.INK, rf.SS

W, H, FPS = 1080, 1920, 30
END = 54.6
FLOOR = 1345

HOST_A = dict(skin=(110, 72, 50), shirt=(35, 35, 40), pants=(70, 70, 80), hair="fade", beard=(30, 22, 18))
HOST_B = dict(skin=(196, 150, 112), shirt=(250, 250, 248), pants=(60, 60, 70), hair="fade", beard=(40, 30, 24))
PROD = dict(skin=(150, 100, 70), shirt=(60, 70, 55), pants=(50, 50, 60), hair="short", beard=None)
DAD = dict(skin=(140, 95, 65), shirt=(230, 120, 60), pants=(60, 70, 110), hair="fade", beard=(35, 25, 20))
KID = dict(skin=(140, 95, 65), shirt=(250, 200, 220), pants=(250, 200, 220), hair="bald", beard=None, dress=True)
GUY = dict(skin=(150, 100, 66), shirt=(90, 170, 110), pants=(60, 70, 110), hair="short", beard=None)
GIRL = dict(skin=(165, 110, 75), shirt=(240, 120, 170), pants=(240, 120, 170), hair="puff", beard=None, dress=True)
LIGHT = dict(skin=(232, 196, 160), shirt=(250, 250, 250), pants=(90, 110, 160), hair="short", beard=(150, 110, 80))
BOSS = dict(skin=(200, 150, 110), shirt=(70, 80, 110), pants=(50, 50, 60), hair="bald", beard=None)


class RotPen(Pen):
    """A Pen that rotates everything it draws around a world-space pivot (for falls and rolls)."""

    def __init__(self, base, px, py, ang, dx=0.0, dy=0.0):
        self.__dict__.update(base.__dict__)
        self.px, self.py, self.ca, self.sa, self.dx, self.dy = px, py, math.cos(ang), math.sin(ang), dx, dy

    def p(self, x, y):
        x, y = x - self.px, y - self.py
        x, y = x * self.ca - y * self.sa + self.px + self.dx, x * self.sa + y * self.ca + self.py + self.dy
        return Pen.p(self, x, y)


# ------------------------------------------------------------------ studio

SEATS = {"A": (360, 1, HOST_A), "B": (720, -1, HOST_B)}
S = 1.7
HIP = 1235


def studio_bg(pen, t):
    pen.rect(-900, -1000, 2000, FLOOR, fill=(170, 115, 75), outline=None)
    for x in range(-900, 2000, 46):  # wood slats
        pen.line([(x, 250), (x, FLOOR - 4)], width=5, fill=(140, 90, 58))
    pen.rect(-900, 250, -40, FLOOR, fill=(90, 160, 220), outline=None)
    pen.rect(1120, 250, 2000, FLOOR, fill=(90, 160, 220), outline=None)
    pen.rect(-900, FLOOR, 2000, 3200, fill=(95, 95, 105), outline=None)
    pen.line([(-900, FLOOR), (2000, FLOOR)], width=7)
    pen.rect(380, 560, 700, 600, fill=(120, 80, 50), width=6)  # shelf
    text(pen, 540, 520, "THE POD", 60, fill=(255, 120, 190), stroke=(120, 40, 90), sw=4)
    for k in range(3):
        pen.rect(420 + k * 70, 600, 455 + k * 70, 640, fill=[(230, 90, 80), (240, 200, 70), (90, 150, 220)][k], width=4)


def chair(pen, x, facing):
    pen.line([(x, HIP + 20), (x, FLOOR - 25)], width=10)
    pen.line([(x - 55, FLOOR - 12), (x + 55, FLOOR - 12)], width=9)
    for wx in (-55, 55):
        pen.ellipse(x + wx, FLOOR - 8, 9, 9, fill=(40, 40, 45), width=3)
    pen.rect(x - 60, HIP + 4, x + 70 * facing * 0 + 60, HIP + 26, fill=(40, 40, 45), width=6)
    bx = x - facing * 62
    pen.rect(bx - 14, HIP - 190, bx + 14, HIP + 10, fill=(40, 40, 45), width=6)


def mic_stand(pen, x, facing):
    mx, my = x + facing * 125, HIP - 205
    pen.line([(x + facing * 330, 330), (x + facing * 230, my - 160), (mx, my - 20)], width=8, fill=(40, 40, 45))
    pen.poly([(mx - 22, my - 36), (mx + 22, my - 36), (mx + 22, my + 36), (mx - 22, my + 36)], fill=(35, 35, 40), width=5)


def host(pen, who, t, st_):
    """st_: ang (fall, radians, + = backwards), dx, dy, pose, mouth, eyes, kick, spin, prop, chair_dx."""
    x, facing, look = SEATS[who]
    if st_.get("spin"):
        facing = facing if int(t * 7) % 2 == 0 else -facing
    ang = -facing * st_.get("ang", 0.0)
    piv = (x - facing * 70, FLOOR)
    if st_.get("fwd"):
        ang = facing * st_["fwd"]
        piv = (x + facing * 80, FLOOR)
    slide_in = facing * 260 * clamp(st_.get("ang", 0.0) / (math.pi / 2), 0, 1)
    rp = RotPen(pen, piv[0], piv[1], ang, st_.get("dx", 0) + slide_in, st_.get("dy", 0))
    cdx = st_.get("chair_dx", 0)
    if st_.get("chair", True):
        chair(RotPen(pen, piv[0], piv[1], ang if not cdx else 0, cdx, 0) if cdx else rp, x, facing)
    bob = st_.get("bob", 0) * math.sin(t * 19)
    person(rp, x, HIP + bob, S, look, facing=facing, pose=st_.get("pose", "rest"), mouth=st_.get("mouth", 0.0),
           eyes=st_.get("eyes", "open"), brows=st_.get("brows"), t=t, sitting=True, lean=st_.get("lean", 0),
           prop=st_.get("prop"), kick=st_.get("kick", 0.0))
    if st_.get("spin"):
        for k in range(3):
            pen.arc(x, HIP - 120, 150 + k * 22, 60 + k * 10, 200 + k * 20, 330 + k * 10, width=5, fill=(250, 250, 250))


def producer_shot(pen, t, st_):
    pen.rect(-900, -1000, 2000, 3200, fill=(60, 45, 40), outline=None)
    for x in range(-900, 2000, 46):
        pen.line([(x, -1000), (x, 1400)], width=5, fill=(80, 60, 50))
    lean = 30 * st_.get("down", 0.0)
    bob = 10 * math.sin(t * 20)
    person(pen, 540, 1300 + bob, 2.2, PROD, facing=1, pose=st_.get("pose", "facepalm"), mouth="laugh", eyes="laughcry", t=t,
           sitting=True, lean=lean)
    hx, hy = 540 + lean * 2.2, 1300 + bob - 95 * 2.2 - 40 * 2.2 * 0.95
    pen.poly([(hx - 90, hy - 40), (hx - 70, hy - 95), (hx + 50, hy - 100), (hx + 85, hy - 45)], fill=(55, 65, 50), width=6)
    pen.poly([(hx + 60, hy - 45), (hx + 160, hy - 40), (hx + 150, hy - 25), (hx + 60, hy - 28)], fill=(55, 65, 50), width=5)
    pen.ellipse(hx - 5, hy - 70, 18, 16, fill=(230, 80, 80), width=4)
    pen.rect(-200, 1340, 1300, 1460, fill=(150, 110, 80), width=8)  # desk
    pen.poly([(620, 1180), (900, 1180), (930, 1340), (650, 1340)], fill=(190, 195, 205), width=6)  # laptop
    if st_.get("bang"):
        k = abs(math.sin(t * 16))
        text(pen, 230, 1250 - 40 * k, "BANG", 46, fill=(255, 210, 60), sw=6)


def studio(pen, t, a, b):
    studio_bg(pen, t)
    for who, st_ in (("A", a), ("B", b)):
        host(pen, who, t, st_)
    for who, st_ in (("A", a), ("B", b)):
        if st_.get("ang", 0) < 0.3 and not st_.get("fwd") and not st_.get("nomic"):
            mic_stand(pen, SEATS[who][0], SEATS[who][1])


# ------------------------------------------------------------------ stories


def room(pen, wall=(240, 228, 210), floor=(200, 175, 145)):
    pen.rect(-900, -1000, 2000, 1400, fill=wall, outline=None)
    for x in range(-900, 2000, 80):
        pen.line([(x, 300), (x, 1395)], width=3, fill=tuple(max(0, c - 14) for c in wall))
    pen.rect(-900, 1400, 2000, 3200, fill=floor, outline=None)
    pen.line([(-900, 1400), (2000, 1400)], width=7)


def story_hair(pen, t):
    room(pen, wall=(250, 235, 220))
    # baby mama photo: also hardly any hair
    pen.rect(640, 560, 860, 760, fill=(255, 250, 240), width=8)
    pen.ellipse(750, 650, 55, 55, fill=(150, 100, 70), width=5)
    pen.arc(750, 650, 55, 55, 250, 290, width=8, fill=(30, 22, 18))
    text(pen, 750, 735, "MAMA", 26, fill=(230, 90, 130), sw=3)
    person(pen, 300, 1290, 1.5, DAD, facing=1, pose="talk", t=t, eyes="half", brows="sad", mouth="flat")
    person(pen, 520, 1330, 0.95, KID, facing=-1, pose="rest", t=t, eyes="open", mouth=0.0)
    # three lonely hairs + ruler
    hx, hy = 520 + 0, 1330 - 95 * 0.95 - 40 * 0.95 * 2
    for k in (-1, 0, 1):
        pen.line([(hx + k * 8, hy + 4), (hx + k * 9 + 3 * math.sin(t * 5 + k), hy - 10)], width=3)
    pen.rect(560, hy - 120, 590, hy + 40, fill=(250, 220, 90), width=5)
    for k in range(8):
        pen.line([(560, hy - 110 + k * 20), (575, hy - 110 + k * 20)], width=3)
    pen.rect(120, 560, 300, 720, fill=(255, 255, 255), width=6)
    text(pen, 210, 610, "DAY", 34, fill=(230, 80, 80), sw=3)
    text(pen, 210, 670, "400", 52, fill=(40, 40, 40), stroke=(40, 40, 40), sw=1)


def story_chest(pen, t):
    room(pen, wall=(235, 225, 245))
    pen.poly([(140, 1080), (940, 1080), (940, 1240), (140, 1240)], fill=(200, 110, 110), width=8)
    pen.poly([(100, 1230), (980, 1230), (980, 1400), (100, 1400)], fill=(220, 130, 130), width=8)
    person(pen, 620, 1235, 1.5, GIRL, facing=-1, pose="rest", sitting=True, t=t, eyes="open")
    rp = RotPen(pen, 330, 1235, 0.5)
    person(rp, 330, 1235, 1.5, GUY, facing=1, pose="rest", sitting=True, t=t, eyes="closed", mouth=0.0)
    if t >= 26.04:  # x-ray bubble: tiny guys laughing inside
        k = ease_out((t - 26.04) / 0.25)
        cx, cy = 560, 640
        rf.bubble(pen, cx, cy, 250 * k, 190 * k, 600, 1000)
        if k > 0.9:
            for i, x in enumerate((470, 560, 650)):
                person(pen, x, 700, 0.55, dict(skin=(120, 80, 55), shirt=[(90, 150, 220), (240, 190, 60), (220, 90, 80)][i],
                                               pants=(50, 50, 60), hair="short", beard=None),
                       facing=-1 if i else 1, pose="belly" if i != 1 else "point", mouth="laugh", eyes="happy", t=t + i)
            ha(pen, 520, 520, t, 0)
            ha(pen, 620, 540, t, 0.5)


def story_wall(pen, t):
    room(pen, wall=(200, 205, 225), floor=(160, 160, 170))
    pen.poly([(330, -300), (750, -300), (900, 1400), (180, 1400)], fill=(225, 228, 240), outline=None)  # spotlight
    slide = ease_io((t - 39.6) / 1.0)
    hip = lerp(1225, 1340, slide)
    person(pen, 560, hip, 1.55, LIGHT, facing=1, pose="cry" if slide < 0.5 else "rest", sitting=slide > 0.5,
           eyes="cry", brows="sad", mouth="frown", t=t, lean=-6)
    for k in range(10):  # rain of sadness
        ry = 250 + ((t * 500 + k * 90) % 700)
        pen.line([(200 + k * 75, ry), (192 + k * 75, ry + 28)], width=4, fill=(120, 150, 210))


def story_job(pen, t):
    room(pen, wall=(235, 240, 245), floor=(180, 180, 190))
    if t < 44.44:
        pen.rect(380, 1150, 760, 1210, fill=(150, 110, 80), width=7)
        person(pen, 300, 1290, 1.45, GUY, facing=1, pose="point", t=t, eyes="happy", mouth="laugh")
        person(pen, 820, 1290, 1.45, BOSS, facing=-1, pose="point", t=t, eyes="happy", mouth=0.0)
        if t > 44.06:
            k = ease_out((t - 44.06) / 0.15)
            text(pen, 560, 700, "HIRED!", 110 * (1.6 - 0.6 * k), fill=(90, 200, 100), sw=9)
    else:
        robber = dict(GUY, shirt=(245, 245, 245))
        person(pen, 470, 1290, 1.6, robber, facing=1, pose="leash", t=t, eyes="half", mouth=0.0)
        for k in range(4):  # stripes
            pen.line([(470 - 40, 1290 - 140 + k * 30), (470 + 40, 1290 - 140 + k * 30)], width=8, fill=(30, 30, 35))
        hx, hy = 470, 1290 - 95 * 1.6 - 40 * 1.6 * 0.95
        pen.rect(hx - 50, hy - 25, hx + 70, hy + 2, fill=(20, 20, 25), outline=None)  # eye mask
        rf.money_bag(pen, 640, 1260, 1.6)
        pen.rect(760, 560, 960, 760, fill=(255, 255, 255), width=7)
        pen.rect(760, 560, 960, 610, fill=(230, 80, 80), width=7)
        text(pen, 860, 690, "FRIDAY", 44, fill=(40, 40, 40), stroke=(40, 40, 40), sw=1)


def story_breath(pen, t):
    room(pen, wall=(245, 240, 220))
    person(pen, 330, 1290, 1.5, GUY, facing=1, pose="talk", t=t, eyes="open", mouth=0.6 * abs(math.sin(t * 13)))
    faint = ease_io((t - 49.6) / 0.6)
    rp = RotPen(pen, 830, 1390, faint * 1.2)
    person(rp, 760, 1290, 1.5, GIRL, facing=-1, pose="facepalm" if faint < 0.5 else "rest", t=t, eyes="huge", mouth="O")
    for k in range(6):  # stink clouds
        a = ((t - 48.24) * 0.8 + k / 6) % 1
        cx, cy = 430 + a * 300, 1000 - 60 * math.sin(a * 6 + k)
        pen.ellipse(cx, cy, 34 * (0.5 + a), 26 * (0.5 + a), fill=(150, 200, 90), width=5)
    pen.rect(560, 1240, 620, 1400, fill=(200, 120, 80), width=6)  # dying plant
    droop = ease_io((t - 48.8) / 1.0)
    pen.line([(590, 1240), (590 + 60 * droop, 1150 + 70 * droop)], width=8, fill=(110, 140, 60))
    if t > 50.4:
        k = ease_out((t - 50.4) / 0.2)
        text(pen, 330, 760, "3 DAYS", 80 * (0.6 + 0.4 * k), fill=(150, 200, 90), sw=8)
        for d in (-1, 1):  # cobwebs on his mouth
            pen.line([(330 + 50, 1065), (330 + 50 + d * 25, 1095)], width=3, fill=(240, 240, 240))


# ------------------------------------------------------------------ captions

rf.CAPTIONS[:] = [
    [("CHOOSE", 0.00), ("YOUR", 0.28), ("BABY", 0.44), ("MAMA", 0.60), ("WISELY.", 0.88)],
    [("MY", 1.42), ("DAUGHTER'S", 1.62), ("HAIR'S", 2.00), ("NOT", 2.16), ("GROWING.", 2.36)],
    [("NO!", 12.60)],
    [("THAT", 20.42), ("WAS", 20.82), ("TAKING", 21.02), ("ME", 21.34), ("OUT!", 21.66)],
    [("NO!", 22.70)],
    [("THAT", 23.02), ("WAS", 23.18), ("TAKING", 23.34), ("ME", 23.62), ("OUT!", 23.86)],
    [("LAST", 24.08), ("TIME", 24.34), ("I", 24.54), ("LAID", 24.68), ("ON", 24.76), ("A", 24.94), ("B*TCH'S", 25.06), ("CHEST,", 25.38)],
    [("I", 26.04), ("HEARD", 26.14), ("N*GGAS", 26.28), ("IN", 26.52), ("THERE", 26.66)],
    [("LAUGHING", 26.86), ("AT", 27.18), ("ME.", 27.50)],
    [("OH,", 34.04)],
    [("LIGHT", 37.14), ("SKINS", 37.42), ("CRY", 38.10)],
    [("WITH", 38.26), ("THEIR", 38.48), ("BACKS", 38.62), ("AGAINST", 38.84), ("THE", 39.18), ("WALL.", 39.36)],
    [("THEN", 39.58), ("SLIDE", 39.70), ("DOWN", 39.90), ("SLOWLY.", 40.16)],
    [("I", 43.08), ("GOT", 43.60), ("THE", 43.80), ("JOB.", 44.06)],
    [("I", 44.44), ("START", 44.82), ("ROBBING", 44.96), ("N*GGAS", 45.24), ("ON", 45.52), ("FRIDAY.", 45.80)],
    [("THIS", 48.24), ("BREATH", 48.76), ("SMELLS", 49.10), ("LIKE", 49.40)],
    [("HE", 49.58), ("HASN'T", 49.74), ("SAID", 49.92), ("NOTHING", 50.04)],
    [("IN", 50.30), ("THREE", 50.50), ("DAYS.", 50.66)],
]
rf._cap.clear()

# ------------------------------------------------------------------- timeline


def speaker(t):
    for a, b, who in ((0, 2.7, "A"), (12.5, 13.0, "B"), (20.4, 24.0, "B"), (24.0, 27.8, "A"), (33.9, 34.6, "B"),
                      (37.1, 40.7, "A"), (43.0, 46.4, "A"), (48.2, 51.2, "A")):
        if a <= t < b:
            return who
    return None


LAUGH = dict(mouth="laugh", eyes="laughcry", pose="belly", bob=7)


def frame(t, m, seed):
    img = Image.new("RGB", (W * SS, H * SS), (200, 160, 120))
    wide, cA, cB = (540, 1080, 1.22), (390, 1010, 1.8), (700, 1010, 1.8)

    def P(c, push=0.0, sx=0.0):
        return Pen(img, c[0], c[1], c[2] * (1 + push), sx=sx, seed=seed)

    mA = m if speaker(t) == "A" else 0.0
    mB = m if speaker(t) == "B" else 0.0
    fall = lambda t0, d=0.45: (math.pi / 2 * 1.02) * ease_out((t - t0) / d) if t >= t0 else 0.0  # noqa: E731
    shake = lambda t0: 30 * kick(t, t0 + 0.4, 0.2) * math.sin(t * 70)  # noqa: E731

    if t < 0.9:
        studio(P(cA), t, dict(pose="phone", prop="phone", mouth=mA, eyes="half"), dict())
    elif t < 2.75:
        story_hair(P((480, 1040, 1.45)), t)
    elif t < 6.0:  # laughing, leaning back further and further
        lean = ease_io((t - 2.75) / 3.2)
        studio(P(wide, 0.03 * math.sin(t * 3)), t, dict(LAUGH, prop="phone", lean=10 * lean),
               dict(LAUGH, pose="facepalm", ang=0.25 * lean + 0.05 * math.sin(t * 9)))
        ha(P(wide), 300, 740, t, 0)
    elif t < 9.0:  # B tips over backwards, chair and all
        pen = P(wide, sx=shake(6.2))
        studio(pen, t, dict(LAUGH, pose="point", prop=None, lean=18), dict(LAUGH, ang=fall(6.2), kick=1.0 if t > 6.7 else 0, nomic=True))
        if 6.55 < t < 7.1:
            text(pen, 860, 980, "CRASH!", 90, fill=(255, 210, 60), sw=9)
    elif t < 12.0:  # A slides off his chair onto the floor too
        pen = P(wide, sx=shake(9.3))
        studio(pen, t, dict(LAUGH, ang=fall(9.3), kick=1.0 if t > 9.8 else 0, nomic=True),
               dict(LAUGH, ang=fall(0), kick=1.0, nomic=True))
        ha(pen, 300, 900, t, 0)
        ha(pen, 780, 900, t, 0.5)
    elif t < 13.6:  # B on the floor: NO!
        pen = P((800, 1230, 1.6), 0.12 * kick(t, 12.6, 0.2))
        studio(pen, t, dict(LAUGH, ang=fall(0), kick=1.0, nomic=True), dict(LAUGH, ang=fall(0), mouth=mB if mB else "laugh",
                                                                            pose="point", kick=0.5, nomic=True))
    elif t < 16.0:  # producer can't breathe
        producer_shot(P((540, 1000, 1.0)), t, dict(pose="facepalm", down=ease_io((t - 14.6) / 0.5), bang=t > 15.0))
    elif t < 20.4:  # rolling on the floor
        pen = P(wide)
        studio_bg(pen, t)
        for who, ph in (("A", 0), ("B", 1.3)):
            x, facing, look = SEATS[who]
            roll = math.pi / 2 + 0.35 * math.sin(t * 6 + ph)
            rp = RotPen(pen, x - facing * 70, FLOOR, -facing * roll, facing * 260 + 25 * math.sin(t * 3 + ph), -20)
            person(rp, x, HIP, S, look, facing=facing, pose="belly", mouth="laugh", eyes="laughcry", t=t, sitting=True, kick=1.0)
        for who in ("A", "B"):  # empty chairs
            chair(pen, SEATS[who][0] + SEATS[who][1] * -10, SEATS[who][1])
        if int(t * 4) % 2:
            text(pen, 540, 820, "HAHAHAHA", 80, fill=(255, 210, 60), sw=9)
    elif t < 22.7:  # B still on the floor: that was taking me out
        pen = P((800, 1230, 1.6), 0.04)
        studio(pen, t, dict(LAUGH, ang=fall(0), kick=1.0, nomic=True), dict(ang=fall(0), mouth=mB if mB else "laugh", eyes="happy",
                                                                            pose="point", nomic=True))
    elif t < 24.05:  # NO! climbs back into the chair
        up = ease_io((t - 23.0) / 0.8)
        pen = P((760, 1150, 1.5), 0.12 * kick(t, 22.7, 0.15))
        studio(pen, t, dict(LAUGH, ang=fall(0)), dict(ang=(math.pi / 2) * (1 - up), mouth=mB if mB else "laugh", eyes="happy",
                                                      pose="handshead"))
    elif t < 27.75:
        story_chest(P((520, 1030, 1.4)), t)
    elif t < 30.2:
        pen = P(wide, 0.04 * math.sin(t * 2))
        studio(pen, t, dict(LAUGH, prop="phone"), dict(LAUGH, pose="facepalm", lean=-14))
        ha(pen, 300, 740, t, 0)
    elif t < 33.0:  # A falls forward on his face, pounding the floor
        pen = P(wide, sx=shake(30.3))
        fwd = (math.pi / 2 * 0.95) * ease_out((t - 30.3) / 0.5)
        studio(pen, t, dict(LAUGH, fwd=fwd, pose="belly", chair_dx=-120 * ease_out((t - 30.3) / 0.6), nomic=True),
               dict(LAUGH, pose="point", lean=-10))
        if t > 30.8:
            text(pen, 520, 1240, "BAM", 60 * (1 + 0.2 * math.sin(t * 25)), fill=(255, 210, 60), sw=7)
    elif t < 33.9:  # B's chair rolls away, he drops to the floor
        cd = 260 * ease_in_out(t, 33.0, 0.6)
        drop = ease_io((t - 33.2) / 0.3)
        pen = P(wide)
        studio(pen, t, dict(LAUGH, fwd=math.pi / 2 * 0.95, chair_dx=-120, nomic=True),
               dict(LAUGH, chair_dx=cd, dy=110 * drop, nomic=True))
    elif t < 37.1:  # Oh, ... both wheezing on the floor
        pen = P(wide if t < 35.5 else cB)
        studio(pen, t, dict(LAUGH, ang=fall(0), kick=0.6, nomic=True, chair=False),
               dict(LAUGH, dy=110, chair=False, mouth=mB if mB else "laugh", pose="cry" if t > 35.5 else "belly", nomic=True))
    elif t < 40.7:
        story_wall(P((540, 1040, 1.45)), t)
    elif t < 43.05:  # B spins in his chair
        pen = P(wide, 0.03 * math.sin(t * 3))
        studio(pen, t, dict(LAUGH, prop="phone", pose="phone"), dict(LAUGH, spin=True, nomic=True))
    elif t < 46.35:
        story_job(P((540, 1040, 1.4)), t)
    elif t < 48.2:  # A tips over backwards
        pen = P(wide, sx=shake(46.6))
        studio(pen, t, dict(LAUGH, ang=fall(46.6), kick=1.0 if t > 47.1 else 0, nomic=True), dict(LAUGH, pose="point", lean=-12))
    elif t < 51.15:
        story_breath(P((540, 1040, 1.4)), t)
    else:  # everybody down
        pen = P(wide, 0.03 * math.sin(t * 2.5))
        studio(pen, t, dict(LAUGH, ang=fall(0), kick=1.0, nomic=True), dict(LAUGH, ang=fall(51.3), kick=1.0 if t > 51.8 else 0, nomic=True))
        if int(t * 4) % 2:
            text(pen, 540, 820, "HAHAHAHA", 80, fill=(255, 210, 60), sw=9)
    return img.resize((W, H), Image.LANCZOS)


def ease_in_out(t, t0, d):
    return ease_io((t - t0) / d)


def main(audio_path, out):
    a, sr = rf.load_audio(audio_path)
    rf.END = END
    track = rf.talk_track(a, sr)
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
        f = np.asarray(frame(t, max(track[i], track[i + 1]), seed=di), np.float32)
        mx, my = maps[di % len(maps)]
        f = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * paper
        img = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
        for k in range(2 if i + 1 < n else 1):
            tt = (i + k) / FPS
            cap = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            rf.draw_caption(cap, tt)
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
