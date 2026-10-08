"""Crazy Girls podcast short, hand-drawn stick animation on a sunset rooftop set.

Reuses the stick characters, falls and captions from ../tweets-podcast.
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
from PIL import Image

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("tw", HERE.parent / "tweets-podcast" / "render.py")
tw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tw)
rf, st = tw.rf, tw.st
Pen, RotPen, clamp, lerp, ease_io, ease_out, kick = tw.Pen, tw.RotPen, tw.clamp, tw.lerp, tw.ease_io, tw.ease_out, tw.kick
person, text, ha, INK, SS = tw.person, tw.text, tw.ha, tw.INK, tw.SS
SEATS, S, HIP, FLOOR, LAUGH = tw.SEATS, tw.S, tw.HIP, tw.FLOOR, tw.LAUGH
HOST_A, HOST_B = tw.HOST_A, tw.HOST_B

W, H, FPS = 1080, 1920, 30
END = 44.93

GIRL = dict(skin=(170, 115, 80), shirt=(170, 90, 200), pants=(170, 90, 200), hair="puff", beard=None, dress=True)
EX = dict(skin=(150, 100, 66), shirt=(90, 170, 110), pants=(60, 70, 110), hair="short", beard=(40, 30, 24))
WIFE = dict(skin=(200, 150, 110), shirt=(230, 90, 90), pants=(230, 90, 90), hair="puff", beard=None, dress=True)
HUSBAND = dict(skin=(185, 135, 95), shirt=(80, 120, 200), pants=(50, 50, 60), hair="short", beard=None)
DOCTOR = dict(skin=(120, 80, 55), shirt=(245, 245, 250), pants=(245, 245, 250), hair="bald", beard=None, dress=True)
OLD_A = dict(HOST_A, beard=(205, 205, 210))
BALD_A = dict(HOST_A, hair="bald")

# ------------------------------------------------------------------ rooftop set

_rng = np.random.default_rng(3)
BUILDINGS = [(x, int(_rng.integers(620, 980)), int(_rng.integers(90, 170))) for x in range(-500, 1700, 150)]
WINDOWS = [(bx + 18 + 34 * c, top + 30 + 50 * r, _rng.random() < 0.55) for bx, top, bw in BUILDINGS
           for r in range(8) for c in range(max(1, (bw - 30) // 34))]


def rooftop_bg(pen, t):
    top, bot = (70, 45, 120), (255, 160, 100)
    for k in range(14):  # sunset sky bands
        y0 = -1000 + k * 160
        c = tuple(int(lerp(a, b, k / 13)) for a, b in zip(top, bot))
        pen.rect(-900, y0, 2000, y0 + 165, fill=c, outline=None)
    pen.ellipse(760, 930, 150, 150, fill=(255, 205, 120), outline=None)
    for bx, by, bw in BUILDINGS:
        pen.rect(bx, by, bx + bw, 1180, fill=(80, 55, 100), width=5)
    for wx, wy, lit in WINDOWS:
        if wy < 1160 and lit:
            pen.rect(wx, wy, wx + 14, wy + 20, fill=(255, 220, 120), outline=None)
    # string lights
    pts = [(x, 430 + 140 * (1 - ((x - 540) / 820) ** 2)) for x in range(-280, 1361, 80)]
    pen.line(pts, width=4, fill=(40, 30, 40))
    for k, (x, y) in enumerate(pts):
        on = (int(t * 3) + k) % 4 != 0
        pen.ellipse(x, y + 16, 11, 14, fill=[(255, 230, 140), (255, 160, 190), (150, 220, 255)][k % 3] if on else (150, 140, 120), width=3)
    # parapet + railing
    pen.rect(-900, 1180, 2000, FLOOR, fill=(190, 175, 165), width=7)
    pen.line([(-900, 1090), (2000, 1090)], width=8, fill=(50, 45, 55))
    for x in range(-880, 2000, 90):
        pen.line([(x, 1090), (x, 1180)], width=6, fill=(50, 45, 55))
    pen.rect(420, 1000, 660, 1070, fill=(60, 40, 70), width=6)
    text(pen, 540, 1036, "THE POD", 50, fill=(255, 120, 190), stroke=(120, 40, 90), sw=4)
    # wooden deck
    pen.rect(-900, FLOOR, 2000, 3200, fill=(150, 105, 70), outline=None)
    pen.line([(-900, FLOOR), (2000, FLOOR)], width=7)
    for k, y in enumerate((FLOOR + 60, FLOOR + 150, FLOOR + 280, FLOOR + 450)):
        pen.line([(-900, y), (2000, y)], width=4, fill=(120, 80, 52))
    for px in (-60, 1140):  # potted plants
        pen.rect(px - 60, 1220, px + 60, FLOOR, fill=(200, 120, 80), width=7)
        for k in range(6):
            a = math.radians(-160 + k * 28)
            pen.line([(px, 1220), (px + math.cos(a) * 150, 1220 + math.sin(a) * 200)], width=10, fill=(70, 150, 80))


tw.studio_bg = rooftop_bg  # swap the set used by tw.studio()


def head_of(who, lean=0.0):
    x, facing, _ = SEATS[who]
    nx = x + facing * lean * S
    return nx, HIP - 95 * S - 40 * S * 0.95, 40 * S, facing


def shades(pen, who, lean=0.0):
    hx, hy, r, f = head_of(who, lean)
    ex = hx + f * r * 0.3
    pen.rect(ex - r * 0.55, hy - r * 0.32, ex + r * 0.55, hy + 0.02 * r, fill=(20, 20, 25), outline=None)


def qmarks(pen, x, y, t):
    for k in range(3):
        if (t * 2 + k / 3) % 1 < 0.7:
            text(pen, x + (k - 1) * 70, y - 20 * k, "?", 70, fill=(255, 230, 90), sw=7)


# ------------------------------------------------------------------ story scenes


def phone_panel(pen, x, y, wdt, hgt):
    pen.rect(x - wdt / 2, y - hgt / 2, x + wdt / 2, y + hgt / 2, fill=(30, 30, 40), width=8)
    pen.rect(x - wdt / 2 + 18, y - hgt / 2 + 30, x + wdt / 2 - 18, y + hgt / 2 - 30, fill=(240, 244, 250), outline=None)


def story_unblock(pen, t):
    tw.room(pen, wall=(250, 225, 235))
    pen.rect(90, 560, 330, 760, fill=(255, 255, 255), width=7)
    pen.rect(90, 560, 330, 610, fill=(90, 150, 230), width=7)
    text(pen, 210, 650, "FATHER'S", 34, fill=(40, 40, 40), stroke=(40, 40, 40), sw=1)
    text(pen, 210, 705, "DAY", 46, fill=(230, 80, 80), sw=2)
    pen.poly([(60, 1180), (560, 1180), (560, 1400), (60, 1400)], fill=(150, 120, 200), width=8)
    smug = t > 5.5
    person(pen, 300, 1235, 1.45, GIRL, facing=1, pose="phone", prop="phone", sitting=True, t=t,
           eyes="half" if smug else "open", brows=None, mouth=0.0)
    phone_panel(pen, 770, 900, 400, 640)
    if t < 3.9:
        text(pen, 770, 760, "BLOCKED", 44, fill=(230, 70, 70), sw=4)
        if t > 3.14:
            pen.rect(620, 860, 920, 950, fill=(90, 200, 110), width=6)
            text(pen, 770, 905, "UNBLOCK", 40, fill=(255, 255, 255), sw=4)
    else:
        k = ease_out((t - 4.05) / 0.25)
        pen.rect(640, 700, 930, 700 + 150 * k, fill=(90, 150, 240), width=6)
        if k > 0.9:
            text(pen, 785, 745, "HAPPY", 38, fill=(255, 255, 255), sw=3)
            text(pen, 785, 800, "FATHER'S DAY!", 30, fill=(255, 255, 255), sw=3)
    if t > 5.66:
        k = ease_out((t - 5.66) / 0.18)
        text(pen, 770, 1060, "BLOCKED!", 100 * (1.8 - 0.8 * k), fill=(230, 50, 50), sw=9)


def story_calling(pen, t):
    tw.room(pen, wall=(215, 220, 235), floor=(170, 165, 170))
    yr = 1 + min(2, int((t - 6.3) / 0.55))
    pen.rect(700, 540, 950, 760, fill=(255, 255, 255), width=7)
    pen.rect(700, 540, 950, 590, fill=(230, 80, 80), width=7)
    text(pen, 825, 690, f"YEAR {yr}", 52, fill=(40, 40, 40), stroke=(40, 40, 40), sw=1)
    look = dict(EX, beard=(40, 30, 24) if yr < 3 else (120, 120, 125))
    person(pen, 380, 1290, 1.55, look, facing=1, pose="phone", prop="phone", t=t, eyes="cry" if yr == 3 else "huge",
           brows="sad", mouth="frown")
    for k in range(yr + 1):
        text(pen, 620, 860 + k * 80, "CALL FAILED", 34, fill=(230, 70, 70), sw=4)
    for k in range(4):  # sweat
        pen.ellipse(260 + k * 30, 940 + ((t * 300 + k * 50) % 120), 7, 11, fill=(120, 180, 255), width=3)


def story_nobaby(pen, t):
    tw.room(pen, wall=(235, 230, 250))
    pen.poly([(420, -300), (660, -300), (780, 1400), (300, 1400)], fill=(250, 248, 225), outline=None)
    pen.rect(380, 1180, 700, 1260, fill=(240, 200, 220), width=7)  # empty crib
    for k in range(8):
        pen.line([(395 + k * 42, 1080), (395 + k * 42, 1180)], width=6)
    pen.line([(380, 1080), (700, 1080)], width=8)
    pen.line([(390, 1260), (390, 1400)], width=8)
    pen.line([(690, 1260), (690, 1400)], width=8)
    tx = lerp(-100, 1150, (t - 8.2) / 0.8)  # tumbleweed
    for k in range(3):
        pen.ellipse(tx, 1350, 45 - k * 10, 45 - k * 10, fill=None, outline=(150, 110, 60), width=5)
    text(pen, 540, 820, "NO BABY", 90, fill=(230, 80, 120), sw=9)


def story_communicate(pen, t):
    pen.rect(-900, -1000, 2000, 3200, fill=(170, 215, 245), outline=None)
    pen.rect(-900, 1400, 2000, 3200, fill=(130, 200, 110), outline=None)
    person(pen, 350, 1300, 1.5, HOST_A, facing=1, pose="talk", t=t, mouth=0.6 * abs(math.sin(t * 13)), eyes="huge")
    # megaphone, paper planes, carrier pigeons
    pen.poly([(470, 1000), (600, 930), (600, 1080)], fill=(240, 90, 80), width=6)
    for k in range(3):
        a = ((t - 13.6) * 0.7 + k / 3) % 1
        px, py = 600 + a * 500, 900 - a * 500 + 60 * math.sin(a * 8 + k)
        if k == 1:
            pen.poly([(px - 40, py), (px + 40, py - 10), (px - 20, py + 20)], fill=(255, 255, 255), width=4)
        else:
            pen.ellipse(px, py, 32, 20, fill=(200, 200, 210), width=4)
            wing = 22 * math.sin(t * 25 + k)
            pen.poly([(px - 10, py - 5), (px + 15, py - 5), (px, py - 30 - wing)], fill=(230, 230, 235), width=4)
            pen.rect(px - 12, py + 14, px + 14, py + 30, fill=(255, 250, 220), width=3)
    if t > 14.56:
        text(pen, 540, 600, "I DON'T CARE", 76, fill=(255, 230, 90), sw=8)


def story_block(pen, t):
    pen.rect(-900, -1000, 2000, 3200, fill=(180, 210, 240), outline=None)
    for k, x in enumerate((-200, 120, 760, 1040)):
        pen.rect(x, 520 + 60 * (k % 2), x + 280, 1350, fill=[(200, 120, 100), (160, 150, 200), (220, 180, 110), (150, 190, 160)][k], width=7)
        for r in range(6):
            pen.rect(x + 40, 600 + 60 * (k % 2) + r * 110, x + 100, 670 + 60 * (k % 2) + r * 110, fill=(255, 240, 180), width=4)
    pen.rect(-900, 1350, 2000, 3200, fill=(110, 110, 120), outline=None)
    pen.line([(420, 1350), (420, 900)], width=10)
    pen.rect(330, 860, 640, 940, fill=(60, 150, 90), width=6)
    text(pen, 485, 900, "YOUR BLOCK", 40, fill=(255, 255, 255), sw=3)
    x = lerp(950, 600, (t - 17.6) / 1.3)
    hoodie = dict(HOST_A, shirt=(60, 60, 65))
    person(pen, x, 1250, 1.5, hoodie, facing=-1, pose="walk", walk=True, t=t, eyes="half", mouth=0.0)


def story_youth(pen, t):
    pen.rect(-900, -1000, 2000, 3200, fill=(120, 125, 150), outline=None)
    pen.rect(-900, 1400, 2000, 3200, fill=(90, 95, 110), outline=None)
    for k in range(12):
        ry = 200 + ((t * 600 + k * 80) % 1100)
        pen.line([(80 + k * 85, ry), (72 + k * 85, ry + 30)], width=4, fill=(170, 190, 230))
    person(pen, 540, 1300, 1.6, OLD_A, facing=1, pose="handshead", t=t, sitting=True, eyes="cry", brows="sad", mouth=0.8)
    pen.line([(640, 1180), (700, 1400)], width=9, fill=(140, 100, 60))  # cane


def story_divorce(pen, t):
    tw.room(pen, wall=(245, 235, 220))
    pen.rect(620, 1160, 980, 1400, fill=(120, 120, 140), width=7)  # TV stand
    pen.rect(650, 900, 950, 1100, fill=(30, 30, 40), width=8)
    person(pen, 300, 1290, 1.5, WIFE, facing=1, pose="shrug" if False else "belly", t=t, eyes="half", mouth="flat")
    rf.bubble(pen, 330, 620, 230, 160, 330, 880)
    text(pen, 330, 600, "DIVORCE", 60, fill=(230, 60, 60), sw=6)
    pen.line([(270, 650), (300, 690), (280, 720)], width=5)
    person(pen, 800, 1330, 1.3, HUSBAND, facing=-1, pose="rest", sitting=True, t=t, eyes="happy")


def story_deal(pen, t):
    tw.room(pen, wall=(245, 235, 220))
    pen.rect(330, 1170, 750, 1220, fill=(150, 110, 80), width=7)
    pen.line([(360, 1220), (360, 1400)], width=9)
    pen.line([(720, 1220), (720, 1400)], width=9)
    person(pen, 230, 1290, 1.45, WIFE, facing=1, pose="point", t=t, mouth=0.5 * abs(math.sin(t * 12)), eyes="half")
    person(pen, 860, 1290, 1.45, HUSBAND, facing=-1, pose="pray", t=t, eyes="happy", mouth=0.0)
    for k in range(3):
        a = ((t - 28) * 0.8 + k / 3) % 1
        rf.text(pen, 860 + 40 * math.sin(a * 6 + k), 860 - a * 300, "<3", 40, fill=(240, 90, 120), sw=4)
    if t > 30.4:
        pen.rect(430, 760, 650, 980, fill=(250, 250, 240), width=6)
        text(pen, 540, 820, "DEAL?", 40, fill=(40, 40, 40), stroke=(40, 40, 40), sw=1)
        pen.line([(470, 900), (610, 900)], width=4)


def story_snip(pen, t):
    tw.room(pen, wall=(225, 240, 240), floor=(200, 210, 210))
    pen.rect(140, 1200, 760, 1260, fill=(250, 250, 250), width=7)  # hospital bed
    pen.line([(170, 1260), (170, 1400)], width=8)
    pen.line([(730, 1260), (730, 1400)], width=8)
    rp = RotPen(pen, 450, 1200, -math.pi / 2 + 0.05)
    person(rp, 450, 1110, 1.3, HUSBAND, facing=1, pose="rest", t=t, eyes="huge", mouth="O", brows="up")
    person(pen, 880, 1300, 1.5, DOCTOR, facing=-1, pose="point", t=t, eyes="half", mouth=0.0)
    sx, sy = 720, 1020
    open_ = 0.4 * abs(math.sin(t * 10))
    for d in (-1, 1):
        pen.line([(sx, sy), (sx - 90, sy + d * 40 * (0.3 + open_))], width=10, fill=(150, 160, 170))
        pen.ellipse(sx + 25, sy + d * 18, 16, 12, fill=None, outline=(230, 80, 80), width=6)
    if t > 31.36:
        k = ease_out((t - 31.36) / 0.15)
        text(pen, 520, 760, "SNIP!", 120 * (1.6 - 0.6 * k), fill=(240, 70, 70), sw=10)


def story_leave(pen, t):
    tw.room(pen, wall=(230, 225, 245))
    pen.rect(820, 900, 990, 1400, fill=(150, 100, 70), width=7)  # door
    pen.rect(80, 1200, 500, 1260, fill=(250, 250, 250), width=7)
    rp = RotPen(pen, 290, 1200, -math.pi / 2 + 0.05)
    person(rp, 290, 1110, 1.2, HUSBAND, facing=1, pose="rest", t=t, eyes="huge", mouth="O")
    x = lerp(560, 860, ease_io((t - 32.1) / 1.2))
    person(pen, x, 1290, 1.45, WIFE, facing=1, pose="leash", walk=True, t=t, eyes="happy", mouth=0.0)
    pen.rect(x + 70, 1210, x + 150, 1300, fill=(230, 180, 80), width=6)  # suitcase
    pen.rect(200, 560, 440, 740, fill=(255, 255, 255), width=7)
    text(pen, 320, 620, "1 WEEK", 46, fill=(230, 80, 80), sw=3)
    text(pen, 320, 690, "LATER", 40, fill=(40, 40, 40), stroke=(40, 40, 40), sw=1)


def story_streets(pen, t):
    pen.rect(-900, -1000, 2000, 3200, fill=(250, 190, 140), outline=None)
    for k, x in enumerate(range(-300, 1500, 260)):
        pen.rect(x, 560 + 80 * (k % 3), x + 230, 1300, fill=(160 - 10 * k % 40, 120, 150), width=6)
    pen.rect(-900, 1300, 2000, 3200, fill=(110, 110, 120), outline=None)
    for k in range(6):
        pen.rect(-100 + k * 220, 1480, 0 + k * 220, 1500, fill=(250, 250, 250), outline=None)
    pen.line([(160, 1300), (160, 880)], width=10)
    pen.rect(40, 840, 440, 920, fill=(60, 150, 90), width=6)
    text(pen, 240, 880, "THESE STREETS", 36, fill=(255, 255, 255), sw=3)
    x = lerp(380, 860, (t - 43.05) / 1.45)
    person(pen, x, 1220, 1.5, WIFE, facing=1, pose="leash", walk=True, t=t, eyes="happy", mouth=0.0)
    hx, hy = x + 12, 1220 - 95 * 1.5 - 40 * 1.5 * 0.95
    pen.rect(hx - 10, hy - 14, hx + 50, hy + 4, fill=(20, 20, 25), outline=None)
    pen.rect(x + 70, 1140, x + 150, 1230, fill=(230, 180, 80), width=6)
    text(pen, 540, 640, "BEST OF LUCK", 70, fill=(255, 230, 90), sw=8)


# ------------------------------------------------------------------ captions

rf.CAPTIONS[:] = [
    [("CRAZY", 0.00), ("GIRLS,", 0.58)],
    [("WHAT", 1.04), ("YOU'VE", 1.18), ("BEEN", 1.42), ("UP", 1.48), ("TO", 1.68), ("RECENTLY?", 1.78)],
    [("EVERY", 2.48), ("YEAR", 2.66), ("I", 2.90), ("UNBLOCK", 3.14), ("HIM", 3.50)],
    [("AND", 3.78), ("TEXT", 4.04), ("HIM", 4.24), ("HAPPY", 4.40), ("FATHER'S", 4.68), ("DAY.", 5.16)],
    [("THEN", 5.50), ("BLOCK", 5.66), ("HIM", 5.88), ("AGAIN.", 6.06)],
    [("HE'S", 6.30), ("BEEN", 6.52), ("TRYING", 6.64), ("TO", 6.82), ("REACH", 6.96), ("ME", 7.14)],
    [("FOR", 7.28), ("THREE", 7.46), ("YEARS.", 7.74)],
    [("THERE'S", 8.28), ("NO", 8.54), ("BABY.", 8.68)],
    [("OH,", 12.38), ("I'D", 12.78), ("FIND", 13.18), ("A", 13.38), ("WAY", 13.54)],
    [("TO", 13.62), ("COMMUNICATE", 13.80), ("WITH", 14.02), ("YOU.", 14.18)],
    [("I", 14.56), ("DON'T", 14.62), ("CARE.", 14.98)],
    [("WHAT'S", 16.82), ("A", 17.08), ("BLOCK?", 17.22)],
    [("YEAH,", 17.60), ("I'LL", 17.82), ("COME", 17.96), ("INTO", 18.00), ("YOUR", 18.10), ("BLOCK.", 18.28)],
    [("YEAH,", 18.92), ("BRO.", 19.26)],
    [("GIVE", 19.76), ("ME", 19.84), ("MY", 20.02), ("YOUTH.", 20.22)],
    [("SO", 21.24), ("YOU'RE", 21.38), ("TEXTING", 21.52), ("ME", 21.70)],
    [("HAPPY", 22.00), ("FATHER'S", 22.06), ("DAY.", 22.26)],
    [("WHERE", 22.70), ("ARE", 23.10), ("THEY?", 23.34)],
    [("WHERE'S", 23.50), ("THE", 23.74), ("CHILD,", 23.82), ("BRO?", 24.42)],
    [("THIS", 25.00), ("ONE", 25.24), ("WOULD", 25.46), ("MAKE", 25.68), ("YOUR", 25.86), ("HAIR", 25.98), ("FALL", 26.10), ("OUT.", 26.26)],
    [("I", 26.40), ("KNEW", 26.52), ("I", 26.66), ("WAS", 26.78), ("GONNA", 26.88), ("DIVORCE", 27.04), ("MY", 27.42), ("HUSBAND,", 27.64)],
    [("BUT", 28.04), ("I", 28.28), ("TOLD", 28.44), ("HIM", 28.64), ("I", 28.90), ("WOULD", 29.10), ("STAY", 29.38)],
    [("AND", 29.68), ("WORK", 29.96), ("THINGS", 30.12), ("OUT", 30.38)],
    [("IF", 30.70), ("HE", 30.98), ("GOT", 31.12), ("A", 31.22), ("VASECTOMY.", 31.36)],
    [("I", 32.16), ("LEFT", 32.30), ("HIS", 32.42), ("ASS", 32.60), ("THE", 32.72), ("WEEK", 32.86), ("AFTER.", 33.04)],
    [("OH,", 33.78), ("YEAH.", 34.24)],
    [("NOOO!", 36.02)],
    [("YOU'RE", 38.00), ("NEVER", 38.40), ("HAVING", 38.68), ("KIDS", 38.86), ("AGAIN.", 39.06)],
    [("IF", 41.12), ("NOT", 41.32), ("WITH", 41.48), ("ME,", 41.66), ("WITH", 41.90), ("NOBODY.", 42.04)],
    [("BEST", 43.10), ("OF", 43.30), ("LUCK", 43.52), ("IN", 43.66), ("THESE", 43.84), ("STREETS.", 44.10)],
    [("I'M", 44.52), ("GONE.", 44.76)],
]
rf._cap.clear()


def speaker(t):
    for a, b, who in ((0, 9.0, "B"), (12.3, 15.4, "A"), (16.8, 18.6, "A"), (18.9, 19.6, "B"), (19.7, 20.6, "A"),
                      (21.2, 23.45, "B"), (23.45, 24.6, "A"), (24.6, 33.2, "B"), (33.7, 34.6, "A"), (36.0, 36.8, "A"),
                      (38.0, 39.3, "A"), (41.1, END, "B")):
        if a <= t < b:
            return who
    return None


def frame(t, m, seed):
    img = Image.new("RGB", (W * SS, H * SS), (200, 150, 120))
    wide, cA, cB = (540, 1080, 1.22), (390, 1010, 1.8), (700, 1010, 1.8)

    def P(c, push=0.0, sx=0.0):
        return Pen(img, c[0], c[1], c[2] * (1 + push), sx=sx, seed=seed)

    sp = speaker(t)
    mA = m if sp == "A" else 0.0
    mB = m if sp == "B" else 0.0
    fall = lambda t0, d=0.45: (math.pi / 2 * 1.02) * ease_out((t - t0) / d) if t >= t0 else 0.0  # noqa: E731
    shake = lambda t0: 30 * kick(t, t0 + 0.4, 0.2) * math.sin(t * 70)  # noqa: E731
    reader = dict(pose="phone", prop="phone", eyes="half")

    if t < 1.2:
        tw.studio(P(wide, 0.05 * ease_io(t / 1.2)), t, dict(), dict(reader, mouth=mB))
    elif t < 2.45:
        tw.studio(P(cB), t, dict(), dict(reader, mouth=mB, eyes="up"))
    elif t < 6.25:
        story_unblock(P((540, 1040, 1.4)), t)
    elif t < 8.2:
        story_calling(P((540, 1040, 1.4)), t)
    elif t < 9.0:
        story_nobaby(P((540, 1040, 1.35)), t)
    elif t < 12.3:  # OHHHH - A goes over backwards
        pen = P(wide, sx=shake(10.2))
        tw.studio(pen, t, dict(LAUGH, pose="handshead", eyes="huge" if t < 10.2 else "laughcry", mouth="O" if t < 10.2 else "laugh",
                                ang=fall(10.2), kick=1.0 if t > 10.7 else 0, nomic=t > 10.2),
                  dict(LAUGH, prop="phone", pose="facepalm"))
        if 10.55 < t < 11.1:
            text(pen, 260, 980, "CRASH!", 90, fill=(255, 210, 60), sw=9)
        text(pen, 540, 760, "OHHHHH!", 80 + 10 * math.sin(t * 20), fill=(255, 230, 90), sw=8)
    elif t < 13.6:
        pen = P((280, 1230, 1.6))
        tw.studio(pen, t, dict(ang=fall(0), mouth=mA, eyes="huge", pose="point", nomic=True), dict(LAUGH, prop="phone"))
    elif t < 15.4:
        story_communicate(P((540, 1040, 1.3)), t)
    elif t < 16.8:  # climbs back up, both laughing
        up = ease_io((t - 15.4) / 0.9)
        tw.studio(P(wide), t, dict(LAUGH, ang=(math.pi / 2) * (1 - up), nomic=up < 0.8), dict(LAUGH, prop="phone"))
    elif t < 17.6:
        pen = P(cA, 0.1 * kick(t, 16.82, 0.15))
        tw.studio(pen, t, dict(pose="shrug", mouth=mA, eyes="huge", brows="up"), dict())
        qmarks(pen, 390, 600, t)
    elif t < 18.9:
        story_block(P((620, 1040, 1.35)), t)
    elif t < 19.7:
        tw.studio(P(cB), t, dict(), dict(pose="point", mouth=mB if mB else "laugh", eyes="happy", bob=6))
    elif t < 21.2:
        story_youth(P((540, 1060, 1.35)), t)
        text(P((540, 1060, 1.35)), 540, 640, "GIVE ME MY YOUTH!", 64, fill=(255, 230, 90), sw=8)
    elif t < 22.7:
        tw.studio(P(cB), t, dict(), dict(reader, mouth=mB, eyes="open"))
    elif t < 23.45:
        pen = P(wide)
        tw.studio(pen, t, dict(eyes="happy"), dict(pose="think", mouth=mB, eyes="open"))
        qmarks(pen, 720, 640, t)
    elif t < 24.6:
        pen = P(cA, 0.08 * kick(t, 24.42, 0.15))
        tw.studio(pen, t, dict(pose="shrug", mouth=mA, eyes="huge", brows="up"), dict())
        qmarks(pen, 390, 600, t)
    elif t < 26.4:  # "this one would make your hair fall out" - A's hair flies off
        pen = P(wide)
        bald = t > 26.1
        tw.SEATS["A"] = (360, 1, BALD_A if bald else HOST_A)
        tw.studio(pen, t, dict(eyes="huge" if bald else "open", mouth="O" if bald else 0.0, brows="up" if bald else None),
                  dict(reader, mouth=mB))
        tw.SEATS["A"] = (360, 1, HOST_A)
        if bald:
            hx, hy, r, _ = head_of("A")
            k = t - 26.1
            rp = RotPen(pen, hx, hy, -6 * k, -260 * k, -700 * k + 1400 * k * k)
            rp.arc(hx, hy, r * 0.98, r * 0.98, 195, 345, width=r * 0.3, fill=(32, 24, 20))
            text(pen, 360, 620, "POOF", 60, fill=(255, 230, 90), sw=7)
    elif t < 28.04:
        story_divorce(P((540, 1020, 1.35)), t)
    elif t < 30.7:
        story_deal(P((540, 1040, 1.35)), t)
    elif t < 32.1:
        story_snip(P((560, 1060, 1.3)), t)
    elif t < 33.3:
        story_leave(P((540, 1040, 1.35)), t)
    elif t < 36.0:
        pen = P(wide, 0.03 * math.sin(t * 3))
        tw.studio(pen, t, dict(LAUGH, pose="facepalm", mouth=mA if mA else "laugh"), dict(LAUGH, prop="phone", lean=-10))
        ha(pen, 760, 760, t, 0.3)
    elif t < 38.0:  # NOOO! and over he goes
        pen = P(wide, 0.08 * kick(t, 36.02, 0.2), sx=shake(36.3))
        tw.studio(pen, t, dict(LAUGH, pose="handshead", mouth=mA if t < 36.8 else "laugh", eyes="huge" if t < 36.3 else "laughcry",
                                ang=fall(36.3), kick=1.0 if t > 36.8 else 0, nomic=t > 36.3),
                  dict(LAUGH, prop="phone", pose="facepalm"))
        if 36.65 < t < 37.2:
            text(pen, 260, 980, "CRASH!", 90, fill=(255, 210, 60), sw=9)
    elif t < 39.4:
        pen = P((280, 1230, 1.6))
        tw.studio(pen, t, dict(ang=fall(0), mouth=mA, eyes="huge", pose="point", nomic=True), dict(LAUGH))
    elif t < 41.1:  # both rolling on the deck
        pen = P(wide)
        rooftop_bg(pen, t)
        for who, ph in (("A", 0), ("B", 1.3)):
            x, facing, look = SEATS[who]
            roll = math.pi / 2 + 0.35 * math.sin(t * 6 + ph)
            rp = RotPen(pen, x - facing * 70, FLOOR, -facing * roll, facing * 260 + 25 * math.sin(t * 3 + ph), -20)
            person(rp, x, HIP, S, look, facing=facing, pose="belly", mouth="laugh", eyes="laughcry", t=t, sitting=True, kick=1.0)
        for who in ("A", "B"):
            tw.chair(pen, SEATS[who][0], SEATS[who][1])
        if int(t * 4) % 2:
            text(pen, 540, 820, "HAHAHAHA", 80, fill=(255, 210, 60), sw=9)
    elif t < 43.05:  # B puts the shades on
        pen = P(cB)
        tw.studio(pen, t, dict(), dict(pose="talk", mouth=mB, eyes="half", lean=-4))
        if t > 41.6:
            shades(pen, "B", -4)
    elif t < 44.5:
        story_streets(P((560, 1040, 1.35)), t)
    else:
        pen = P(cB, 0.1 * kick(t, 44.52, 0.15))
        tw.studio(pen, t, dict(), dict(pose="point", mouth=mB, eyes="half"))
        shades(pen, "B")
    return img.resize((W, H), Image.LANCZOS)


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
        f = np.asarray(frame(t, max(track[i], track[min(i + 1, len(track) - 1)]), seed=di), np.float32)
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
            if tt > END - 0.35:
                px *= clamp((END - tt) / 0.35, 0, 1)
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
