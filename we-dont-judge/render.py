"""We Listen, We Don't Judge (Bible edition) - hand-drawn stick animation.

A couch confession game (Eve & Cain, then Potiphar & his wife) with cut-aways
to stick-figure Bible flashbacks. Audio is the original clip, unchanged.

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
person, text, INK, SS = tw.person, tw.text, tw.INK, tw.SS

W, H, FPS = 1080, 1920, 30
END = 80.4
S, HIP = 1.7, 1235
LEFT, RIGHT = 330, 750

CAIN = dict(skin=(110, 72, 50), shirt=(40, 50, 75), pants=(50, 55, 70), hair="short", beard=None)
EVE = dict(skin=(110, 72, 50), shirt=(30, 90, 80), pants=(30, 90, 80), hair="bald", beard=None, dress=True, wrap=(200, 160, 50))
POTIPHAR = dict(skin=(110, 72, 50), shirt=(175, 190, 165), pants=(175, 190, 165), hair="short", beard=None, cap=True)
WIFE = dict(skin=(110, 72, 50), shirt=(230, 190, 60), pants=(230, 190, 60), hair="bald", beard=None, dress=True, wrap=(130, 40, 60))
ADAM = dict(skin=(110, 72, 50), shirt=(90, 160, 80), pants=(60, 60, 70), hair="short", beard=(30, 22, 18))
ABEL = dict(skin=(120, 80, 55), shirt=(200, 170, 120), pants=(150, 120, 80), hair="short", beard=None)
JOSEPH = dict(skin=(120, 80, 55), shirt=(240, 90, 80), pants=(60, 70, 110), hair="short", beard=None, coat=True)
GARDEN_EVE = dict(EVE, wrap=None, hair="puff", shirt=(90, 170, 80), pants=(90, 170, 80))
GARDEN_ADAM = dict(ADAM, shirt=(90, 170, 80))


def char(pen, x, hip, s, look, facing=1, **kw):
    (hx, hy, r), hand = person(pen, x, hip, s, look, facing=facing, **kw)
    lean = kw.get("lean", 0.0)
    if look.get("coat"):  # coat of many colours
        nx, ny = x + facing * lean * s, hip - 95 * s
        for k, c in enumerate(((240, 90, 80), (250, 190, 60), (90, 180, 90), (80, 140, 230), (170, 100, 210))):
            y = ny + 18 * s + k * 16 * s
            w = lerp(24, 32, k / 4) * s
            pen.line([(nx - w, y), (nx + w, y)], width=12 * s / 1.5, fill=c)
    if look.get("wrap"):
        c = look["wrap"]
        pen.poly([(hx - r * 1.12, hy - r * 0.1), (hx - r * 1.3, hy + r * 1.4), (hx + r * 1.3, hy + r * 1.4), (hx + r * 1.12, hy - r * 0.1)],
                 fill=c, width=6)
        pen.arc(hx, hy, r * 1.07, r * 1.07, 160, 380, width=r * 0.42, fill=c)
        pen.arc(hx, hy, r * 1.28, r * 1.28, 160, 380, width=6)
        for a in range(170, 380, 30):
            pen.ellipse(hx + math.cos(math.radians(a)) * r * 1.07, hy + math.sin(math.radians(a)) * r * 1.07, r * 0.08, r * 0.08,
                        fill=(30, 25, 25), outline=None)
        # redraw the face over the wrap
        rf.face(pen, hx, hy, r, look, facing, eyes=kw.get("eyes", "open"), mouth=kw.get("mouth", 0.0), brows=kw.get("brows"), t=kw.get("t", 0))
    if look.get("cap"):
        pen.poly([(hx - r * 0.95, hy - r * 0.5), (hx + r * 0.95, hy - r * 0.5), (hx + r * 1.15 * facing if facing > 0 else hx + r * 0.6, hy - r * 1.35),
                  (hx - r * 0.6 if facing > 0 else hx - r * 1.15, hy - r * 1.4)], fill=(60, 90, 190), width=6)
        for k in range(3):
            pen.line([(hx - r * 0.8, hy - r * (0.65 + k * 0.22)), (hx + r * 0.8, hy - r * (0.75 + k * 0.22))], width=4, fill=(230, 235, 255))
        ex = hx + facing * r * 0.3
        for dx in (-0.32, 0.32):
            pen.ellipse(ex + dx * r, hy - r * 0.15, r * 0.21, r * 0.21, fill=None, outline=(40, 30, 25), width=4)
        pen.line([(ex - r * 0.11, hy - r * 0.17), (ex + r * 0.11, hy - r * 0.17)], width=4)
    return (hx, hy, r), hand


# ------------------------------------------------------------------ living room


def living(pen, t, door=False):
    pen.rect(-900, -1000, 2000, 1400, fill=(242, 228, 170), outline=None)
    pen.rect(140, 300, 940, 980, fill=(150, 125, 100), width=7)  # blinds
    for y in range(320, 980, 34):
        pen.line([(150, y), (930, y)], width=6, fill=(110, 90, 70))
    pen.rect(-900, 1400, 2000, 3200, fill=(200, 180, 150), outline=None)
    pen.line([(-900, 1400), (2000, 1400)], width=7)
    if door:
        pen.rect(960, 760, 1160, 1400, fill=(160, 110, 70), width=7)
    pen.poly([(70, 1020), (1010, 1020), (1010, 1240), (70, 1240)], fill=(215, 212, 208), width=8)  # couch back
    pen.poly([(30, 1230), (1050, 1230), (1050, 1420), (30, 1420)], fill=(228, 225, 220), width=8)
    for x in (30, 1000):
        pen.rect(x, 1130, x + 50, 1420, fill=(205, 200, 195), width=7)


def couch(pen, t, left, right, ml, mr, ls=None, rs=None, door=False, tags=False):
    living(pen, t, door)
    ls, rs = ls or {}, rs or {}
    hl = char(pen, LEFT, HIP + ls.get("dy", 0), S, left, facing=1, sitting=True, t=t, mouth=ls.get("mouth", ml), eyes=ls.get("eyes", "open"),
              pose=ls.get("pose", "rest"), brows=ls.get("brows"), lean=ls.get("lean", 0))
    if rs.get("stand"):
        hr = char(pen, RIGHT, HIP + 70, S, right, facing=-1, t=t, mouth=rs.get("mouth", mr), eyes=rs.get("eyes", "open"),
                  pose=rs.get("pose", "point"), brows=rs.get("brows"))
    else:
        hr = char(pen, RIGHT, HIP + rs.get("dy", 0), S, right, facing=-1, sitting=True, t=t, mouth=rs.get("mouth", mr),
                  eyes=rs.get("eyes", "open"), pose=rs.get("pose", "rest"), brows=rs.get("brows"), lean=rs.get("lean", 0))
    if tags:
        for (hx, hy, r), _ in (hl, hr):
            pass
    return hl, hr


def tag(pen, x, y, s):
    text(pen, x, y, s, 44, fill=(255, 220, 70), sw=5)


# ------------------------------------------------------------------ flashbacks


def sky_ground(pen, sky=(170, 220, 250), ground=(120, 195, 100), gy=1350):
    pen.rect(-900, -1000, 2000, 3200, fill=sky, outline=None)
    pen.rect(-900, gy, 2000, 3200, fill=ground, outline=None)
    pen.line([(-900, gy), (2000, gy)], width=6)


def tree(pen, x, gy, s=1.0, apples=True):
    pen.rect(x - 30 * s, gy - 330 * s, x + 30 * s, gy, fill=(140, 95, 60), width=7)
    for k in range(5):
        pen.ellipse(x + (k - 2) * 75 * s, gy - 400 * s - (k % 2) * 70 * s, 115 * s, 100 * s, fill=(70, 160, 80), width=6)
    if apples:
        for k in range(6):
            pen.ellipse(x + (k - 2.5) * 55 * s, gy - 380 * s + (k % 2) * 60 * s, 16 * s, 16 * s, fill=(230, 60, 60), width=4)


def eden(pen, t):
    sky_ground(pen)
    pen.ellipse(880, 420, 80, 80, fill=(255, 220, 90), width=6)
    tree(pen, 200, 1350, 1.1)
    tree(pen, 900, 1350, 0.9)
    pen.poly([(-900, 1480), (2000, 1440), (2000, 1520), (-900, 1560)], fill=(120, 190, 240), width=5)  # river
    for k in range(10):
        fx = 60 + k * 110
        pen.ellipse(fx, 1400 + (k % 3) * 20, 12, 12, fill=[(250, 120, 160), (255, 220, 80), (250, 250, 250)][k % 3], width=3)
    text(pen, 540, 560, "GARDEN OF EDEN", 64, fill=(255, 255, 255), stroke=(60, 120, 60), sw=7)


def flash_garden(pen, t):
    eden(pen, t)
    char(pen, 430, 1250, 1.3, GARDEN_ADAM, facing=1, pose="talk", t=t, eyes="happy", walk=True)
    char(pen, 640, 1250, 1.3, GARDEN_EVE, facing=-1, pose="talk", t=t, eyes="happy", walk=True)
    for k in range(3):
        a = (t * 0.6 + k / 3) % 1
        text(pen, 540 + 80 * math.sin(a * 6 + k), 1000 - a * 300, "<3", 40, fill=(240, 90, 120), sw=4)


def flash_chased(pen, t):
    eden(pen, t)
    for k in range(5):  # cloud
        pen.ellipse(560 + k * 70, 400 - (k % 2) * 40, 90, 70, fill=(255, 255, 255), width=6)
    pen.poly([(740, 470), (620, 610), (650, 640), (770, 500)], fill=(255, 240, 200), width=6)  # pointing hand
    pen.ellipse(615, 625, 30, 30, fill=(255, 240, 200), width=5)
    k = ease_out((t - 13.68) / 0.2)
    text(pen, 600, 780, "GET OUT!", 100 * (0.5 + 0.5 * k), fill=(240, 70, 70), sw=9)
    run = (t - 12.07) * 260
    char(pen, 420 - run, 1250, 1.2, GARDEN_ADAM, facing=-1, pose="walk", walk=True, t=t * 2, eyes="huge", mouth="O")
    char(pen, 600 - run, 1250, 1.2, GARDEN_EVE, facing=-1, pose="walk", walk=True, t=t * 2 + 0.3, eyes="huge", mouth="O")
    for k in range(3):
        pen.line([(560 - run + 120, 1050 + k * 50), (700 - run + 120, 1050 + k * 50)], width=5)


def flash_snake(pen, t):
    eden(pen, t)
    char(pen, 640, 1250, 1.45, GARDEN_EVE, facing=-1, pose="sip" if t > 17.86 else "talk", t=t, eyes="happy" if t > 17.86 else "open",
         mouth=0.4 * abs(math.sin(t * 10)) if t > 17.86 else 0.0)
    # snake curled on the tree branch, swaying
    sx, sy = 330, 860
    pts = [(sx + k * 22, sy + 30 * math.sin(k * 0.9 + t * 4)) for k in range(14)]
    pen.line(pts, width=26, fill=(90, 180, 70))
    hx, hy = pts[-1]
    pen.ellipse(hx + 18, hy, 30, 22, fill=(90, 180, 70), width=6)
    pen.ellipse(hx + 26, hy - 8, 6, 6, fill=INK, outline=None)
    pen.line([(hx + 46, hy + 4), (hx + 70, hy + 4 + 6 * math.sin(t * 30))], width=4, fill=(230, 60, 60))
    if t < 17.86:
        pen.ellipse(hx + 90, hy + 50, 26, 26, fill=(230, 60, 60), width=5)  # apple offered
    if t > 21.5:  # "make me wise"
        k = ease_out((t - 21.5) / 0.2)
        hx2, hy2 = 640 + 0, 1250 - 95 * 1.45 - 40 * 1.45 * 0.95
        pen.poly([(hx2 - 70 * k, hy2 - 60), (hx2 + 70 * k, hy2 - 60), (hx2, hy2 - 90)], fill=(30, 30, 40), width=4)
        text(pen, 640, 640, "WISE!", 80 * k + 1, fill=(255, 230, 90), sw=8)


def flash_abel(pen, t):
    sky_ground(pen, sky=(240, 200, 150), ground=(170, 190, 100))
    rp = RotPen(pen, 720, 1350, math.pi / 2)
    char(rp, 720, 1250, 1.2, ABEL, facing=1, pose="rest", t=t, eyes="huge", mouth="O")
    for k in range(4):  # cartoon stars
        a = t * 4 + k * 1.6
        text(pen, 760 + 70 * math.cos(a), 1180 + 25 * math.sin(a), "*", 50, fill=(255, 220, 60), sw=4)
    char(pen, 330, 1250, 1.4, CAIN, facing=1, pose="walk", t=t, eyes="half", mouth=0.0)
    pen.line([(270, 1150), (230, 1020)], width=14, fill=(140, 95, 60))  # stick hidden behind his back
    if t > 27.86:  # the lie
        for k in range(4):
            pen.ellipse(560 + k * 80, 440 - (k % 2) * 40, 100, 80, fill=(255, 255, 255), width=6)
        text(pen, 680, 445, "WHERE'S ABEL?", 40, fill=(80, 80, 120), sw=4)
        n = 40 + 120 * ease_out((t - 28.28) / 0.4)
        hx, hy = 330 + 12, 1250 - 95 * 1.4 - 40 * 1.4 * 0.95
        pen.line([(hx + 40, hy), (hx + 40 + n, hy - 5)], width=12, fill=(110, 72, 50))  # nose grows
        text(pen, 330, 760, "NO IDEA!", 56, fill=(255, 255, 255), sw=6)


def flash_joseph_arrives(pen, t):
    living(pen, t, door=True)
    char(pen, 400, 1300, 1.5, POTIPHAR, facing=1, pose="point", t=t, eyes="happy")
    x = lerp(1050, 700, ease_io((t - 46.3) / 1.3))
    char(pen, x, 1300, 1.5, JOSEPH, facing=-1, pose="rest", t=t, eyes="open", walk=t < 47.6)
    if t > 49.16:
        k = ease_out((t - 49.16) / 0.2)
        text(pen, 700, 820, "CHIEF SERVANT", 56 * k + 1, fill=(255, 220, 70), sw=6)


def flash_joseph_coat(pen, t):
    living(pen, t, door=True)
    sweep = math.sin(t * 6)
    grab = t > 64.9
    jx = 640 if not grab else lerp(640, 1150, ease_io((t - 65.2) / 0.8))
    jlook = dict(JOSEPH, coat=not grab, shirt=(245, 245, 245) if grab else JOSEPH["shirt"])
    char(pen, jx, 1300, 1.5, jlook, facing=1 if grab else -1, pose="walk" if grab else "push", walk=grab, t=t,
         eyes="huge" if grab else "open", mouth="O" if grab else 0.0)
    if not grab:
        pen.line([(560 + 20 * sweep, 1200), (520 + 30 * sweep, 1400)], width=8, fill=(160, 120, 70))  # broom
        for k in range(3):
            text(pen, 600 + k * 60, 950 - 30 * k, "*", 40, fill=(255, 255, 255), sw=3)
    char(pen, 280, 1300, 1.5, WIFE, facing=1, pose="point" if grab else "think", t=t, eyes="happy" if not grab else "huge",
         mouth=0.0 if not grab else "O")
    if not grab:
        for k in range(3):
            a = (t + k / 3) % 1
            text(pen, 300 + 40 * math.sin(a * 6 + k), 880 - a * 200, "<3", 40, fill=(240, 90, 120), sw=4)
    else:  # she's left holding the coat
        for k, c in enumerate(((240, 90, 80), (250, 190, 60), (90, 180, 90), (80, 140, 230))):
            pen.rect(380, 1120 + k * 30, 470, 1150 + k * 30, fill=c, width=4)


def flash_jail(pen, t):
    pen.rect(-900, -1000, 2000, 3200, fill=(150, 150, 160), outline=None)
    pen.rect(-900, 1350, 2000, 3200, fill=(120, 120, 128), outline=None)
    for k in range(10):
        pen.line([(-60 + k * 70, 400), (-60 + k * 70, 1400)], width=4, fill=(130, 130, 140))
    char(pen, 330, 1300, 1.4, dict(JOSEPH, coat=False, shirt=(245, 245, 245)), facing=1, pose="cry", t=t, eyes="cry", brows="sad",
         mouth="frown", sitting=True)
    for k in range(7):  # bars
        pen.line([(130 + k * 70, 600), (130 + k * 70, 1400)], width=12, fill=(60, 60, 70))
    pen.line([(110, 600), (590, 600)], width=12, fill=(60, 60, 70))
    char(pen, 810, 1300, 1.4, POTIPHAR, facing=-1, pose="sip", prop="cup", t=t, eyes="half")
    text(pen, 810, 820, "UNBOTHERED", 46, fill=(120, 200, 255), sw=6)


# ------------------------------------------------------------------ captions

rf.CAPTIONS[:] = [
    [("OKAY,", 5.52), ("CAIN.", 5.94)],
    [("REMEMBER", 6.36), ("THE", 6.76), ("GARDEN", 7.00), ("YOUR", 7.22), ("FATHER", 7.32), ("TALKS", 7.54), ("ABOUT,", 7.78)],
    [("GARDEN", 8.16), ("OF", 8.32), ("EDEN?", 8.54)],
    [("YES.", 9.50)],
    [("THAT", 9.92), ("HE", 10.02), ("SAID", 10.14), ("THAT", 10.18), ("WE", 10.32), ("LEFT?", 10.42)],
    [("WE", 12.07), ("DIDN'T", 12.28), ("ACTUALLY", 12.44), ("LEAVE.", 12.68)],
    [("GOD", 13.32), ("CHASED", 13.68), ("US", 13.96), ("OUT", 14.16)],
    [("BECAUSE", 14.86), ("OF", 15.62), ("ME.", 15.86)],
    [("I", 17.14), ("WENT", 17.56), ("AND", 17.70), ("ATE", 17.86), ("THE", 18.00), ("FOOD", 18.16)],
    [("THAT", 18.30), ("HE", 18.46), ("SAID", 18.60), ("I", 18.66), ("SHOULD", 18.74), ("NOT", 18.84), ("EAT", 18.98)],
    [("BECAUSE", 19.16), ("A", 19.42), ("SNAKE", 19.68), ("GAVE", 19.80), ("IT", 20.02), ("TO", 20.22), ("ME", 20.28)],
    [("AND", 20.48), ("SAID", 20.60), ("IT", 20.68), ("WILL", 20.86), ("MAKE", 20.98), ("ME", 21.06), ("BE", 21.32), ("WISE.", 21.50)],
    [("A", 21.88), ("SNAKE?!", 22.22)],
    [("HOW", 23.52), ("DO", 23.94), ("YOU-", 24.10)],
    [("WE", 24.22), ("LISTEN,", 24.44), ("WE", 25.36), ("DON'T", 25.42), ("JUDGE.", 26.40)],
    [("I", 27.02), ("KILLED", 27.42), ("ABEL", 27.52)],
    [("AND", 27.86), ("I", 28.14), ("LIED.", 28.28)],
    [("CALM", 29.30), ("DOWN,", 29.52)],
    [("WE", 29.84), ("LISTEN,", 30.04), ("WE", 30.46), ("DON'T", 30.80), ("JUDGE.", 31.40)],
    [("ADAM!", 32.10)],
    [("WE", 32.76), ("DON'T", 33.00), ("JUDGE.", 33.10)],
    [("ADAM,", 33.44), ("CONFESS,", 33.88), ("CONFESS,", 34.36), ("CONFESS!", 34.78)],
    [("COME,", 34.98), ("COME,", 35.10), ("COME", 35.20), ("QUICK!", 35.38)],
    [("WE", 35.52), ("DON'T", 35.60), ("JUDGE.", 35.70)],
    [("WE", 35.98), ("JUDGE!", 36.52)],
    [("CAIN,", 37.74), ("WE", 38.40), ("JUDGE!", 38.60)],
    [("ADAM!", 40.44)],
    [("WE", 40.88), ("LISTEN,", 41.44)],
    [("WE", 41.92), ("DON'T", 42.38), ("JUDGE.", 42.64)],
    [("POTIPHAR,", 45.88)],
    [("REMEMBER", 46.32), ("ONE", 46.84), ("BOY", 47.06), ("ONE", 47.30), ("TIME", 47.54)],
    [("THAT", 47.66), ("YOU", 47.76), ("BROUGHT", 47.86), ("TO", 47.98), ("THE", 48.08), ("HOUSE", 48.24)],
    [("THAT", 48.52), ("YOU", 48.72), ("NOW", 48.86), ("MADE", 48.90), ("YOUR", 48.98), ("CHIEF", 49.16), ("SERVANT?", 49.38)],
    [("JOSEPH,", 50.70), ("THAT'S", 51.16), ("HIS", 51.42), ("NAME.", 51.54)],
    [("THAT", 51.64), ("NONSENSE", 52.18), ("BOY", 52.62)],
    [("WE", 52.78), ("TRIED", 52.92), ("TO-", 53.12)],
    [("I'M", 53.32), ("THE", 53.52), ("ONE", 53.66), ("THAT", 53.80), ("TRIED", 53.98), ("TO", 54.22), ("SAVE", 54.52), ("HIM", 54.70)],
    [("DARLING,", 58.16), ("WE", 59.02), ("DON'T", 59.10), ("JUDGE.", 59.32)],
    [("WHY", 62.00), ("WILL", 62.40), ("YOU", 62.58), ("BRING", 62.90), ("SUCH", 63.20), ("A", 63.46), ("FINE", 63.86), ("BOY", 63.96)],
    [("TO", 64.14), ("THE", 64.30), ("HOUSE", 64.44)],
    [("KNOWING", 65.04), ("THAT", 65.36), ("I'M", 65.52), ("STILL", 65.70), ("YOUNG...", 65.88)],
    [("NO", 66.46), ("NOW,", 66.70)],
    [("IT'S", 67.30), ("YOUR", 67.48), ("FAULT!", 67.68)],
    [("I", 68.34), ("WILL", 68.68), ("JUDGE", 68.82), ("YOU!", 69.06)],
    [("I", 69.60), ("HAVE", 69.80), ("NO", 69.90), ("FEAR", 70.04), ("OF", 70.16), ("MYSELF,", 70.30)],
    [("I", 71.26), ("HAVE", 71.70), ("NO", 71.85), ("FEAR", 71.95), ("OF", 72.05), ("YOU.", 72.12)],
    [("YOU", 72.30), ("GET", 72.36), ("LOCKED", 72.42), ("UP,", 72.56), ("YOU", 73.02), ("DON'T", 73.16), ("EVEN", 73.34), ("GRIEVE.", 73.58)],
    [("YOU", 73.84), ("DON'T", 73.96), ("EVEN", 74.04), ("GRIEVE.", 74.18)],
    [("WE", 74.34), ("DON'T", 74.82), ("JUDGE.", 76.38)],
    [("WE", 76.92), ("DON'T", 77.52), ("JUDGE.", 77.76)],
    [("THAT'S", 78.02), ("HOW", 78.30), ("THE", 78.44), ("GAME", 78.64), ("IS.", 78.76)],
]
rf._cap.clear()

# speaker: "L" = left seat (Cain / Potiphar), "R" = right seat (Eve / wife), "LR" = both
SPEAKERS = [(0, 3.7, "LR"), (5.5, 9.4, "R"), (9.4, 9.9, "L"), (9.9, 21.85, "R"), (21.85, 24.2, "L"), (24.2, 25.0, "R"),
            (25.0, 29.2, "L"), (29.2, 32.1, "L"), (32.1, 35.5, "R"), (35.5, 35.95, "L"), (35.95, 40.85, "R"),
            (40.85, 41.9, "R"), (41.9, 43.4, "L"), (45.8, 50.6, "R"), (50.6, 53.3, "L"), (53.3, 56.6, "R"), (58.1, 60.0, "L"),
            (62.0, 67.2, "R"), (67.2, 69.5, "L"), (69.5, 80.4, "R")]


def speaker(t):
    for a, b, who in SPEAKERS:
        if a <= t < b:
            return who
    return ""


def frame(t, m, seed):
    img = Image.new("RGB", (W * SS, H * SS), (242, 228, 170))
    wide, cL, cR = (540, 1110, 1.35), (380, 1010, 1.75), (700, 1010, 1.75)

    def P(c, push=0.0, sx=0.0):
        return Pen(img, c[0], c[1], c[2] * (1 + push), sx=sx, seed=seed)

    sp = speaker(t)
    mL = m if "L" in sp else 0.0
    mR = m if "R" in sp else 0.0
    part2 = t >= 40.85
    left, right = (POTIPHAR, WIFE) if part2 else (CAIN, EVE)

    if t < 5.5 or (40.85 <= t < 45.85):  # title shots
        pen = P(wide, 0.04 * ease_io(((t if t < 5.5 else t - 40.85)) / 5))
        couch(pen, t, left, right, mL, mR, ls=dict(eyes="happy"), rs=dict(eyes="happy"))
        k = ease_out((t if t < 5.5 else t - 40.85) / 0.3)
        text(pen, 540, 470, "WE LISTEN,", 84 * k + 1, fill=(255, 220, 70), sw=8)
        text(pen, 540, 570, "WE DON'T JUDGE", 84 * k + 1, fill=(255, 220, 70), sw=8)
        text(pen, 540, 660, "BIBLE EDITION" if not part2 else "PART 2", 50 * k + 1, fill=(255, 255, 255), sw=6)
        names = ("POTIPHAR", "HIS WIFE") if part2 else ("CAIN", "EVE")
        tag(pen, LEFT, 760, names[0])
        tag(pen, RIGHT, 760, names[1])
    elif t < 6.36:
        couch(P(cR), t, left, right, mL, mR, rs=dict(pose="talk"))
    elif t < 9.4:
        flash_garden(P((540, 1050, 1.3)), t)
    elif t < 9.9:
        couch(P(cL, 0.05 * kick(t, 9.5, 0.15)), t, left, right, mL, mR, ls=dict(eyes="happy"))
    elif t < 12.07:
        couch(P(cR), t, left, right, mL, mR, rs=dict(pose="talk", eyes="half"))
    elif t < 14.86:
        flash_chased(P((520, 1040, 1.3)), t)
    elif t < 17.14:
        couch(P(cR), t, left, right, mL, mR, rs=dict(pose="facepalm", eyes="closed", brows="sad"))
    elif t < 21.88:
        flash_snake(P((520, 1040, 1.3)), t)
    elif t < 24.2:
        pen = P(cL, 0.12 * kick(t, 22.22, 0.2))
        couch(pen, t, left, right, mL, mR, ls=dict(pose="handshead", eyes="huge", brows="up"))
    elif t < 25.0:
        couch(P(cR), t, left, right, mL, mR, rs=dict(pose="point", eyes="half"))
    elif t < 27.02:
        couch(P(cL), t, left, right, mL, mR, ls=dict(pose="shrug", eyes="half"))
    elif t < 29.3:
        flash_abel(P((540, 1040, 1.3)), t)
    elif t < 32.1:
        pen = P(wide)
        couch(pen, t, left, right, mL, mR, ls=dict(pose="shrug", eyes="open"), rs=dict(eyes="huge", mouth="O", lean=-14, brows="up"))
    elif t < 35.5:  # ADAM! - Adam peeks in at the door
        pen = P((600, 1080, 1.1))
        couch(pen, t, left, right, mL, mR, door=True, ls=dict(eyes="huge"), rs=dict(pose="handshead", eyes="huge", lean=8))
        if t > 34.9:
            pk = ease_out((t - 34.9) / 0.3)
            char(pen, 1140 - 90 * pk, 1300, 1.4, ADAM, facing=-1, pose="rest", t=t, eyes="huge", mouth="O")
    elif t < 35.95:
        couch(P(cL), t, left, right, mL, mR, ls=dict(pose="shrug"))
    elif t < 40.85:  # WE JUDGE! - Eve stands up, gavel down
        pen = P(wide, 0.1 * kick(t, 36.52, 0.18) + 0.1 * kick(t, 38.6, 0.18))
        couch(pen, t, left, right, mL, mR, ls=dict(eyes="huge", lean=-10), rs=dict(stand=True, pose="point", eyes="half", brows="sad"))
        for t0 in (36.52, 38.6):
            if t > t0:
                k = ease_out((t - t0) / 0.15)
                text(pen, 540, 600 if t0 < 38 else 700, "WE JUDGE!", 110 * (1.6 - 0.6 * k), fill=(230, 60, 60), sw=10)
        gx, gy = RIGHT - 120, 1060
        sw = 0.6 * kick(t, 36.52, 0.12) + 0.6 * kick(t, 38.6, 0.12)
        rp = RotPen(pen, gx, gy, -0.5 + sw)
        rp.line([(gx, gy), (gx - 90, gy)], width=10, fill=(140, 95, 60))
        rp.rect(gx - 130, gy - 30, gx - 80, gy + 30, fill=(150, 100, 60), width=6)
    elif t < 50.6:
        if t < 46.3:
            couch(P(cR), t, left, right, mL, mR, rs=dict(pose="talk", eyes="half"))
        else:
            flash_joseph_arrives(P((620, 1060, 1.3)), t)
    elif t < 53.3:
        couch(P(cL), t, left, right, mL, mR, ls=dict(pose="talk", brows="sad"))
    elif t < 56.6:  # the halo of innocence
        pen = P(cR)
        (hx, hy, r), _ = couch(pen, t, left, right, mL, mR, rs=dict(pose="pray", eyes="closed"))[1]
        pen.ellipse(hx, hy - r * 1.6, r * 0.9, r * 0.22, fill=None, outline=(250, 210, 60), width=8)
    elif t < 62.0:
        couch(P(cL if t < 60 else wide), t, left, right, mL, mR, ls=dict(pose="think", eyes="half"))
    elif t < 62.9:
        couch(P(cR), t, left, right, mL, mR, rs=dict(pose="talk"))
    elif t < 66.46:
        flash_joseph_coat(P((620, 1060, 1.3)), t)
    elif t < 67.2:
        couch(P(cR), t, left, right, mL, mR, rs=dict(pose="shrug", eyes="half"))
    elif t < 69.5:
        pen = P(cL, 0.1 * kick(t, 68.82, 0.18))
        couch(pen, t, left, right, mL, mR, ls=dict(pose="point", eyes="huge", brows="sad", lean=10))
        if t > 68.82:
            text(pen, 470, 640, "I WILL JUDGE YOU!", 50, fill=(230, 60, 60), sw=7)
    elif t < 72.25:
        couch(P(cR), t, left, right, mL, mR, rs=dict(stand=True, pose="shrug", eyes="half"))
    elif t < 74.3:
        flash_jail(P((560, 1060, 1.3)), t)
    elif t < 78.0:  # Potiphar gives up: cap off, flops over on the couch
        pen = P(wide)
        living(pen, t)
        flop = ease_io((t - 75.3) / 1.0)
        rp = RotPen(pen, LEFT, HIP, -flop * 1.25, 0, 0)
        look = dict(POTIPHAR, cap=t < 75.0)
        char(rp, LEFT, HIP, S, look, facing=1, sitting=True, t=t, pose="handshead" if t < 75.3 else "rest",
             eyes="closed" if flop > 0.5 else "huge", mouth="frown")
        char(pen, RIGHT, HIP, S, WIFE, facing=-1, sitting=True, t=t, pose="talk", mouth=mR, eyes="half")
        if t >= 75.0:  # the cap falls on the floor
            k = ease_out((t - 75.0) / 0.5)
            pen.poly([(280, 1000 + 360 * k), (380, 1000 + 360 * k), (360, 960 + 360 * k), (300, 955 + 360 * k)], fill=(60, 90, 190), width=5)
    else:
        pen = P(cR, 0.05 * kick(t, 78.02, 0.2))
        (hx, hy, r), _ = couch(pen, t, left, right, mL, mR, rs=dict(pose="talk", eyes="half"))[1]
        if t > 78.6:
            text(pen, 700, 600, "WE DON'T JUDGE", 64, fill=(255, 220, 70), sw=7)
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
