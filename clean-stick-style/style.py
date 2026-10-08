"""Clean cartoon stickman style (white round heads, bold outlines, pastel flat
backgrounds, red-box word captions, POV title card).

Drawing helpers work in a 1080x1920 "world" and are rendered through a camera
(cx, cy, zoom) with 2x supersampling for smooth lines.
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, SS = 1080, 1920, 2
INK = (25, 25, 30)
WHITE = (255, 255, 255)
HERE = Path(__file__).parent
CAP_FONT = HERE.parent / "angel-demon-video" / "fonts" / "Bangers.ttf"
TITLE_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class Cam:
    def __init__(self, img, cx=540, cy=960, z=1.0):
        self.d = ImageDraw.Draw(img)
        self.cx, self.cy, self.z = cx, cy, z

    def p(self, x, y):
        return ((x - self.cx) * self.z + W / 2) * SS, ((y - self.cy) * self.z + H / 2) * SS

    def w(self, v):
        return max(1, int(v * self.z * SS))

    def line(self, pts, width=8, fill=INK):
        q = [self.p(*p) for p in pts]
        self.d.line(q, fill=fill, width=self.w(width), joint="curve")
        r = self.w(width) / 2
        for x, y in (q[0], q[-1]):
            self.d.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    def ellipse(self, x, y, rx, ry, fill=None, outline=INK, width=6):
        (x0, y0), (x1, y1) = self.p(x - rx, y - ry), self.p(x + rx, y + ry)
        self.d.ellipse((x0, y0, x1, y1), fill=fill, outline=outline, width=self.w(width) if outline else 0)

    def poly(self, pts, fill, outline=INK, width=5):
        q = [self.p(*p) for p in pts]
        self.d.polygon(q, fill=fill)
        if outline:
            self.d.line(q + [q[0]], fill=outline, width=self.w(width), joint="curve")

    def rect(self, x0, y0, x1, y1, fill, outline=INK, width=5, r=0):
        (a, b), (c, d) = self.p(x0, y0), self.p(x1, y1)
        if r:
            self.d.rounded_rectangle((a, b, c, d), radius=self.w(r), fill=fill, outline=outline, width=self.w(width) if outline else 0)
        else:
            self.d.rectangle((a, b, c, d), fill=fill, outline=outline, width=self.w(width) if outline else 0)

    def arc(self, x, y, rx, ry, a0, a1, width=5, fill=INK):
        (x0, y0), (x1, y1) = self.p(x - rx, y - ry), self.p(x + rx, y + ry)
        self.d.arc((x0, y0, x1, y1), a0, a1, fill=fill, width=self.w(width))


# ------------------------------------------------------------------ background


def living_room(c):
    """Pastel flat living room with a city window, lamps, plants and two sofas."""
    c.rect(-600, -600, 1700, 1250, (214, 232, 222), outline=None)          # wall
    c.rect(-600, 1250, 1700, 2600, (196, 226, 214), outline=None)          # floor
    for k in range(9):
        c.line([(-400 + k * 260, 1330 + (k % 3) * 120), (-340 + k * 260, 1330 + (k % 3) * 120)], width=4, fill=(175, 205, 192))
    c.rect(-600, 1240, 1700, 1262, (240, 236, 226), outline=None)          # skirting
    # big window with skyline
    c.rect(-80, 380, 520, 1060, (232, 243, 246), width=8)
    for k, (bx, bh) in enumerate(((-40, 420), (40, 330), (130, 470), (230, 300), (320, 390), (420, 260))):
        c.rect(bx, 1060 - bh, bx + 80, 1060, (200, 222, 232), outline=(170, 196, 210), width=3)
        for r in range(int(bh / 50)):
            c.rect(bx + 15, 1060 - bh + 20 + r * 50, bx + 30, 1060 - bh + 40 + r * 50, (230, 240, 245), outline=None)
    c.line([(220, 380), (220, 1060)], width=7, fill=(240, 236, 226))
    c.line([(-80, 720), (520, 720)], width=7, fill=(240, 236, 226))
    # framed painting
    c.rect(640, 520, 1000, 820, (247, 241, 220), width=7)
    c.poly([(670, 790), (760, 640), (850, 760), (900, 700), (970, 790)], (150, 190, 160), outline=None)
    c.ellipse(920, 600, 30, 30, (250, 220, 120), outline=None)
    # floor lamps
    for lx in (560, 1040):
        c.line([(lx, 1240), (lx, 900)], width=6)
        c.poly([(lx - 60, 900), (lx + 60, 900), (lx + 40, 800), (lx - 40, 800)], (250, 244, 225), width=5)
    # plant
    c.rect(600, 1120, 680, 1240, (230, 225, 215), width=5)
    for k in range(7):
        a = math.radians(-160 + k * 23)
        c.ellipse(640 + math.cos(a) * 80, 1080 + math.sin(a) * 90, 40, 18, (110, 170, 110), width=4)


def sofa(c, x0, x1, y=1150):
    c.rect(x0, y - 200, x1, y, (246, 240, 222), width=6, r=30)             # back
    c.rect(x0 - 30, y - 60, x1 + 30, y + 90, (250, 245, 230), width=6, r=24)  # seat
    for x in (x0 - 50, x1 + 10):
        c.rect(x, y - 120, x + 60, y + 90, (243, 236, 216), width=6, r=24)  # arms
    c.line([(x0, y + 90), (x0, y + 120)], width=8)
    c.line([(x1, y + 90), (x1, y + 120)], width=8)
    c.rect(x0 + 40, y - 150, x0 + 170, y - 60, (190, 220, 210), width=5, r=20)  # cushion


# ------------------------------------------------------------------ characters

BRO = dict(head=WHITE, hair=(245, 200, 60), body=WHITE)
GENIE = dict(head=(40, 110, 230), hair=None, body=(40, 110, 230), turban=True)


def stickman(c, x, hip, look, facing=1, mouth=0.0, eyes="open", brows="flat", pose="rest", sit=True, s=1.0, t=0.0):
    r = 72 * s
    neck = (x, hip - 170 * s)
    hx, hy = x, neck[1] - r * 0.9
    # body: white tube with outline (or genie tail)
    if look.get("turban"):
        c.poly([(x - 40 * s, neck[1] + 10), (x + 40 * s, neck[1] + 10), (x + 50 * s, hip), (x + 140 * s * facing, hip + 30 * s),
                (x - 60 * s, hip + 40 * s)], look["body"], width=6)
    else:
        c.rect(x - 26 * s, neck[1] - 4 * s, x + 26 * s, hip + 6 * s, look["body"], width=6 * s, r=10 * s)
    # legs
    if sit and not look.get("turban"):
        for dx in (-6, 10):
            kx, ky = x + facing * 95 * s + dx, hip - 8 * s
            c.line([(x, hip), (kx, ky), (kx + facing * 8 * s, hip + 150 * s)], width=7 * s)
    # arms
    arms = {"rest": [(-55, 120), (60, 110)], "chin": [(-30, 120), (25, -10)], "point": [(-50, 120), (150, -10)],
            "shrug": [(-110, 20), (110, 20)], "cross": [(45, 60), (-45, 60)]}[pose]
    for ax, ay in arms:
        ex, ey = x + facing * ax * 0.5 * s, neck[1] + (ay * 0.45 + 30) * s
        c.line([(x, neck[1] + 20 * s), (ex, ey), (x + facing * ax * s, neck[1] + ay * s)], width=7 * s)
    # head
    c.ellipse(hx, hy, r, r * 0.95, look["head"], width=7 * s)
    if look.get("hair"):
        for k in range(5):
            a = math.radians(-150 + k * 22)
            px, py = hx + math.cos(a) * r * 0.85, hy + math.sin(a) * r * 0.8
            c.poly([(px - 18 * s, py + 10 * s), (px + 18 * s, py + 10 * s), (px + 8 * s * facing, py - 34 * s)], look["hair"], width=4)
    if look.get("turban"):
        c.poly([(hx - r * 1.0, hy - r * 0.15), (hx - r * 0.95, hy - r * 0.75), (hx - r * 0.4, hy - r * 1.25), (hx + r * 0.4, hy - r * 1.25),
                (hx + r * 0.95, hy - r * 0.75), (hx + r * 1.0, hy - r * 0.15)], (235, 235, 240), width=6)
        for k in range(3):
            c.arc(hx, hy - r * 0.2 - k * r * 0.3, r * 0.95, r * 0.35, 200, 340, width=4, fill=(190, 190, 200))
        c.ellipse(hx, hy - r * 0.75, 13 * s, 13 * s, (240, 70, 70), width=4)
        c.line([(hx, hy - r * 0.9), (hx + 8 * s, hy - r * 1.45)], width=5, fill=(245, 200, 60))
        c.ellipse(hx + facing * r * 0.98, hy + r * 0.15, 10 * s, 14 * s, (245, 200, 60), width=3)
    ex = hx + facing * r * 0.18
    if look.get("turban"):
        hy += r * 0.08
    for k, dx in enumerate((-0.32, 0.32)):
        cx, cy = ex + dx * r, hy - r * 0.05
        if eyes == "closed":
            c.arc(cx, cy, r * 0.12, r * 0.08, 20, 160, width=5 * s)
        else:
            c.ellipse(cx, cy, r * 0.09, r * 0.11, INK if look["head"] == WHITE else WHITE, outline=None)
            if look["head"] != WHITE:
                c.ellipse(cx, cy, r * 0.05, r * 0.06, INK, outline=None)
        by = cy - r * 0.3
        if brows == "flat":
            c.line([(cx - r * 0.16, by), (cx + r * 0.16, by)], width=6 * s)
        elif brows == "skeptic":
            tilt = r * 0.12 if k == (0 if facing > 0 else 1) else -r * 0.06
            c.line([(cx - r * 0.16, by + tilt), (cx + r * 0.16, by - tilt * 0.3)], width=6 * s)
        elif brows == "up":
            c.arc(cx, by + r * 0.05, r * 0.16, r * 0.1, 200, 340, width=6 * s)
    mx, my = hx + facing * r * 0.15, hy + r * 0.45
    if mouth > 0.1:
        c.ellipse(mx, my, r * 0.2, r * (0.06 + 0.2 * mouth), (90, 40, 40), width=5 * s)
        c.ellipse(mx, my + r * 0.07 * mouth, r * 0.11, r * 0.06 * mouth, (230, 110, 110), outline=None)
    else:
        c.line([(mx - r * 0.12, my), (mx + r * 0.12, my - r * 0.02)], width=5 * s)


# ------------------------------------------------------------------ text


def pov_title(img, text1, text2):
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(TITLE_FONT, 50 * SS)
    w = max(f.getlength(text1), f.getlength(text2)) + 60 * SS
    x0 = (W * SS - w) / 2
    d.rounded_rectangle((x0, 150 * SS, x0 + w, 300 * SS), radius=16 * SS, fill=WHITE)
    for i, tx in enumerate((text1, text2)):
        d.text(((W * SS - f.getlength(tx)) / 2, (165 + i * 64) * SS), tx, font=f, fill=INK)


def caption(img, words, active, y=1500):
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(str(CAP_FONT), 92 * SS)
    sp = f.getlength(" ")
    tw = sum(f.getlength(w) for w in words) + sp * (len(words) - 1)
    x = (W * SS - tw) / 2
    for i, w in enumerate(words):
        ww = f.getlength(w)
        if i == active:
            d.rounded_rectangle((x - 10 * SS, y * SS - 4 * SS, x + ww + 10 * SS, y * SS + 100 * SS), radius=10 * SS, fill=(200, 20, 50))
        d.text((x, y * SS), w, font=f, fill=WHITE, stroke_width=5 * SS, stroke_fill=INK)
        x += ww + sp


if __name__ == "__main__":
    import sys
    img = Image.new("RGB", (W * SS, H * SS))
    c = Cam(img, 540, 1000, 1.0)
    living_room(c)
    sofa(c, -60, 300)
    sofa(c, 520, 1060)
    stickman(c, 150, 1110, GENIE, facing=1, mouth=0.0, brows="flat")
    stickman(c, 700, 1110, BRO, facing=-1, mouth=0.7, brows="skeptic", pose="chin")
    pov_title(img, "POV: When you have", "three wishes")
    caption(img, ["HA!", "OKAY.", "UM."], 1)
    img.resize((W, H), Image.LANCZOS).save(sys.argv[1], quality=92)
