"""When Her Smile Takes Away Yours - Google Flow (3D) version.

Same timeline as the chalkboard version (render.py), built from the Flow
stills in images/: camera moves, shakes, lip-sync on the reading shots,
drawn overlays and word-timed captions. Audio is the original clip.

Usage: python3 render_flow.py <original_audio.wav> <output.mp4>
"""
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("chalk", HERE / "render.py")
chalk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chalk)
rf = chalk.rf
clamp, lerp, ease_io, ease_out, kick = chalk.clamp, chalk.lerp, chalk.ease_io, chalk.ease_out, chalk.kick

W, H, FPS = 1080, 1920, 30
END = chalk.END
SW, SH = 768, 1376
FONT = HERE.parent / "angel-demon-video" / "fonts" / "LuckiestGuy.ttf"

# mouth centre + radii (source px) for the lip-sync pairs
MOUTHS = {"s1": (266, 650, 30, 24), "s2": (411, 676, 62, 44)}

YEL, PINK, RED, BLUE, WHITE, INK = (255, 214, 0), (255, 120, 170), (240, 60, 60), (120, 190, 255), (255, 255, 255), (30, 20, 40)


# ------------------------------------------------------------------ helpers


def load():
    imgs = {}
    for p in (HERE / "images").glob("*.jpg"):
        imgs[p.stem] = np.asarray(Image.open(p).convert("RGB").resize((SW, SH), Image.LANCZOS), np.float32)
    for s, (mx, my, rx, ry) in MOUTHS.items():
        mask = np.zeros((SH, SW), np.float32)
        cv2.ellipse(mask, (mx, my), (rx, ry), 0, 0, 360, 1.0, -1)
        imgs[s + "-mask"] = cv2.GaussianBlur(mask, (0, 0), 9)[..., None]
    return imgs


def talking(imgs, s, o):
    c, op, m = imgs[s + "-closed"], imgs[s + "-open"], imgs[s + "-mask"]
    return c * (1 - m * o) + op * (m * o) if o > 0 else c


def cam(img, cx, cy, zoom, rot=0.0, sx=0.0, sy=0.0):
    vw = SW / zoom
    vh = vw * H / W
    s = W / vw
    cx = clamp(cx - sx / s, vw / 2, SW - vw / 2)
    cy = clamp(cy - sy / s, vh / 2, SH - vh / 2)
    m = cv2.getRotationMatrix2D((cx, cy), rot, s)
    m[:, 2] += np.array([W / 2 - cx, H / 2 - cy])
    out = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    def proj(px, py):
        v = m @ np.array([px, py, 1.0])
        return float(v[0]), float(v[1])

    return out, proj


def shake(t, t0, amp, decay=0.18):
    k = kick(t, t0, decay) * amp
    return k * math.sin(t * 71), k * math.cos(t * 53)


_f = {}


def font(px):
    px = max(6, int(px))
    return _f.get(px) or _f.setdefault(px, ImageFont.truetype(str(FONT), px))


def txt(d, x, y, s, size, fill=WHITE, rot=0.0):
    f = font(size)
    w = f.getlength(s)
    d.text((x - w / 2, y - size * 0.55), s, font=f, fill=fill, stroke_width=max(2, int(size / 10)), stroke_fill=INK)


def pop(t, t0, d=0.18):
    return lerp(0.4, 1.0, rf.back_out((t - t0) / d)) if t >= t0 else 0.0


def bubble(d, x, y, w, h, tx, ty, lines, size=44, fill=INK):
    d.rounded_rectangle((x - w / 2, y - h / 2, x + w / 2, y + h / 2), radius=36, fill=(255, 255, 255), outline=INK, width=6)
    d.polygon([(x - 30, y + h / 2 - 4), (x + 30, y + h / 2 - 4), (tx, ty)], fill=(255, 255, 255), outline=None)
    d.line([(x - 30, y + h / 2), (tx, ty), (x + 30, y + h / 2)], fill=INK, width=6)
    f = font(size)
    for i, line in enumerate(lines):
        lw = f.getlength(line)
        d.text((x - lw / 2, y - h / 2 + 22 + i * size * 1.15 + (h - 44 - size * 1.15 * len(lines)) / 2), line, font=f, fill=fill)


def heart(d, x, y, r, fill):
    pts = []
    for i in range(30):
        a = 2 * math.pi * i / 30
        pts.append((x + r * math.sin(a) ** 3, y - r * (13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)) / 16))
    d.polygon(pts, fill=fill, outline=INK)


def has(d, t, x, y, k=0.0, size=90):
    if (t * 3 + k) % 1 < 0.65:
        txt(d, x + 8 * math.sin(t * 20 + k), y, "HA", size, YEL)


def hahaha(d, t, y=360):
    if int(t * 4) % 2 == 0:
        txt(d, W / 2, y, "HAHAHAHA", 120, YEL)


# ------------------------------------------------------------------ shots


def frame(imgs, t, m):
    sp = chalk.speaker(t)
    mA = m if sp == "A" else 0.0
    mB = m if sp == "B" else 0.0
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    flash = 0.0

    def push(name, t0, t1, c0, c1, z0, z1, src=None, sx=0.0, sy=0.0):
        k = ease_io((t - t0) / max(0.01, t1 - t0))
        return cam(imgs[name] if src is None else src, lerp(c0[0], c1[0], k), lerp(c0[1], c1[1], k), lerp(z0, z1, k), sx=sx, sy=sy)

    if t < 1.2:
        f, P = push("s1", 0, 1.2, (384, 760), (330, 700), 1.0, 1.25, talking(imgs, "s1", mA))
    elif t < 4.3:  # bored on the bench + love meter draining
        f, P = push("story01", 1.2, 4.3, (384, 800), (330, 640), 1.0, 1.3)
        k = ease_io((t - 1.6) / 2.4)
        d.rounded_rectangle((240, 250, 840, 330), radius=20, fill=(255, 255, 255, 230), outline=INK, width=6)
        fw = 580 * (1 - k)
        if fw > 6:
            d.rounded_rectangle((250, 260, 250 + fw, 320), radius=14, fill=PINK if k < 0.6 else RED)
        txt(d, 540, 205, "LOVE METER", 60, PINK)
        heart(d, 540, 450, 70 * (1 - 0.4 * k), PINK)
        if k > 0.7:
            d.line([(520, 400), (555, 440), (528, 475), (562, 510)], fill=INK, width=9)
    elif t < 5.2:
        f, P = push("s2", 4.3, 5.2, (384, 700), (400, 690), 1.05, 1.2, talking(imgs, "s2", mB))
    elif t < 7.5:  # holding hands: CUT HERE
        f, P = push("story02", 5.2, 7.5, (384, 760), (400, 900), 1.0, 1.35)
        if t > 6.6:
            hx, hy = P(415, 1050)
            for k in range(7):
                d.line([(hx - 15 + k * 0, hy - 140 + k * 40), (hx - 15, hy - 120 + k * 40)], fill=YEL, width=10)
            txt(d, hx, hy - 200, "CUT HERE", 70 * pop(t, 6.6), YEL)
            sx = hx + 120 + 12 * math.sin(t * 14)
            d.line([(sx, hy - 20), (sx + 90, hy + 40)], fill=WHITE, width=12)
            d.line([(sx, hy + 40), (sx + 90, hy - 20)], fill=WHITE, width=12)
            for yy in (hy - 35, hy + 55):
                d.ellipse((sx + 85, yy - 22, sx + 129, yy + 22), outline=RED, width=10)
    elif t < 12.1:  # laugh 1
        if t < 10.3:
            sx, sy = shake(t, 8.0, 40)
            f, P = push("s3", 7.5, 10.3, (384, 760), (330, 700), 1.0, 1.15, sx=sx, sy=sy)
            if 7.9 < t < 8.7:
                txt(d, 300, 500, "CRASH!", 140 * pop(t, 7.9), YEL)
            has(d, t, 800, 600, 0.3)
        else:
            f, P = push("s4", 10.3, 12.1, (384, 700), (384, 640), 1.05, 1.2)
            has(d, t, 780, 420, 0.0)
            has(d, t, 300, 520, 0.5, 70)
    elif t < 15.3:  # what's the problem
        if t < 13.0:
            f, P = push("s1", 12.1, 13.0, (300, 680), (280, 660), 1.4, 1.55, talking(imgs, "s1", mA))
        else:
            f, P = push("story03", 13.0, 15.3, (384, 760), (330, 820), 1.0, 1.15)
            bubble(d, 360, 420, 620, 140, 300, 640, ["WHAT'S THE PROBLEM?"], 50)
            if t > 14.22:
                s = pop(t, 14.22)
                txt(d, 760, 640, "THE PROBLEM", 80 * s, YEL)
                d.line([(700, 700), (420, 900)], fill=YEL, width=16)
                d.polygon([(400, 870), (450, 930), (390, 940)], fill=YEL)
    elif t < 17.3:  # laugh 2
        sx, sy = shake(t, 15.3, 18)
        f, P = push("s4", 15.3, 17.3, (500, 800), (520, 760), 1.25, 1.4, sx=sx, sy=sy)
        has(d, t, 760, 360, 0.2, 110)
    elif t < 21.3:  # her battery died
        if t < 18.0:
            f, P = push("s2", 17.3, 18.0, (400, 700), (410, 690), 1.2, 1.3, talking(imgs, "s2", mB))
        elif t < 20.2:
            f, P = push("story04", 18.0, 20.2, (384, 800), (384, 700), 1.0, 1.15)
            for k in range(3):
                a = ((t - 18) * 0.6 + k / 3) % 1
                txt(d, 620 + 120 * a, 640 - 300 * a, "~", 70, YEL)
        else:
            f, P = push("story05", 20.2, 21.3, (384, 900), (384, 960), 1.1, 1.4)
            x0, y0 = P(300, 885)
            x1, y1 = P(455, 1100)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            d.rounded_rectangle((cx - 70, cy - 110, cx + 70, cy - 20), radius=10, outline=RED, width=10)
            d.rectangle((cx + 70, cy - 85, cx + 88, cy - 45), fill=RED)
            if int(t * 4) % 2:
                d.rectangle((cx - 55, cy - 95, cx - 35, cy - 35), fill=RED)
            txt(d, cx, cy + 50, "0%", 90 * pop(t, 20.5), RED)
            txt(d, W / 2, 200, "BATTERY DEAD", 90 * pop(t, 20.86), RED)
    elif t < 23.1:  # laugh 3
        f, P = push("s5", 21.3, 23.1, (384, 980), (384, 900), 1.0, 1.15, sy=8 * math.sin(t * 18))
        hahaha(d, t)
    elif t < 28.8:  # silent treatment
        if t < 23.9:
            f, P = push("s1", 23.1, 23.9, (300, 680), (280, 660), 1.4, 1.55, talking(imgs, "s1", mA))
        else:
            k = ease_io((t - 26.6) / 0.8)
            f, P = cam(imgs["story06"], lerp(300, 520, k), lerp(900, 780, k), lerp(1.35, 1.6, k))
            if t > 26.8:
                bx, by = P(540, 540)
                txt(d, bx, by - 60, "...", 110 * pop(t, 26.8))
            if t > 27.6:
                txt(d, W / 2, 330, "SILENT", 120 * pop(t, 27.6), PINK)
                txt(d, W / 2, 460, "TREATMENT", 120 * pop(t, 27.8), PINK)
    elif t < 31.3:  # laugh 4
        sx, sy = shake(t, 28.8, 25)
        f, P = push("s3", 28.8, 31.3, (520, 760), (560, 800), 1.3, 1.45, sx=sx, sy=sy + 6 * math.sin(t * 17))
        has(d, t, 300, 420, 0.0, 100)
        has(d, t, 780, 520, 0.5, 80)
    elif t < 35.6:  # talk to me / it's fine
        if t < 32.0:
            f, P = push("s2", 31.3, 32.0, (410, 700), (410, 690), 1.25, 1.35, talking(imgs, "s2", mB))
        else:
            f, P = push("story07", 32.0, 35.6, (384, 820), (420, 760), 1.05, 1.2)
            bubble(d, 330, 380, 440, 130, 300, 600, ["TALK TO ME..."], 54, (200, 60, 110))
            # effort meter
            d.rounded_rectangle((900, 420, 980, 820), radius=14, fill=(255, 255, 255, 220), outline=INK, width=6)
            lvl = 0.12 if t > 33.68 else 0.65
            d.rounded_rectangle((910, 810 - 380 * lvl, 970, 810), radius=10, fill=(90, 200, 110) if lvl > 0.3 else RED)
            txt(d, 940, 370, "EFFORT", 46, YEL)
            if t > 34.84:
                bubble(d, 720, 950, 380, 130, 760, 1080, ["IT'S FINE."], 60)
    elif t < 37.4:  # OHHHH
        f, P = push("s6", 35.6, 37.4, (384, 760), (384, 700), 1.05, 1.25, sx=shake(t, 35.8, 14)[0])
        txt(d, W / 2, 330, "OHHHH!", 150 * pop(t, 35.8), YEL)
    elif t < 42.7:  # I'll call you / remove the battery
        if t < 39.6:
            f, P = push("story08", 37.4, 39.6, (384, 860), (330, 900), 1.05, 1.3)
            bubble(d, 420, 380, 700, 190, 330, 640, ["I'LL CALL YOU WHEN", "I GET HOME!"], 54, (200, 60, 110))
        else:
            f, P = push("story09", 39.6, 42.7, (384, 760), (430, 860), 1.05, 1.4)
            if t > 40.1:
                txt(d, 760, 420, "PHONE: OFF", 80 * pop(t, 40.1), RED)
            if t > 41.0:
                txt(d, W / 2, 560, "BATTERY: GONE", 90 * pop(t, 41.0), YEL)
    elif t < 45.5:  # laugh 5
        sx, sy = shake(t, 42.9, 40)
        f, P = push("s3", 42.7, 45.5, (300, 760), (260, 720), 1.25, 1.45, sx=sx, sy=sy)
        if 42.9 < t < 43.6:
            txt(d, 330, 480, "CRASH!", 140 * pop(t, 42.9), YEL)
        has(d, t, 800, 620, 0.4)
    elif t < 47.4:  # her smile takes yours
        if t < 46.0:
            f, P = push("s2", 45.5, 46.0, (410, 700), (410, 690), 1.25, 1.35, talking(imgs, "s2", mB))
        else:
            f, P = push("story10", 46.0, 47.4, (384, 780), (384, 760), 1.0, 1.15)
            k = ease_io((t - 46.1) / 0.8)
            a0, a1 = P(560, 820), P(220, 820)
            sx, sy = lerp(a0[0], a1[0], k), lerp(a0[1], a1[1], k) - 260 * math.sin(math.pi * k)
            d.arc((sx - 60, sy - 70, sx + 60, sy + 30), 20, 160, fill=YEL, width=14)
            if t > 46.5:
                txt(d, a1[0] + 60, 380, "+1 SMILE", 70 * pop(t, 46.5), (90, 220, 120))
                txt(d, a0[0], 470, "-1 SMILE", 70 * pop(t, 46.7), RED)
    elif t < 50.0:  # laugh 6
        f, P = push("s5", 47.4, 50.0, (384, 900), (384, 960), 1.15, 1.35, sy=8 * math.sin(t * 18))
        hahaha(d, t, 420)
    elif t < 53.4:  # kisses / licked by a dog
        if t < 50.6:
            f, P = push("s1", 50.0, 50.6, (290, 670), (280, 660), 1.45, 1.55, talking(imgs, "s1", mA))
        elif t < 51.75:
            f, P = push("story11", 50.6, 51.75, (384, 760), (400, 700), 1.05, 1.25)
            for k in range(4):
                a = ((t - 50.6) * 0.9 + k / 4) % 1
                heart(d, 330 + 60 * math.sin(a * 6 + k), 700 - 400 * a, 34, PINK)
        else:
            sx, sy = shake(t, 51.75, 16)
            f, P = push("story12", 51.75, 53.4, (384, 700), (420, 640), 1.1, 1.35, sx=sx, sy=sy)
            txt(d, W / 2, 330, "SLURP!", (130 + 10 * math.sin(t * 20)) * pop(t, 51.8), PINK)
    elif t < 56.0:  # WOW / BRO
        z = 1.05 + 0.12 * kick(t, 54.54, 0.2) + 0.12 * kick(t, 55.4, 0.2)
        f, P = cam(imgs["s7"], 384, 720, z)
        if t > 54.54:
            txt(d, W / 2, 330, "WOW!" if t < 55.4 else "BRO!", 160 * pop(t, 54.54 if t < 55.4 else 55.4), YEL)
    elif t < 60.6:  # she said she was coming... I cried
        if t < 56.9:
            f, P = push("s2", 56.0, 56.9, (410, 700), (410, 690), 1.25, 1.35, talking(imgs, "s2", mB))
        elif t < 58.9:
            f, P = push("story13", 56.9, 58.9, (384, 900), (384, 860), 1.05, 1.35)
            if t > 58.2:
                bubble(d, W / 2, 400, 700, 190, 560, 620, ["I'M COMING", "THIS WEEKEND <3"], 60, (200, 60, 110))
        else:
            f, P = push("story14", 58.9, 60.6, (384, 620), (384, 600), 1.32, 1.45, sy=5 * math.sin(t * 25))
            txt(d, W / 2, 330, "I CRIED", 150 * pop(t, 59.6), BLUE)
    else:  # final laugh
        f, P = push("s5", 60.6, END, (384, 960), (384, 900), 1.2, 1.35, sy=8 * math.sin(t * 18))
        hahaha(d, t, 420)
    return f, layer, flash


def main(audio_path, out):
    a, sr = rf.load_audio(audio_path)
    rf.END = END
    track = rf.talk_track(a, sr)
    rf.CAPTIONS[:] = [list(c) for c in chalk.CAPTIONS]
    rf._cap.clear()
    imgs = load()
    tmp = out.with_suffix(".video.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
         "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
    n = int(round(END * FPS))
    for i in range(n):
        t = i / FPS
        f, layer, _ = frame(imgs, t, track[i])
        rf.draw_caption(layer, t)
        base = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
        base.alpha_composite(layer)
        px = np.asarray(base.convert("RGB"), np.float32)
        if t > END - 0.4:
            px *= clamp((END - t) / 0.4, 0, 1)
        ff.stdin.write(px.astype(np.uint8).tobytes())
        if i % 300 == 0:
            print(f"frame {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-i", str(audio_path), "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-t", str(END), "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink()
    print("done", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
