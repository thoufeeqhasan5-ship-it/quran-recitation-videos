#!/usr/bin/env python3
"""
Qur'an - The Recitation: makes one English Qur'an story short (1080x1920 MP4).

  python scripts/make_video.py --topic 01
  python scripts/make_video.py --topic 01 --voice am_michael --no-recitation
  python scripts/make_video.py --topic 01 --test     # no internet: tone instead of voice, placeholder photos and ayah

Voice:   Kokoro-82M (Apache-2.0), English narrator voices (bm_george default).
Ayah:    Arabic (Uthmani), English translation and the recitation are fetched from api.alquran.cloud
         at run time, so the verse is never typed by hand.
Photos:  Pixabay (if PIXABAY_KEY is set) and Wikimedia Commons, landscapes/nature only;
         photos tagged with people, statues or religious images are skipped.
Output:  out/video.mp4, out/voice.wav, out/caption.txt, out/credits.txt
"""
import argparse, json, math, os, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK, OUT = ROOT / "work", ROOT / "out"
FPS, W, H = 30, 1080, 1920
LEAD, GAP, END_CARD, REC_GAP = 0.45, 0.35, 2.8, 0.55
UA = {"User-Agent": "QuranTheRecitationBot/1.0 (https://github.com; Qur'an story shorts)"}
OK_LICENCE = re.compile(r"^(cc0|public domain|pd|cc by(-sa)?( \d\.\d)?( [a-z\-]+)?|cc-by(-sa)?-\d\.\d.*)$", re.I)
API = "https://api.alquran.cloud/v1"
RECITERS = {"ar.alafasy": "Mishary Rashid Alafasy", "ar.husary": "Mahmoud Khalil Al-Husary",
            "ar.minshawi": "Mohamed Siddiq Al-Minshawi", "ar.abdulbasitmurattal": "Abdul Basit Abdus Samad",
            "ar.mahermuaiqly": "Maher Al-Muaiqly"}
# photos whose tags/titles contain these words are skipped (no people, no statues or other religions' images)
AVOID = re.compile(r"\b(people|person|persons|man|men|woman|women|girl|girls|boy|boys|child|children|kid|kids|baby|"
                   r"face|portrait|model|selfie|couple|family|human|crowd|bride|groom|lady|"
                   r"statue|sculpture|idol|buddha|jesus|christ|church|cross|crucifix|hindu|temple|deity|angel|"
                   r"nude|bikini|sexy|alcohol|wine|beer|pig|halloween|christmas)\b", re.I)


def log(*a):
    print("[quran]", *a, flush=True)


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode:
        log("FAILED:", " ".join(map(str, cmd))[:400]); log(r.stderr[-2500:]); sys.exit(1)
    return r.stdout


def dur(p):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]).strip())


# ---------------------------------------------------------------- the ayah (fetched, never typed)
def fetch_ayah(ref, translation, reciter, test=False):
    if not re.fullmatch(r"\d{1,3}:\d{1,3}", ref):
        log(f"bad ayah reference '{ref}' (use surah:ayah, e.g. 21:87)"); sys.exit(1)
    if test:
        return {"ref": ref, "arabic": "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
                "english": "Test translation text. The real English translation of this ayah is fetched when the video is made, "
                           "so it can be long like this sentence and still has to fit inside the panel.",
                "translator": "Test translation", "surah": "Test", "surah_no": 1, "ayah_no": 1, "audio": None}
    import requests
    eds = ",".join(["quran-uthmani", translation] + ([reciter] if reciter else []))
    data = None
    for attempt in range(4):
        try:
            r = requests.get(f"{API}/ayah/{ref}/editions/{eds}", headers=UA, timeout=60)
            r.raise_for_status()
            data = r.json()["data"]; break
        except Exception as e:
            log(f"ayah {ref}: fetch failed ({str(e)[:120]}), retrying"); time.sleep(4)
    if not data:
        log(f"could not fetch ayah {ref} from alquran.cloud"); sys.exit(1)
    by = {d["edition"]["identifier"]: d for d in data}
    ar, en = by["quran-uthmani"], by[translation]
    arabic = ar["text"].strip()
    # the first ayah of a surah carries the Bismillah in this text; it is not part of the ayah (except Al-Fatiha)
    if ar["numberInSurah"] == 1 and ar["surah"]["number"] not in (1, 9) and arabic.startswith("بِسْمِ"):
        words = arabic.split()
        if len(words) > 4:
            arabic = " ".join(words[4:])
    out = {"ref": ref, "arabic": arabic, "english": re.sub(r"\s+", " ", en["text"]).strip(),
           "translator": en["edition"]["englishName"], "surah": ar["surah"]["englishName"],
           "surah_no": ar["surah"]["number"], "ayah_no": ar["numberInSurah"],
           "audio": (by.get(reciter) or {}).get("audio") if reciter else None}
    log(f"ayah {ref} ({out['surah']}): {out['english'][:90]}…")
    return out


def spoken(text):
    """What the narrator reads: brackets of the translation removed, ellipses become pauses."""
    t = text.replace("[", "").replace("]", "").replace("…", ",").replace("ﷺ", "peace be upon him")
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------- voice
def expected_seconds(text):
    return max(1.0, len(text.split()) / 2.5)          # ~150 words per minute


def tts_kokoro(texts, voice, outdir):
    import numpy as np, soundfile as sf, torch
    from kokoro import KPipeline
    torch.set_num_threads(os.cpu_count() or 2)
    speed = float(os.getenv("VOICE_SPEED", "0.92"))
    log(f"loading Kokoro, voice {voice}, speed {speed}")
    pipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M")
    files = []
    for i, text in enumerate(texts):
        t0 = time.time()
        parts = []
        for r in pipe(text, voice=voice, speed=speed):
            a = r.audio if hasattr(r, "audio") else r[2]
            if a is not None:
                parts.append(a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a))
        audio = np.concatenate(parts).astype(np.float32) if parts else np.zeros(2400, np.float32)
        raw = outdir / f"raw_{i:02d}.wav"
        sf.write(raw, audio, 24000)
        final = outdir / f"phrase_{i:02d}.wav"
        trim(raw, final)
        log(f"phrase {i + 1}/{len(texts)}: {dur(final):.1f}s in {time.time() - t0:.0f}s")
        files.append(final)
    return files


def trim(src, dst, extra=""):
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af",
         "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
         "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.08,areverse,"
         f"{extra}aresample=44100", "-ac", "1", dst])


def tts_test(texts, outdir):
    files = []
    for i, text in enumerate(texts):
        f = outdir / f"phrase_{i:02d}.wav"
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             f"sine=frequency={180 + 20 * (i % 4)}:sample_rate=44100:d={expected_seconds(text):.2f}",
             "-af", "tremolo=f=4:d=0.6,volume=0.4", "-ac", "1", f])
        files.append(f)
    return files


def fetch_recitation(url, idx, outdir, test=False):
    dst = outdir / f"recite_{idx:02d}.wav"
    if test:
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100:d=6",
             "-af", "tremolo=f=2:d=0.5,volume=0.35", "-ac", "1", dst])
        return dst
    import requests
    src = outdir / f"recite_{idx:02d}.mp3"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=120); r.raise_for_status()
            src.write_bytes(r.content); break
        except Exception as e:
            log(f"recitation download failed ({str(e)[:120]}), retrying"); time.sleep(4)
    else:
        log("recitation unavailable, the ayah is narrated only"); return None
    trim(src, dst)
    return dst


# ---------------------------------------------------------------- photos
def clean_meta(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def commons_search(query, n=30):
    import requests
    r = requests.get("https://commons.wikimedia.org/w/api.php", headers=UA, timeout=60, params={
        "action": "query", "format": "json", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6, "gsrlimit": n, "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": 1800})
    r.raise_for_status()
    pages = (r.json().get("query") or {}).get("pages") or {}
    out = []
    for p in sorted(pages.values(), key=lambda x: x.get("index", 99)):
        ii = (p.get("imageinfo") or [{}])[0]
        md = ii.get("extmetadata") or {}
        lic = clean_meta((md.get("LicenseShortName") or {}).get("value", ""))
        artist = clean_meta((md.get("Artist") or {}).get("value", ""))
        about = p["title"] + " " + clean_meta((md.get("ImageDescription") or {}).get("value", ""))[:300]
        if ii.get("mime") not in ("image/jpeg", "image/png") or not OK_LICENCE.match(lic) or AVOID.search(about):
            continue
        if min(ii.get("width", 0), ii.get("height", 0)) < 800:
            continue
        out.append({"title": p["title"], "url": ii.get("thumburl") or ii["url"], "licence": lic,
                    "artist": artist[:60] or "Unknown", "page": ii.get("descriptionurl", ""),
                    "credit": f"Photo: {artist[:40] or 'Unknown'} · {lic} · Wikimedia Commons"})
    return out


def pixabay_search(query, n=30):
    """Pixabay photos (free, no attribution needed). Only if the secret PIXABAY_KEY is set."""
    import requests
    key = os.getenv("PIXABAY_KEY", "").strip()
    if not key:
        return []
    out = []
    for orient in ("vertical", "all"):
        r = requests.get("https://pixabay.com/api/", timeout=60, params={
            "key": key, "q": query[:100], "image_type": "photo", "orientation": orient,
            "safesearch": "true", "per_page": n, "order": "popular"})
        r.raise_for_status()
        for h in r.json().get("hits", []):
            if AVOID.search(h.get("tags", "")) or not h.get("largeImageURL"):
                continue
            out.append({"title": f"Pixabay {h['id']} ({h.get('tags', '')})", "url": h["largeImageURL"],
                        "licence": "Pixabay Content License", "artist": h.get("user") or "Pixabay",
                        "page": h.get("pageURL", ""), "credit": "Photo: Pixabay"})
    return out


def fetch_photos(shots, outdir):
    import requests
    used, got = set(), []
    for k, s in enumerate(shots):
        chosen = None
        for q in s["q"]:
            for fn in (pixabay_search, commons_search):
                try:
                    for c in fn(q):
                        if c["url"] not in used:
                            chosen = c; break
                except Exception as e:
                    log(f"{fn.__name__} failed '{q}': {str(e)[:150]}")
                if chosen:
                    break
            if chosen:
                break
        if not chosen:
            log(f"shot {k}: no suitable photo found, reusing previous")
            got.append(got[-1] if got else None); continue
        used.add(chosen["url"])
        path = outdir / f"photo_{k:02d}.jpg"
        r = requests.get(chosen["url"], headers=UA, timeout=120); r.raise_for_status()
        path.write_bytes(r.content)
        chosen["path"] = path
        log(f"shot {k}: {chosen['title'][:90]} ({chosen['licence']})")
        got.append(chosen)
    if not any(got):
        log("no photos at all were found; check the internet connection or the search words"); sys.exit(1)
    return got


def test_photos(shots, outdir):
    from PIL import Image, ImageDraw
    got = []
    for k, _ in enumerate(shots):
        p = outdir / f"photo_{k:02d}.jpg"
        im = Image.new("RGB", (1600, 1100 if k % 2 else 2000), (30 + 25 * k % 120, 60 + 15 * k % 100, 70 + 10 * k % 90))
        dr = ImageDraw.Draw(im)
        for i in range(0, 1600, 100):
            dr.line([(i, 0), (i + 600, 2000)], fill=(90, 120, 110), width=6)
        im.save(p, quality=90)
        got.append({"path": p, "title": f"Test photo {k}", "licence": "CC BY-SA 4.0", "artist": "Test",
                    "page": "", "credit": "Photo: test"})
    return got


def compose_still(src, dst):
    """Photo fitted over a blurred copy of itself, 1080x1920 (fallback only)."""
    from PIL import Image, ImageFilter, ImageEnhance
    im = Image.open(src).convert("RGB")
    s = max(W / im.width, H / im.height)
    bg = im.resize((math.ceil(im.width * s), math.ceil(im.height * s)))
    bg = bg.crop(((bg.width - W) // 2, (bg.height - H) // 2, (bg.width - W) // 2 + W, (bg.height - H) // 2 + H))
    if im.width / im.height >= 0.75:
        bg = ImageEnhance.Brightness(bg.filter(ImageFilter.GaussianBlur(40))).enhance(0.55)
        fs = min(W / im.width, 1250 / im.height)
        fg = im.resize((int(im.width * fs), int(im.height * fs)))
        bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2 - 60))
    bg.save(dst, quality=92)


# ---------------------------------------------------------------- photo -> "camera shot" (2.5D parallax)
_depth_model = None
_depth_cache = {}


def depth_map(img_bgr, key, test=False):
    """How near each pixel is (0 = far, 1 = near). Depth Anything V2 Small (Apache-2.0)."""
    import numpy as np, cv2
    if key in _depth_cache:
        return _depth_cache[key]
    h, w = img_bgr.shape[:2]
    d = None
    if test:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = 0.6 * (yy / h) + 0.4 * np.exp(-(((xx - w / 2) / (w * 0.2)) ** 2 + ((yy - h * 0.6) / (h * 0.2)) ** 2))
    else:
        global _depth_model
        if _depth_model is not False:
            try:
                from PIL import Image
                if _depth_model is None:
                    from transformers import pipeline
                    _depth_model = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
                    log("depth model loaded")
                out = _depth_model(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
                d = np.asarray(out["depth"].convert("L").resize((w, h)), dtype=np.float32)
            except Exception as e:
                log(f"depth unavailable ({str(e)[:160]}), camera moves without 3D depth")
                _depth_model = False
    if d is None:
        d = np.zeros((h, w), np.float32)
    else:
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        d = np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
        d = cv2.GaussianBlur(d, (0, 0), max(3, w / 160))
    _depth_cache[key] = d
    return d


def camera_shot(photo, seconds, idx, dst, test=False, calm=False):
    """The photo fills the screen and a slow 'camera' travels across it, near things moving more than far ones.
    calm=True (used under the ayah) makes the move gentler."""
    import numpy as np, cv2
    try:
        src = cv2.imread(str(photo))
        if src is None:
            raise RuntimeError("cannot read photo")
        sc = max(W * 1.18 / src.shape[1], H * 1.18 / src.shape[0])
        cw, ch = int(src.shape[1] * sc), int(src.shape[0] * sc)
        canvas = cv2.resize(src, (cw, ch), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        if cw > W * 3.2:
            x0 = (cw - int(W * 3.2)) // 2
            canvas = np.ascontiguousarray(canvas[:, x0:x0 + int(W * 3.2)])
        ch, cw = canvas.shape[:2]
        D = depth_map(canvas, f"{photo}", test)
        n = max(1, int(round(seconds * FPS)))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        ox, oy = xx - W / 2.0, yy - H / 2.0
        wide = cw > W * 1.6
        room_x, room_y = (cw - W) / 2.0, (ch - H) / 2.0
        mode = ["pan_r", "pan_l", "pan_r_in"][idx % 3] if wide else ["push", "rise", "push_side", "pull"][idx % 4]
        amp = 0.55 if calm else 1.0
        proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                                 "-r", str(FPS), "-i", "-", "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
                                 "-crf", "19", str(dst)], stdin=subprocess.PIPE)
        for f in range(n):
            p = f / max(1, n - 1)
            e = p * p * (3 - 2 * p)
            zoom, cx, cy, par_x, par_y, dz = 1.0, cw / 2.0, ch / 2.0, 0.0, 0.0, 0.0
            if mode in ("pan_r", "pan_l", "pan_r_in"):
                dirn = -1 if mode == "pan_l" else 1
                cx = cw / 2.0 + dirn * (-1 + 2 * e) * room_x * 0.92 * (0.7 if calm else 1)
                zoom = 1.0 + (0.10 * e if mode == "pan_r_in" else 0.0)
                par_x = dirn * (-1 + 2 * e) * 0.07 * W * amp
            elif mode == "push":
                zoom, dz = 1.0 + 0.16 * e * amp, 0.18 * e * amp
            elif mode == "rise":
                cy = ch / 2.0 + (1 - 2 * e) * room_y * 0.9 * (0.7 if calm else 1)
                par_y = (1 - 2 * e) * 0.06 * H * amp
                zoom = 1.04
            elif mode == "push_side":
                zoom, dz = 1.0 + 0.12 * e * amp, 0.12 * e * amp
                cx = cw / 2.0 + (-1 + 2 * e) * room_x * 0.8 * (0.7 if calm else 1)
                par_x = (-1 + 2 * e) * 0.05 * W * amp
            else:
                zoom, dz = 1.18 - 0.16 * e * amp, 0.18 * (1 - e) * amp
            bx, by = cx + ox / zoom, cy + oy / zoom
            Dv = cv2.remap(D, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            s3 = 1.0 + dz * Dv
            mapx = cx + ox / (zoom * s3) - par_x * Dv / zoom
            mapy = cy + oy / (zoom * s3) - par_y * Dv / zoom
            proc.stdin.write(cv2.remap(canvas, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing the camera move")
        return
    except Exception as e:
        log(f"camera move failed ({str(e)[:160]}), using a slow zoom")
    still = Path(str(dst) + ".jpg")
    compose_still(photo, still)
    n = max(1, int(round(seconds * FPS)))
    z = f"min(1.0+0.10*on/{n},1.10)" if idx % 2 == 0 else f"max(1.10-0.10*on/{n},1.0)"
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", still, "-vf",
         f"scale={W * 2}:{H * 2},zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s={W}x{H}:fps={FPS},format=yuv420p",
         "-frames:v", n, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", dst])


# ---------------------------------------------------------------- timeline
def caption_chunks(text, start, end, max_words=3):
    words = text.split()
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    if len(chunks) > 1 and len(chunks[-1].split()) == 1:
        last = chunks.pop(); chunks[-1] += " " + last
    total = sum(len(c) for c in chunks)
    out, t = [], start
    for c in chunks:
        d = (end - start) * len(c) / total
        out.append({"text": c, "start": round(t, 3), "end": round(t + d, 3)})
        t += d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--voice", default=os.getenv("VOICE") or "bm_george")
    ap.add_argument("--reciter", default=os.getenv("RECITER") or "ar.alafasy")
    ap.add_argument("--translation", default=os.getenv("TRANSLATION") or "en.sahih")
    ap.add_argument("--no-recitation", action="store_true", default=os.getenv("RECITATION", "true").lower() == "false")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    topic = json.loads((ROOT / "topics" / f"{a.topic}.json").read_text(encoding="utf-8"))
    shutil.rmtree(WORK, ignore_errors=True); WORK.mkdir(parents=True); OUT.mkdir(exist_ok=True)
    phrases, shots = topic["phrases"], topic["shots"]
    recite = not a.no_recitation
    reciter = a.reciter if recite else None

    # 1. fetch every ayah, then the voice
    ayahs = {}
    for i, ph in enumerate(phrases):
        if ph.get("ayah"):
            ay = fetch_ayah(ph["ayah"], a.translation, reciter, a.test)
            if ph.get("surah"):
                ay["surah"] = ph["surah"]
            ayahs[i] = ay
    texts = [spoken(ayahs[i]["english"]) if i in ayahs else spoken(ph.get("say", ph["text"])) for i, ph in enumerate(phrases)]
    files = tts_test(texts, WORK) if a.test else tts_kokoro(texts, a.voice, WORK)
    recs = {}
    if recite:
        for i, ay in ayahs.items():
            if phrases[i].get("recite", True) and (ay.get("audio") or a.test):
                recs[i] = fetch_recitation(ay.get("audio"), i, WORK, a.test)

    # 2. timeline + voice track
    t, segs, parts = LEAD, [], []
    for i, (ph, f) in enumerate(zip(phrases, files)):
        seg = {}
        if recs.get(i):
            d = dur(recs[i])
            parts.append((recs[i], t, 1.0))
            seg["rec"] = (t, t + d); t += d + REC_GAP
        d = dur(f)
        parts.append((f, t, 1.0))
        seg["say"] = (t, t + d)
        seg["start"], seg["end"] = seg.get("rec", seg["say"])[0], seg["say"][1]
        segs.append(seg)
        t += d + GAP + float(ph.get("pause_after", 0)) + (0.3 if i in ayahs else 0)
    voice_end = segs[-1]["end"]
    inputs, filt = [], []
    for i, (f, st, _) in enumerate(parts):
        inputs += ["-i", f]
        filt.append(f"[{i}:a]adelay={int(st * 1000)}|{int(st * 1000)}[v{i}]")
    filt.append("".join(f"[v{i}]" for i in range(len(parts))) +
                f"amix=inputs={len(parts)}:normalize=0,apad=whole_dur={voice_end + 0.2:.2f},"
                "loudnorm=I=-15:TP=-1.5:LRA=11[out]")
    run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filt), "-map", "[out]",
         "-ar", "44100", "-ac", "2", OUT / "voice.wav"])
    end_start = voice_end + 0.6
    total_dur = end_start + END_CARD

    # 3. photos -> camera shots
    photos = test_photos(shots, WORK) if a.test else fetch_photos(shots, WORK)
    shot_start = {}
    for i, ph in enumerate(phrases):
        shot_start.setdefault(ph.get("shot", 0), segs[i]["start"] if i else 0.0)
    order = sorted(shot_start.items(), key=lambda kv: kv[1])
    ayah_shots = {phrases[i].get("shot", 0) for i in ayahs}
    clips, credits = [], []
    for j, (k, st) in enumerate(order):
        en = order[j + 1][1] if j + 1 < len(order) else end_start
        p = photos[k] if k < len(photos) and photos[k] else next(x for x in photos if x)
        clip = WORK / f"clip_{j:02d}.mp4"
        camera_shot(p["path"], en - st, j, clip, a.test, calm=k in ayah_shots)
        clips.append(clip)
        credits.append({"text": p["credit"], "start": round(st, 3), "end": round(en, 3)})
    lst = WORK / "clips.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips))
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-r", FPS,
         "-vf", "noise=c0s=6:c0f=t+u,vignette=angle=PI/5",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", WORK / "photos.mp4"])

    # 4. overlay timeline
    cards, caps, ay_list = [], [], []
    for i, ph in enumerate(phrases):
        s = segs[i]
        if i in ayahs:
            ay = ayahs[i]
            ay_list.append({"start": round(s["start"], 3), "end": round(min(s["end"] + 0.7, end_start), 3),
                            "say": round(s["say"][0], 3), "recite": i in recs,
                            "arabic": ay["arabic"], "english": ay["english"],
                            "ref": f"Surah {ay['surah']} · {ay['surah_no']}:{ay['ayah_no']}",
                            "credit": f"Translation: {ay['translator']}" +
                                      (f" · Recitation: {RECITERS.get(a.reciter, a.reciter)}" if i in recs else "")})
            continue
        st, en = s["say"]
        if ph.get("card"):
            cards.append({"text": ph["card"], "start": round(st, 3), "end": round(min(en + 0.6, end_start), 3)})
        caps += caption_chunks(ph["text"], st, en)
    first_end = min(order[1][1] if len(order) > 1 else end_start, 3.4)
    hook = {"text": topic.get("hook") or phrases[0].get("card") or topic.get("title", ""),
            "sub": topic.get("hook_sub", "A story from the Qur'an"),
            "start": 0.0, "end": round(max(2.4, first_end), 3)}
    cards = [c for c in cards if c["start"] >= hook["end"] - 0.05]
    data = {"endStart": round(end_start, 3), "cards": cards, "caps": caps, "credits": credits,
            "hook": hook, "ayahs": ay_list}
    frames = WORK / "frames"; frames.mkdir()
    render_overlay(data, total_dur, frames)

    # 5. final mix (no music by default; an optional assets/ambience.m4a is mixed very low)
    amb = ROOT / "assets" / "ambience.m4a"
    fo = max(0, total_dur - 1.0)
    cmd = ["ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", WORK / "photos.mp4",
           "-framerate", FPS, "-i", frames / "%05d.png", "-i", OUT / "voice.wav"]
    v = (f"[0:v]tpad=stop_mode=clone:stop_duration=5[bg];[bg][1:v]overlay=0:0:format=auto,format=yuv420p,"
         f"fade=t=out:st={fo:.2f}:d=1[v]")
    if amb.exists():
        cmd += ["-stream_loop", "-1", "-i", amb]
        af = (f"[3:a]volume=0.10,afade=t=in:d=1,afade=t=out:st={fo:.2f}:d=1[m];"
              f"[2:a]apad[vo];[vo][m]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.9[a]")
    else:
        af = f"[2:a]apad,afade=t=out:st={fo:.2f}:d=1[a]"
    run(cmd + ["-filter_complex", v + ";" + af, "-map", "[v]", "-map", "[a]", "-t", f"{total_dur:.3f}", "-r", FPS,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
               "-movflags", "+faststart", OUT / "video.mp4"])

    caption = topic.get("caption", "")
    if ayahs:
        ay = next(iter(ayahs.values()))
        caption += f"\n\nAyah: Surah {ay['surah']} {ay['surah_no']}:{ay['ayah_no']} · Translation: {ay['translator']}"
    (OUT / "caption.txt").write_text(caption, encoding="utf-8")
    (OUT / "credits.txt").write_text("\n".join(
        [f"{p['title']} — {p['artist']} — {p['licence']} — {p['page']}" for p in photos if p] +
        [f"Ayah {ay['ref']}: text and translation ({ay['translator']}) from api.alquran.cloud" for ay in ayahs.values()] +
        ([f"Recitation: {RECITERS.get(a.reciter, a.reciter)} (cdn.islamic.network)"] if recs else [])),
        encoding="utf-8")
    log(f"done: out/video.mp4 ({total_dur:.1f}s), voice {a.voice}, recitation {'on' if recs else 'off'}")


def render_overlay(data, total, frames):
    from playwright.sync_api import sync_playwright
    n = int(math.ceil(total * FPS))
    prev, prevf, uniq, t0 = None, None, 0, time.time()
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H})
        pg.goto((ROOT / "scripts" / "overlay.html").as_uri())
        pg.evaluate("d => setup(d)", data)
        pg.evaluate("document.fonts.ready"); pg.wait_for_timeout(400)
        stage = pg.locator("#stage")
        for f in range(n):
            key = pg.evaluate("t => render(t)", f / FPS)
            out = frames / f"{f:05d}.png"
            if key == prev:
                os.link(prevf, out)
            else:
                stage.screenshot(path=str(out), omit_background=True); prev, prevf = key, out; uniq += 1
        b.close()
    log(f"overlay: {uniq} unique of {n} frames in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
