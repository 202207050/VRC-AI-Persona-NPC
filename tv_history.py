"""Recent playback per VRChat client and a shared, on-disk YouTube metadata cache."""
import json
import os
import re
import subprocess
import tempfile
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen

HISTORY_LIMIT = 10
UA = "W04-Voice/1.0 (YouTube oEmbed)"
ATTEMPT = re.compile(r"\[Video Playback\] Attempting to resolve URL '([^\r\n]+?)'\s*$")
TV_TOPIC = re.compile(
    r"티\s*비|텔레비전|유튜브|(?:동?영상)(?!\s*(?:통화|제작|편집))|"
    r"(?:아까|방금|전에|그거|그것|봤던|봤던 거).*?(?:틀어|재생)|"
    r"다시\s*틀어|뭐\s*(?:틀었|틀어져|나오고)|"
    r"\b(?:tv|television|youtube|video|clip|rewatch)\b|テレビ|動画|ユーチューブ", re.I)


def clean(value, limit):
    if not isinstance(value, str):
        return ""
    return " ".join("".join(c if c.isprintable() else " " for c in value).split())[:limit]


def clean_summary(value):
    return "\n".join(line for line in (clean(s, 146) for s in value.splitlines()[:3]) if line) if isinstance(value, str) else ""


def youtube_id(url):
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or parsed.username is not None:
            return ""
        parts = parsed.path.strip("/").split("/")
        host = parsed.hostname
        if host == "youtu.be" and len(parts) == 1:
            value = parts[0]
        elif host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
            if parts == ["watch"]:
                value = parse_qs(parsed.query).get("v", [""])[0]
            elif len(parts) == 2 and parts[0] in ("shorts", "live", "embed", "v"):
                value = parts[1]
            else:
                return ""
        else:
            return ""
        return value if re.fullmatch(r"[A-Za-z0-9_-]{11}", value) else ""
    except ValueError:
        return ""


def world_key(url):
    # FNV-1a over UTF-16 code units, shared with W04TvHistory.
    key = 2166136261
    raw = url.encode("utf-16-le", errors="surrogatepass")
    for i in range(0, len(raw), 2):
        key = ((key ^ (raw[i] | raw[i + 1] << 8)) * 16777619) & 0xffffffff
    return f"h{key:08x}"


class PlaybackLog:
    def __init__(self):
        self.rows = deque(maxlen=HISTORY_LIMIT)
        self.active = False
        self.confirmed = False

    def feed(self, line):
        marker = "[W04AI] TV_HISTORY "
        if marker in line:
            try:
                row = json.loads(line.split(marker, 1)[1])
                url = row["url"]
                if isinstance(url, str) and world_key(url) == row["key"]:
                    if not self.confirmed:
                        self.rows.clear()
                    self._put(url, clean(row.get("title"), 100), row["key"])
                    self.confirmed = True
                    self.active = True
                    return youtube_id(url)
            except (ValueError, KeyError, TypeError):
                pass
        elif "[W04AI] TV_HISTORY_IDLE" in line:
            self.active = False
        else:
            match = ATTEMPT.search(line)
            if match and match[1].startswith(("https://", "http://")):
                url = match[1]
                if not self.confirmed:
                    self._put(url, "", "")
                return youtube_id(url)
        return ""

    def _put(self, url, title, key):
        video = youtube_id(url)
        for old in list(self.rows):
            if old[0] != url and (key or not video or youtube_id(old[0]) != video):
                continue
            if not title:
                title = old[1]
            self.rows.remove(old)
        self.rows.appendleft((url, title, key))


def oembed(url):
    req = Request("https://www.youtube.com/oembed?" + urlencode({"url": url, "format": "json"}),
                  headers={"User-Agent": UA})
    with urlopen(req, timeout=4) as response:
        return clean(json.loads(response.read(65_537)).get("title"), 100)


def _caption_text(path):
    with path.open(encoding="utf-8", errors="replace") as stream:
        data = stream.read(1_000_000)
    if path.suffix == ".json3":
        events = json.loads(data).get("events", [])
        return clean(" ".join(seg.get("utf8", "") for event in events for seg in event.get("segs", [])), 4600)
    lines = []
    for line in data.splitlines():
        if line and "-->" not in line and not line.startswith(("WEBVTT", "Kind:", "Language:")):
            line = re.sub(r"<[^>]*>", "", line).strip()
            if line and line not in lines[-4:]:
                lines.append(line)
    return clean(" ".join(lines), 4600)


def metadata(exe, url):
    if not exe.is_file():
        return "", "", ""
    with tempfile.TemporaryDirectory(prefix="w04-tv-") as directory:
        args = [str(exe), "--ignore-config", "--no-playlist", "--no-cache-dir", "--encoding", "utf-8",
                "--socket-timeout", "4", "--retries", "0", "--extractor-retries", "0",
                "--skip-download", "--no-simulate", "--dump-single-json",
                "--write-subs", "--write-auto-subs", "--sub-langs", "ko,en,ja",
                "--sub-format", "json3/vtt", "-o", "caption.%(ext)s", "--", url]
        try:
            run = subprocess.run(args, cwd=directory, capture_output=True, timeout=14,
                                 encoding="utf-8", errors="replace",
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            info = json.loads(run.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return "", "", ""
        directory = Path(directory)
        for path in sorted(directory.glob("caption.*"), key=lambda p: ("ko" not in p.name, "en" not in p.name)):
            try:
                captions = _caption_text(path)
            except (OSError, ValueError, AttributeError, TypeError):
                continue
            if captions:
                break
        else:
            captions = ""
        return clean(info.get("title"), 100), clean(info.get("description"), 2400), captions


class VideoCache:
    def __init__(self, root, exe, summarize):
        self.path = Path(root) / "tv_video_cache.json"
        self.exe = Path(exe)
        self.summarize = summarize
        self.lock = threading.Lock()
        self.pending = set()
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="w04-tv")
        try:
            self.rows = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(self.rows, dict):
                self.rows = {}
        except (OSError, ValueError):
            self.rows = {}

    def watch(self, video):
        if not video:
            return
        with self.lock:
            existing = self.rows.get(video)
            if video in self.pending or (isinstance(existing, dict) and ("summary" in existing or self.summarize is None)):
                return
            self.pending.add(video)
        self.pool.submit(self._fetch, video)

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent, delete=False) as out:
            temp = out.name
            try:
                json.dump(self.rows, out, ensure_ascii=False)
                out.close()
                os.replace(temp, self.path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)

    def _fetch(self, video):
        try:
            url = "https://www.youtube.com/watch?v=" + video
            try:
                title = oembed(url)
            except (OSError, ValueError, AttributeError):
                title = ""
            if title:
                with self.lock:
                    self.rows[video] = {"title": title}
            alt, desc, captions = metadata(self.exe, url)
            title = title or alt
            summary = ""
            if self.summarize and (title or desc or captions):
                try:
                    summary = clean_summary(self.summarize(title, desc, captions))
                except Exception:
                    pass
            with self.lock:
                self.rows[video] = {"title": title}
                if self.summarize:
                    self.rows[video]["summary"] = summary
                try:
                    self._save()
                except OSError:
                    pass
        finally:
            with self.lock:
                self.pending.discard(video)

    def context(self, playback, heard, followup=""):
        if not TV_TOPIC.search(heard + " " + followup):
            return []
        rows = []
        with self.lock:
            for i, (url, world_title, key) in enumerate(playback.rows):
                video = youtube_id(url)
                cached = self.rows.get(video, {})
                if not isinstance(cached, dict):
                    cached = {}
                title = clean(cached.get("title"), 100) or world_title
                if not title:
                    title = ("YouTube " + video if video else "제목 미확인 영상") + f" ({i + 1}번)"
                rows.append({"title": title, "key": key, "summary": clean_summary(cached.get("summary")) if i < 3 else "",
                             "state": "resolving" if not key else "current" if i == 0 and playback.active else "recent"})
        return rows

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
