"""Keep search URLs in the local companion and send only short titles to the world."""
from collections import deque
import os
import re
import threading
from urllib.parse import urlsplit

URL = re.compile(r"https?://[^\s<>\"'\])]+", re.IGNORECASE)
MARKDOWN_LINK = re.compile(r"\[([^\]\n]*)\]\(https?://[^\s)]+\)", re.IGNORECASE)
LIMIT = 3


def hide_urls(text):
    return URL.sub("", MARKDOWN_LINK.sub(r"\1", text)).strip()


def public_https(url):
    if not isinstance(url, str) or len(url) > 2048 or re.search(r"[\s\\\x00-\x1f\x7f]", url):
        return False
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").rstrip(".").lower()
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
            return False
    except ValueError:
        return False
    if "@" in parsed.netloc or not re.fullmatch(r"(?:[a-z0-9-]+\.)+[a-z][a-z0-9-]*", host):
        return False
    if host.endswith((".local", ".localhost", ".internal", ".test", ".invalid", ".onion")):
        return False
    domains = os.environ.get("W04_LINK_DOMAINS", "")
    allow = [part.strip().lower().lstrip(".") for part in domains.split(",") if part.strip()]
    return not allow or any(host == domain or host.endswith("." + domain) for domain in allow)


def extract_links(content, previous=()):
    results, citations = {}, {}
    for block in (*previous, *content):
        if getattr(block, "type", "") == "web_search_tool_result":
            for result in block.content if isinstance(block.content, list) else []:
                if getattr(result, "type", "") == "web_search_result":
                    results[result.url] = result.title
    for block in content:
        for item in getattr(block, "citations", None) or ():
            if getattr(item, "type", "") == "web_search_result_location":
                citations[item.url] = item.title
        for url in URL.findall(getattr(block, "text", "")):
            url = url.rstrip(".,;!?}")
            if url in results:
                citations.setdefault(url, results[url])
    links = []
    for url, title in (citations or results).items():
        if public_https(url):
            title = " ".join(hide_urls(title or "").replace("|", " ").split())[:26] or "검색 출처"
            links.append({"url": url, "title": title})
    return links[:LIMIT]


class RecentLinks:
    def __init__(self):
        self.lock = threading.Lock()
        self.items = deque(maxlen=20)

    def add(self, root, npc, links):
        with self.lock:
            for link in reversed(links):
                self.items.appendleft(dict(link, root=root, npc=npc))

    def snapshot(self, root, npc):
        with self.lock:
            return [{"title": link["title"], "url": link["url"]} for link in self.items
                    if link["root"] == root and link["npc"] == npc]


recent = RecentLinks()
