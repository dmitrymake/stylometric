#!/usr/bin/env python3
"""Download a URL, detect encoding by scoring frequent Russian words, strip HTML, save UTF-8."""
import sys, re, urllib.request
from html import unescape

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
FREQ = ["и", "в", "не", "на", "что", "он", "как", "его", "то", "за",
        "по", "из", "она", "был", "все", "так", "да", "но", "ты", "я",
        "конь", "царь", "сказал", "вот", "себе"]

def score(text):
    low = " " + text.lower() + " "
    return sum(low.count(" " + w + " ") for w in FREQ)

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    raw = urllib.request.urlopen(req, timeout=60).read()
    best, best_s, best_enc = None, -1, None
    for enc in ("windows-1251", "koi8-r", "utf-8"):
        try:
            t = raw.decode(enc)
        except Exception:
            continue
        s = score(t)
        if s > best_s:
            best, best_s, best_enc = t, s, enc
    return best, best_enc

def strip_html(h):
    h = re.sub(r"(?is)<head.*?</head>", " ", h)
    h = re.sub(r"(?is)<script.*?</script>", " ", h)
    h = re.sub(r"(?is)<style.*?</style>", " ", h)
    # keep line breaks for verse
    h = re.sub(r"(?i)<br\s*/?>", "\n", h)
    h = re.sub(r"(?i)</p>", "\n\n", h)
    h = re.sub(r"(?i)</div>", "\n", h)
    h = re.sub(r"(?i)</h[1-6]>", "\n\n", h)
    h = re.sub(r"(?s)<[^>]+>", " ", h)
    h = unescape(h)
    h = re.sub(r"[ \t\xa0]+", " ", h)
    h = re.sub(r"\n[ \t]+", "\n", h)
    h = re.sub(r"\n{3,}", "\n\n", h)
    return h.strip()

if __name__ == "__main__":
    url, out = sys.argv[1], sys.argv[2]
    text, enc = fetch(url)
    body = strip_html(text)
    with open(out, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"{url}\n -> {out}  enc={enc}  chars={len(body)}  score={score(body)}")
