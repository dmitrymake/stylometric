"""Fetch SOCIOLIT full texts through the official authenticated API.

The public SOCIOLIT paper/repository do not redistribute full texts. This
script only uses the official API with credentials supplied by the user:

    SOCIOLIT_EMAIL=... SOCIOLIT_PASSWORD=... \
      python scripts/fetch_sociolit_fulltext.py --limit 10

Downloaded files are written under data/external/sociolit/fulltext/, which is
ignored by git.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stylo.jsonio import dump_strict, dumps_strict  # noqa: E402
API_HOST = "https://sociolit.ru:5100/api/v1"
OUT_DIR = ROOT / "data" / "external" / "sociolit" / "fulltext"


def log(*args: object) -> None:
    print(*args, flush=True)


def safe_part(value: Any, default: str = "unknown") -> str:
    text = str(value or default).strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:160] or default


class SociolitClient:
    def __init__(self, api_host: str = API_HOST) -> None:
        self.api_host = api_host.rstrip("/")
        self.cookie_jar = CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookie_jar))
        self.auth_token: str | None = None

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        raw: bool = False,
    ) -> Any:
        url = self.api_host + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url += "?" + urllib.parse.urlencode(clean, doseq=True)
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = dumps_strict(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.auth_token:
            headers["Custom-Authorization"] = self.auth_token
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=90) as response:
                body = response.read()
                if raw:
                    return body, dict(response.headers)
                ctype = response.headers.get("Content-Type", "")
                if "application/json" in ctype:
                    return json.loads(body.decode("utf-8"))
                return body.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed: HTTP {exc.code} {detail}") from exc

    def login(self, email: str, password: str) -> dict[str, Any]:
        response = self._request(
            "POST",
            "/signin",
            payload={
                "email": email,
                "password": password,
                "authTokenTtl": 3_600_000,
                "refreshTokenTtl": 3_600_000,
            },
        )
        user = unwrap_data(response)
        token = user.get("authToken") if isinstance(user, dict) else None
        if not token:
            raise RuntimeError(f"Signin succeeded but authToken was not found in response: {response!r}")
        self.auth_token = str(token)
        return user

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return unwrap_data(self._request("GET", path, params=params))

    def get_file(self, file_id: Any) -> tuple[bytes, dict[str, str]]:
        return self._request("GET", f"/text_files/{file_id}/file", raw=True)


def unwrap_data(value: Any) -> Any:
    if isinstance(value, dict) and "data" in value and len(value) <= 3:
        return value["data"]
    return value


def extract_items(page: Any) -> tuple[list[dict[str, Any]], int | None]:
    page = unwrap_data(page)
    total = None
    if isinstance(page, list):
        return [x for x in page if isinstance(x, dict)], None
    if not isinstance(page, dict):
        return [], None
    for total_key in ("total", "count", "items_count", "total_count"):
        if isinstance(page.get(total_key), int):
            total = int(page[total_key])
            break
    for key in ("items", "results", "rows", "texts", "data"):
        items = page.get(key)
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)], total
    return [], total


def collect_file_ids(value: Any) -> set[Any]:
    found: set[Any] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"text_file_id", "textFileId", "file_id", "fileId"} and item not in (None, ""):
                found.add(item)
            else:
                found.update(collect_file_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(collect_file_ids(item))
    return found


def list_texts(client: SociolitClient, pp: int, limit: int | None, show_hidden: bool) -> list[dict[str, Any]]:
    texts: list[dict[str, Any]] = []
    page_no = 1
    while True:
        params: dict[str, Any] = {"p": page_no, "pp": pp}
        if show_hidden:
            params["show_hidden"] = "true"
        page = client.get_json("/texts", params)
        items, total = extract_items(page)
        log(f"page={page_no} items={len(items)} total={total if total is not None else '?'}")
        if not items:
            break
        texts.extend(items)
        if limit is not None and len(texts) >= limit:
            return texts[:limit]
        if total is not None and len(texts) >= total:
            break
        if len(items) < pp:
            break
        page_no += 1
    return texts


def enrich_text(client: SociolitClient, text: dict[str, Any]) -> dict[str, Any]:
    text_id = text.get("id") or text.get("text_id")
    if text_id is None:
        return text
    try:
        detail = client.get_json(f"/texts/{text_id}")
    except RuntimeError as exc:
        log(f"warn: cannot fetch /texts/{text_id}: {exc}")
        return text
    if isinstance(detail, dict):
        merged = dict(text)
        merged.update(detail)
        return merged
    return text


def save_file(out_dir: pathlib.Path, text: dict[str, Any], file_id: Any, body: bytes) -> pathlib.Path:
    author = safe_part(text.get("author"))
    title = safe_part(text.get("title"), default=f"text_{text.get('id', 'unknown')}")
    year = text.get("year_created") or text.get("year_published") or "unknown"
    text_id = text.get("id") or text.get("text_id") or "unknown"
    target_dir = out_dir / author
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{safe_part(year)}_{safe_part(text_id)}_{safe_part(file_id)}_{title}.txt"
    path.write_bytes(body)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--pp", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show-hidden", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--email", default=os.environ.get("SOCIOLIT_EMAIL"))
    args = parser.parse_args()

    email = args.email
    password = os.environ.get("SOCIOLIT_PASSWORD")
    if not email:
        raise SystemExit("Set SOCIOLIT_EMAIL or pass --email.")
    if not password:
        password = getpass.getpass("SOCIOLIT password: ")

    client = SociolitClient()
    user = client.login(email, password)
    log(f"signed in as {user.get('email') or email}")

    texts = list_texts(client, args.pp, args.limit, args.show_hidden)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_dir / "metadata.jsonl"
    saved = 0
    missing_files = 0
    started = time.time()
    with metadata_path.open("w", encoding="utf-8") as meta_fh:
        for index, base_text in enumerate(texts, start=1):
            text = enrich_text(client, base_text)
            file_ids = collect_file_ids(text)
            meta_fh.write(dumps_strict(text, ensure_ascii=False) + "\n")
            if not file_ids:
                missing_files += 1
                log(f"{index}/{len(texts)} no file id: {text.get('author')} - {text.get('title')}")
                continue
            for file_id in sorted(file_ids, key=str):
                if args.dry_run:
                    log(f"{index}/{len(texts)} would download file_id={file_id}")
                    continue
                body, _headers = client.get_file(file_id)
                path = save_file(args.out_dir, text, file_id, body)
                saved += 1
                log(f"{index}/{len(texts)} saved {path.relative_to(ROOT)} ({len(body)} bytes)")

    summary = {
        "api_host": API_HOST,
        "texts_seen": len(texts),
        "files_saved": saved,
        "texts_without_file_ids": missing_files,
        "out_dir": str(args.out_dir),
        "metadata": str(metadata_path),
        "runtime_sec": round(time.time() - started, 3),
    }
    (args.out_dir / "fetch_summary.json").write_text(
        dumps_strict(summary, ensure_ascii=False, indent=2),
        "utf-8",
    )
    log(dumps_strict(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
