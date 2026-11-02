"""The text copies `subtitler ocr` made, attached to their titles through the catalog's seam.

    subtitler attach --out DIR --url URL [--token-file FILE] [--only ID ...] [--dry-run]

For each title of DIR (<DIR>/<id>/ocr.json) its copies (`made`) are attached
by katalog-manager's attach seam: POST /api/library/items/<id>/subtitles,
the copy's WebVTT with its language, label, forced and SDH, origin
`recognized`. The seam is asked first (GET .../subtitles), and a copy is not
attached, and said why, where

- the title has no version a subtitle attaches to (`attachable`), or plays
  another version than the one the copy was read from (`versionId`, against
  ocr.json's `version`): its pictures may not be that version's (read it
  again: `subtitler ocr --again`);
- the title has a subtitle of the copy's language and kind that covers it: a
  text copy of an image subtitle (origin `recognized`) attached before, or a
  text subtitle of its own (its package's, a file beside its source, a
  stream of its original; or one attached from its file or a provider,
  origin `source`, `provider`). The kind is told as the catalog's rows tell
  it: signs where the label says so, forced by the flag or the label, SDH by
  the flag or the label, else full; Chinese of either script where one of
  the two names none. A subtitle a model made (translated, transcribed,
  converted) or an image one covers none.

Each title's outcome goes into <DIR>/<id>/attach.json: each copy attached,
with the catalog's id of it, or not, with why; a copy attached before is not
sent again. --dry-run asks the seam alone and writes nothing but
<DIR>/attach-dry-run.json. The token: --token-file, else $KATALOG_TOKEN (a
bearer token of a client the seam takes); the seam's address: --url, else
$KATALOG_URL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import __version__, script
from .langs import Lang
from .package import TEXT_FORMATS
from .recognize import read_json, say, write_json

ORIGIN = "recognized"
COVERING = (None, "source", "provider", ORIGIN)  # what a subtitle's origin is where it covers a copy

# The kind of a subtitle as the catalog's rows say it (deploy/ocr-titles.sql reads them the same way): a
# package's rows carry SDH in their label alone.
_SIGNS = re.compile(r"\bsigns?\b", re.I)
_FORCED = re.compile(r"\bforced\b|foreign (language )?parts", re.I)
_SDH = re.compile(r"\bsdh\b|hearing|\bcc\b|\[cc\]", re.I)


def kind_of(s: dict) -> str:
    """signs | forced | sdh | full, of a subtitle of the seam (or a copy's attach entry)."""
    label = s.get("label") or ""
    if _SIGNS.search(label):
        return "signs"
    if s.get("forced") or _FORCED.search(label):
        return "forced"
    if s.get("sdh") or _SDH.search(label):
        return "sdh"
    return "full"


def language_of(s: dict) -> tuple[str | None, str | None]:
    """(ISO 639-1, Chinese script or None) of a subtitle: its tag ("eng", "zh-Hans"), a Chinese one's
    script from its label where the tag names none."""
    tag = (s.get("language") or "").strip()
    if not tag:
        return None, None
    lang = Lang.parse(tag)
    if lang.code != "zh":
        return lang.code, None
    return "zh", lang.script or script.hints(s.get("label") or "")["script"]


def covering(subtitles: list[dict], attach: dict) -> dict | None:
    """The subtitle of the title that covers the copy `attach` describes (its language, kind and,
    for Chinese, script where both name one), a text subtitle of its own or a copy attached before;
    None where none does."""
    code, sc = language_of(attach)
    kind = kind_of(attach)
    for s in subtitles:
        if (s.get("format") or "").lower() not in TEXT_FORMATS or s.get("origin") not in COVERING:
            continue
        c, ssc = language_of(s)
        if c == code and kind_of(s) == kind and (code != "zh" or sc is None or ssc is None or sc == ssc):
            return s
    return None


class Seam:
    """katalog-manager's attach seam (bearer token)."""

    def __init__(self, url: str, token: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json",
                   "User-Agent": f"subtitler/{__version__}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                answer = json.loads(e.read() or b"{}")
            except ValueError:
                answer = {}
            return e.code, answer if isinstance(answer, dict) else {"error": str(answer)}

    def _item(self, item: str) -> str:
        return f"/api/library/items/{urllib.parse.quote(item, safe='')}/subtitles"

    def subtitles(self, item: str) -> tuple[int, dict]:
        return self._call("GET", self._item(item))

    def attach(self, item: str, body: dict) -> tuple[int, dict]:
        return self._call("POST", self._item(item), body)


def _why(status: int, answer: dict) -> str:
    return f"the seam answered {status}: {answer.get('error') or answer}"


def attach_title(seam: Seam, folder: Path, r: dict, dry_run: bool = False) -> dict:
    """A title's copies attached (or what would be): its attach.json's data."""
    before = read_json(folder / "attach.json") if not dry_run else None
    sent = {x["file"]: x for x in (before or {}).get("attached", [])}
    out = {"id": r["id"], "name": r.get("name") or r["id"], "made_by": f"subtitler {__version__}",
           "version": r.get("version"), "attached": [x for x in sent.values()], "not_attached": [], "errors": []}
    todo = [m for m in r.get("made", []) if m["file"] not in sent]
    if not todo:
        return out
    status, answer = seam.subtitles(r["id"])
    if status != 200:
        out["errors"].append({"why": _why(status, answer)})
        return out
    if not answer.get("attachable"):
        why = f"the title takes no subtitle: {answer.get('reason') or 'it has no version to attach to'}"
        out["not_attached"] += [{"file": m["file"], "why": why} for m in todo]
        return out
    if answer.get("versionId") != r.get("version"):
        why = (f"the title plays version {answer.get('versionId')} now, the copy was read from "
               f"{r.get('version') or 'a version it does not name'}: read it again (subtitler ocr --again)")
        out["not_attached"] += [{"file": m["file"], "why": why} for m in todo]
        return out
    subtitles = list(answer.get("subtitles") or [])
    for m in todo:
        body = {k: m["attach"][k] for k in ("language", "label", "forced", "sdh", "origin", "format")}
        if body["origin"] != ORIGIN:
            out["not_attached"].append({"file": m["file"], "why": f"its origin is {body['origin']}, not {ORIGIN}"})
            continue
        there = covering(subtitles, body)
        if there is not None:
            whose = ("a text copy of an image subtitle" if there.get("origin") == ORIGIN else
                     f"a text subtitle of its own (origin {there.get('origin') or 'none: the title has it'})")
            out["not_attached"].append({"file": m["file"], "covered_by": there.get("id"),
                                        "why": f"the title has {whose} of its language and kind "
                                               f"({body['language']}, {kind_of(body)}): {there.get('id')}"})
            continue
        if dry_run:
            out["attached"].append({"file": m["file"], "dry_run": True, **body})
            subtitles.append({**body, "id": f"(would be) {m['file']}"})
            continue
        try:
            body["content"] = (folder / m["file"]).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            out["errors"].append({"file": m["file"], "why": f"its file cannot be read: {e}"})
            continue
        status, answer = seam.attach(r["id"], body)
        if status not in (200, 201):
            out["errors"].append({"file": m["file"], "why": _why(status, answer)})
            continue
        sub = answer.get("subtitle") or {}
        out["attached"].append({"file": m["file"], "subtitle": sub.get("id"), "created": bool(answer.get("created")),
                                "language": body["language"], "label": sub.get("label", body["label"])})
        subtitles.append(sub or {**body, "id": None})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="subtitler attach", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the folder `subtitler ocr` wrote its copies into")
    ap.add_argument("--url", default=os.environ.get("KATALOG_URL"), help="katalog-manager (default: $KATALOG_URL)")
    ap.add_argument("--token-file", default=None, help="a bearer token (default: $KATALOG_TOKEN)")
    ap.add_argument("--only", nargs="*", default=None, help="these titles' ids alone")
    ap.add_argument("--dry-run", action="store_true", help="ask the seam alone: what would be attached")
    a = ap.parse_args(argv)
    token = (Path(a.token_file).read_text().strip() if a.token_file else os.environ.get("KATALOG_TOKEN", "")).strip()
    if not a.url or not token:
        say("subtitler attach: the seam's address (--url, $KATALOG_URL) and a token (--token-file, $KATALOG_TOKEN) "
            "are needed")
        return 2
    seam = Seam(a.url, token)
    out = Path(a.out)
    folders = sorted(f for f in out.iterdir() if (f / "ocr.json").is_file()) if out.is_dir() else []
    if a.only is not None:
        folders = [f for f in folders if f.name in set(a.only)]
    results, rc = [], 0
    for folder in folders:
        r = read_json(folder / "ocr.json")
        if not isinstance(r, dict) or "error" in r or r.get("dry_run") or not r.get("made"):
            continue
        try:
            res = attach_title(seam, folder, r, a.dry_run)
        except (OSError, ValueError) as e:  # the seam not reached, or no JSON from it
            res = {"id": r["id"], "name": r.get("name") or r["id"], "attached": [], "not_attached": [],
                   "errors": [{"why": f"the seam: {e}"}]}
        if not a.dry_run:
            write_json(folder / "attach.json", res)
        results.append(res)
        rc = rc or int(bool(res["errors"]))
        say(f"{res['name']}: " + ("would attach " if a.dry_run else "attached ")
            + (", ".join(x["file"] for x in res["attached"]) or "nothing")
            + "".join(f"; not {x['file']}: {x['why']}" for x in res["not_attached"])
            + "".join(f"; error{' ' + x['file'] if x.get('file') else ''}: {x['why']}" for x in res["errors"]))
    summary = {"made_by": f"subtitler {__version__}", "mode": "attach" + (" dry-run" if a.dry_run else ""),
               "titles": results}
    if out.is_dir():
        write_json(out / ("attach-dry-run.json" if a.dry_run else "attach-run.json"), summary)
    say(f"attach: {len(results)} titles, {sum(len(x['attached']) for x in results)} copies "
        + ("would be " if a.dry_run else "") + f"attached, {sum(len(x['not_attached']) for x in results)} not, "
        f"{sum(len(x['errors']) for x in results)} errors")
    return rc


if __name__ == "__main__":
    sys.exit(main())
