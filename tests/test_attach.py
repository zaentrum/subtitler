"""subtitler attach: the copies of a run attached through the catalog's seam (a seam of the test's own), each
checked against the version the title plays and the subtitles it has."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from subtitler import attach


class Catalog:
    """What the seam answers: per item its version and subtitles; what was posted."""

    def __init__(self):
        self.items = {}
        self.posted = []
        self.tokens = set()

    def serve(self):
        catalog = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _answer(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _item(self):
                parts = self.path.split("/")  # /api/library/items/<id>/subtitles
                catalog.tokens.add(self.headers.get("Authorization"))
                return catalog.items.get(parts[4]) if len(parts) == 6 and parts[5] == "subtitles" else None

            def do_GET(self):
                item = self._item()
                if item is None:
                    return self._answer(404, {"error": "no such title"})
                self._answer(200, {"itemId": item["id"], "versionId": item["version"], "attachable": True,
                                   "subtitles": item["subtitles"]})

            def do_POST(self):
                item = self._item()
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                catalog.posted.append((item["id"], body))
                sub = {"id": f"att-{len(catalog.posted)}", "language": body["language"], "label": body["label"],
                       "format": "webvtt", "forced": body["forced"], "sdh": body["sdh"], "origin": body["origin"],
                       "attached": True, "versionId": item["version"]}
                item["subtitles"].append(sub)
                self._answer(201, {"itemId": item["id"], "created": True, "subtitle": sub})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server


@pytest.fixture
def catalog():
    c = Catalog()
    server = c.serve()
    c.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield c
    server.shutdown()


def copy(file, language, label, forced=False, sdh=False, kind="full"):
    return {"file": file, "kind": kind, "attach": {"language": language, "label": label, "origin": "recognized",
                                                   "forced": forced, "sdh": sdh, "format": "webvtt"}}


def run_folder(tmp_path, item, version, made):
    folder = tmp_path / "out" / item
    folder.mkdir(parents=True)
    for m in made:
        (folder / m["file"]).write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nThen we stay home.\n")
    (folder / "ocr.json").write_text(json.dumps({"id": item, "name": item, "version": version, "made": made,
                                                 "not_attached": []}))
    return folder


def test_the_copies_are_attached_but_where_the_title_has_one_or_plays_another_version(catalog, tmp_path, monkeypatch):
    catalog.items = {
        "item-1": {"id": "item-1", "version": "v1", "subtitles": [
            {"id": "own-1", "language": "ger", "label": "Deutsch", "format": "webvtt", "forced": False, "sdh": False,
             "origin": None},  # the package's own German text subtitle
            {"id": "own-2", "language": "eng", "label": "English SDH", "format": "pgs", "forced": False,
             "sdh": False, "origin": None},  # an image subtitle covers nothing
            {"id": "att-0", "language": "zh-Hans", "label": "Chinese · OCR", "format": "webvtt", "forced": False,
             "sdh": False, "origin": "recognized"},  # a copy attached before
            {"id": "att-9", "language": "en", "label": "English · AI", "format": "webvtt", "forced": False,
             "sdh": False, "origin": "translated"}]},  # a model's covers nothing
        "item-2": {"id": "item-2", "version": "v7", "subtitles": []}}
    run_folder(tmp_path, "item-1", "v1", [
        copy("en.vtt", "en", "English · OCR"), copy("en.sdh.vtt", "en", "English SDH · OCR", sdh=True, kind="sdh"),
        copy("en.signs.vtt", "en", "English · Signs · OCR", forced=True, kind="signs"),
        copy("de.vtt", "de", "Deutsch · OCR"), copy("zh-Hans.vtt", "zh-Hans", "简体 · OCR")])
    run_folder(tmp_path, "item-2", "v6", [copy("en.vtt", "en", "English · OCR")])  # read from a version gone since
    token = tmp_path / "token"
    token.write_text("secret\n")
    out = str(tmp_path / "out")
    assert attach.main(["--out", out, "--url", catalog.url, "--token-file", str(token), "--dry-run"]) == 0
    assert catalog.posted == [] and not (tmp_path / "out" / "item-1" / "attach.json").exists()
    assert attach.main(["--out", out, "--url", catalog.url, "--token-file", str(token)]) == 0
    assert catalog.tokens == {"Bearer secret"}
    posted = {b["label"]: b for _, b in catalog.posted}
    assert sorted(posted) == ["English SDH · OCR", "English · OCR", "English · Signs · OCR"]
    assert posted["English · Signs · OCR"]["forced"] and posted["English · OCR"]["origin"] == "recognized"
    assert posted["English · OCR"]["content"].startswith("WEBVTT")
    r = json.loads((tmp_path / "out" / "item-1" / "attach.json").read_text())
    why = {x["file"]: x["why"] for x in r["not_attached"]}
    assert "a text subtitle of its own (origin none: the title has it) of its language and kind (de, full): own-1" \
        in why["de.vtt"]
    assert "a text copy of an image subtitle of its language and kind (zh-Hans, full): att-0" in why["zh-Hans.vtt"]
    r2 = json.loads((tmp_path / "out" / "item-2" / "attach.json").read_text())
    assert "plays version v7 now, the copy was read from v6" in r2["not_attached"][0]["why"]
    # A run after it sends nothing again, and a second copy of a language and kind is refused by the seam's list.
    assert attach.main(["--out", out, "--url", catalog.url, "--token-file", str(token)]) == 0
    assert len(catalog.posted) == 3
    run_folder(tmp_path, "item-3", "v3", [copy("en.vtt", "en", "English · OCR")])
    catalog.items["item-3"] = {"id": "item-3", "version": "v3", "subtitles": [
        {"id": "att-5", "language": "en", "label": "English (PGS) · OCR", "format": "webvtt", "forced": False,
         "sdh": False, "origin": "recognized"}]}
    assert attach.main(["--out", out, "--url", catalog.url, "--token-file", str(token), "--only", "item-3"]) == 0
    assert len(catalog.posted) == 3


def test_the_kind_and_language_as_the_catalog_tells_them():
    assert attach.kind_of({"label": "English (Forced)"}) == "forced" == attach.kind_of({"forced": True})
    assert attach.kind_of({"label": "Signs & Songs", "forced": True}) == "signs"
    assert attach.kind_of({"label": "English SDH"}) == "sdh" == attach.kind_of({"label": "[CC]"})
    assert attach.kind_of({"label": "English"}) == "full"
    assert attach.language_of({"language": "chi", "label": "繁體"}) == ("zh", "Hant")
    assert attach.language_of({"language": "zh-Hans"}) == ("zh", "Hans")
    assert attach.language_of({"language": "ger"}) == ("de", None)
    hant = {"language": "zh-Hant", "label": "", "format": "webvtt", "origin": None}
    assert attach.covering([hant], {"language": "zh-Hans", "label": "简体 · OCR"}) is None
    assert attach.covering([{**hant, "language": "chi"}], {"language": "zh-Hans", "label": "简体 · OCR"})


def test_without_a_token_nothing_is_asked(tmp_path, monkeypatch):
    monkeypatch.delenv("KATALOG_TOKEN", raising=False)
    assert attach.main(["--out", str(tmp_path), "--url", "http://127.0.0.1:9"]) == 2
