from __future__ import annotations


import pytest

from data import download


class FakeResponse:
    def __init__(self, *, content=b"bytes", text="text", payload=None, fail=False):
        self.content = content
        self.text = text
        self._payload = [] if payload is None else payload
        self._fail = fail

    def raise_for_status(self):
        if self._fail:
            raise RuntimeError("http failure")

    def json(self):
        return self._payload


def test_download_file_uses_cache(monkeypatch, tmp_path):
    dest = tmp_path / "cached.bin"
    dest.write_bytes(b"cached")
    monkeypatch.setattr(download.requests, "get", lambda *args, **kwargs: pytest.fail("network called"))
    assert download.download_file("https://example.test", dest) == dest
    assert dest.read_bytes() == b"cached"


def test_download_file_fetches_and_writes(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(content=b"new")

    monkeypatch.setattr(download.requests, "get", fake_get)
    dest = tmp_path / "nested" / "file.bin"
    assert download.download_file("https://example.test/file", dest, force=True) == dest
    assert dest.read_bytes() == b"new"
    assert calls[0][1]["timeout"] == download.TIMEOUT


def test_fetch_helpers_parse_expected_response_shapes(monkeypatch):
    responses = iter(
        [
            FakeResponse(payload={"value": [{"x": 1}]}),
            FakeResponse(payload=[{"valor": "5"}]),
            FakeResponse(text="Date,Close\n2020-01-01,2"),
        ]
    )
    monkeypatch.setattr(download.requests, "get", lambda *args, **kwargs: next(responses))
    assert download.fetch_ipeadata("SERIE") == [{"x": 1}]
    assert download.fetch_bcb(1) == [{"valor": "5"}]
    assert "Date,Close" in download.fetch_stooq("ho.f")


def test_download_all_writes_metadata_and_survives_optional_stooq_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "RAW", tmp_path)

    def fake_download(url, dest, force=False):
        dest.write_bytes(b"xlsx")
        return dest

    monkeypatch.setattr(download, "download_file", fake_download)
    monkeypatch.setattr(download, "fetch_ipeadata", lambda code: [{"codigo": code}])
    monkeypatch.setattr(download, "fetch_stooq", lambda symbol: (_ for _ in ()).throw(RuntimeError("offline")))
    paths = download.download_all(force=True)
    assert set(paths) == {"mensal_2013", "mensal_2001", "semanal_2013", "brent", "fx"}
    assert (tmp_path / "ipeadata_brent.json").exists()
    assert (tmp_path / "ipeadata_usdbrl.json").exists()
    assert "offline" in (tmp_path / "stooq_ulsd.error").read_text(encoding="utf-8")
    assert (tmp_path / "download_stamp.txt").read_text(encoding="utf-8")

