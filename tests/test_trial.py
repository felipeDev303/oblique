"""The trial runner's request to arXiv carries a project identifier, never personal data."""

from types import SimpleNamespace

from oblique.product import trial


def test_the_pdf_download_identifies_the_project_only(tmp_path, monkeypatch):
    sent = {}

    def get(url, headers=None, **kw):
        sent.update(url=url, headers=headers or {})
        return SimpleNamespace(content=b"%PDF-1.4", raise_for_status=lambda: None)

    monkeypatch.setattr(trial.httpx, "get", get)
    name, data = trial.fetch({"arxiv_id": "2305.03726", "version": "v2"}, tmp_path)
    assert (name, data) == ("2305.03726v2.pdf", b"%PDF-1.4")
    assert sent["url"] == "https://arxiv.org/pdf/2305.03726v2"
    assert sent["headers"] == {"User-Agent": "oblique-hackathon/0.1"}
    assert "@" not in trial.USER_AGENT
