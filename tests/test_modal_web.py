"""The Modal web wrapper is checked with a fake Gradio mount and no network."""

from __future__ import annotations

import pytest

pytest.importorskip("modal")

from fastapi import FastAPI

from modal_app import web as web_module


def _modal_function():
    # Synchronicity wraps Modal objects, so use the original object to read
    # the resource spec without starting or deploying anything.
    function = web_module.web
    original_name = next(
        name
        for name in dir(function)
        if name.startswith("_sync_original_")
    )
    return getattr(function, original_name)


def test_create_asgi_app_mounts_gradio_at_root(monkeypatch) -> None:
    mounted = {}
    demo = object()

    def fake_mount(app, blocks, path, **kwargs):
        mounted["app"] = app
        mounted["blocks"] = blocks
        mounted["path"] = path
        mounted["kwargs"] = kwargs
        return app

    monkeypatch.setattr(web_module, "build_app", lambda: demo)
    monkeypatch.setattr(web_module.gr, "mount_gradio_app", fake_mount)

    api = web_module.create_asgi_app()

    assert isinstance(api, FastAPI)
    assert mounted["app"] is api
    assert mounted["blocks"] is demo
    assert mounted["path"] == "/"
    assert mounted["kwargs"]["css"] is web_module.CSS
    assert mounted["kwargs"]["footer_links"] == []
    assert mounted["kwargs"]["head"] is web_module.NOTRANSLATE_HEAD
    assert '<meta name="google" content="notranslate">' in (
        web_module.NOTRANSLATE_HEAD
    )


def test_web_function_uses_one_small_cpu_container() -> None:
    spec = _modal_function()._spec_

    assert web_module.MAX_CONTAINERS == 1
    assert web_module.MAX_INPUTS == 100
    assert spec.cpu == 1.0
    assert spec.memory == 2048
    assert list(spec.volumes) == [web_module.CACHE_PATH]
    assert spec.volumes[web_module.CACHE_PATH]._name == "koumba-web-cache"
    assert [secret._name for secret in spec.secrets if secret._name] == [
        "koumba-app"
    ]


def test_web_uses_the_app_secret_and_cache_path() -> None:
    assert web_module.SECRET_KEYS == (
        "BASETEN_API_KEY",
        "KOUMBA_BASE_URL",
        "KOUMBA_API_KEY_ENV",
        "KOUMBA_MODEL",
        "KOUMBA_ASR_URL",
        "KOUMBA_ASR_KEY",
        "KOUMBA_TTS_URL",
    )
    assert web_module.SECRET_NAME == "koumba-app"
    assert web_module.WEB_CACHE_NAME == "koumba-web-cache"
    assert web_module.CACHE_PATH == "/root/data_cache"
    assert web_module.DATA_PATH == "/root/data"
