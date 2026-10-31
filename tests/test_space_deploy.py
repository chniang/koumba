"""The Space entry point and deploy script must stay safe and small."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from huggingface_hub.utils import filter_repo_objects

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


deploy_space = _load(
    "koumba_deploy_space",
    ROOT / "scripts" / "deploy_space.py",
)
space_app = _load("koumba_space_app", ROOT / "app.py")


class FakeDemo:
    def __init__(self) -> None:
        self.launch_kwargs = None

    def launch(self, **kwargs) -> None:
        self.launch_kwargs = kwargs


class FakeApi:
    def __init__(self, token: str) -> None:
        self.token = token
        self.created = None
        self.uploaded_file = None
        self.uploaded_folder = None

    def create_repo(self, **kwargs) -> None:
        self.created = kwargs

    def upload_file(self, **kwargs) -> None:
        self.uploaded_file = kwargs

    def upload_folder(self, **kwargs) -> None:
        self.uploaded_folder = kwargs


def test_space_entry_point_uses_the_spaces_host(monkeypatch) -> None:
    demo = FakeDemo()
    monkeypatch.setattr(space_app, "build_app", lambda: demo)

    space_app.main()

    assert demo.launch_kwargs["server_name"] == "0.0.0.0"
    assert demo.launch_kwargs["share"] is False
    assert "port" not in demo.launch_kwargs


def test_deploy_reads_the_token_from_the_environment(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(deploy_space, "ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "from-environment")

    assert deploy_space._read_token() == "from-environment"


def test_deploy_reads_the_token_from_a_local_dotenv(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(deploy_space, "ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text(
        "HF_TOKEN=from-dotenv\n",
        encoding="utf-8",
    )

    assert deploy_space._read_token() == "from-dotenv"


def test_deploy_allow_list_keeps_private_files_off_the_space() -> None:
    assert ".env" not in deploy_space.ALLOW_PATTERNS
    assert not any("tests" in pattern for pattern in deploy_space.ALLOW_PATTERNS)
    assert ".env" in deploy_space.IGNORE_PATTERNS
    assert any("tests" in pattern for pattern in deploy_space.IGNORE_PATTERNS)

    kept = set(
        filter_repo_objects(
            (
                "app.py",
                "requirements.txt",
                "LICENSE",
                "koumba/agent.py",
                "data/voice_lexicon.csv",
                ".env",
                ".venv/Lib/site-packages/x.py",
                "tests/test_app.py",
                "modal_app/asr.py",
                "data_cache/lignes_dakar.csv",
                "voice.wav",
                "koumba/__pycache__/agent.pyc",
            ),
            allow_patterns=list(deploy_space.ALLOW_PATTERNS),
            ignore_patterns=list(deploy_space.IGNORE_PATTERNS),
        )
    )

    assert kept == {
        "app.py",
        "requirements.txt",
        "LICENSE",
        "koumba/agent.py",
        "data/voice_lexicon.csv",
    }


def test_deploy_with_a_fake_api_uploads_the_runtime_files(capsys) -> None:
    fake = FakeApi("test-token")

    url = deploy_space.main(api_factory=lambda token: fake, token="test-token")

    assert url == "https://huggingface.co/spaces/TIJAANI/koumba"
    assert fake.created == {
        "repo_id": "TIJAANI/koumba",
        "repo_type": "space",
        "space_sdk": "gradio",
        "space_hardware": "cpu-basic",
        "private": False,
        "exist_ok": True,
    }
    assert fake.uploaded_file["path_in_repo"] == "README.md"
    assert fake.uploaded_file["repo_type"] == "space"
    assert fake.uploaded_folder["allow_patterns"] == list(
        deploy_space.ALLOW_PATTERNS
    )
    assert fake.uploaded_folder["ignore_patterns"] == list(
        deploy_space.IGNORE_PATTERNS
    )

    output = capsys.readouterr()
    assert url in output.out
    assert "test-token" not in output.out + output.err


def test_space_readme_has_the_expected_header() -> None:
    text = (ROOT / "space" / "README.md").read_text(encoding="utf-8")

    for line in (
        "title: Koumba",
        "emoji: 🚌",
        "colorFrom: green",
        "colorTo: yellow",
        "sdk: gradio",
        "sdk_version: 6.30.0",
        "app_file: app.py",
        "pinned: false",
        "license: mit",
        'short_description: "Wolof voice agent for Dakar buses '
        '(Meta Omnilingual ASR + Muse Spark Open)"',
    ):
        assert line in text
    assert "https://github.com/chniang/koumba" in text
