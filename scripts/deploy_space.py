"""Create or update the Koumba demo on Hugging Face Spaces.

The token is read from HF_TOKEN (environment or the repository .env) and is
never printed.  Only the runtime files listed below are uploaded.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

SPACE_ID = "TIJAANI/koumba"
SPACE_URL = f"https://huggingface.co/spaces/{SPACE_ID}"
ROOT = Path(__file__).resolve().parent.parent
SPACE_README = ROOT / "space" / "README.md"

# These patterns are the complete upload set.  Anything outside them stays on
# the local machine, including the real .env, tests, and the Modal services.
ALLOW_PATTERNS = (
    "app.py",
    "koumba/**",
    "data/**",
    "requirements.txt",
    "LICENSE",
)

# Keep this list defensive too: a future allow pattern should not leak a local
# token or environment folder into the Space.
IGNORE_PATTERNS = (
    ".env",
    "**/.env",
    ".env.*",
    "**/.env.*",
    ".venv/**",
    "**/.venv/**",
    "tests/**",
    "**/tests/**",
    "modal_app/**",
    "**/modal_app/**",
    "data_cache/**",
    "**/data_cache/**",
    "*.wav",
    "**/*.wav",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.py[cod]",
    "**/*.py[cod]",
    ".pytest_cache/**",
    ".git/**",
    "space/**",
)

REQUIRED_PATHS = (
    ROOT / "app.py",
    ROOT / "koumba",
    ROOT / "data",
    ROOT / "requirements.txt",
    ROOT / "LICENSE",
    SPACE_README,
)


def _read_token() -> str:
    """Read HF_TOKEN without ever echoing its value."""

    load_dotenv(ROOT / ".env")
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Add it to the environment or the local .env."
        )
    return token


def _check_required_paths() -> None:
    missing = [
        str(path.relative_to(ROOT))
        for path in REQUIRED_PATHS
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "Cannot deploy; missing required path(s): " + ", ".join(missing)
        )


def main(*, api_factory=HfApi, token: str | None = None) -> str:
    """Create the Space if needed, upload the runtime files, return its URL."""

    _check_required_paths()
    token = token or _read_token()
    api = api_factory(token=token)

    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="cpu-basic",
        private=False,
        exist_ok=True,
    )

    # The Space README carries its YAML config, so it is uploaded separately
    # from the project folder and placed at the Space root.
    api.upload_file(
        path_or_fileobj=SPACE_README,
        path_in_repo="README.md",
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Add Space README",
    )
    api.upload_folder(
        folder_path=ROOT,
        repo_id=SPACE_ID,
        repo_type="space",
        allow_patterns=list(ALLOW_PATTERNS),
        ignore_patterns=list(IGNORE_PATTERNS),
        commit_message="Deploy Koumba demo",
    )

    print(f"Space URL: {SPACE_URL}")
    return SPACE_URL


if __name__ == "__main__":
    main()
