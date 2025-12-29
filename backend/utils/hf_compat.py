"""Hugging Face compatibility helpers.

This module shields the rest of the codebase from breaking API
changes in huggingface_hub by providing missing helpers (currently
`cached_download`).
"""
from __future__ import annotations

import os
import shutil
from typing import Any, Optional
from urllib.parse import urlparse


def ensure_hf_cached_download() -> None:
    """Ensure huggingface_hub exposes cached_download.

    huggingface_hub 0.24+ removed cached_download in favour of
    hf_hub_download, but sentence-transformers still imports the old
    symbol. We monkeypatch the module before any SentenceTransformer
    import to keep backwards compatibility.
    """

    try:
        import huggingface_hub as hfhub  # type: ignore
    except ImportError:
        return

    if hasattr(hfhub, "cached_download"):
        return

    # Try to reuse the canonical implementation if it still exists but isn't
    # re-exported at the package root (huggingface_hub>=0.24).
    file_cached_download: Optional[Any] = None
    try:
        from huggingface_hub.file_download import cached_download as file_cached_download  # type: ignore
    except Exception:
        file_cached_download = None

    if file_cached_download is not None:
        hfhub.cached_download = file_cached_download  # type: ignore[attr-defined]
        return

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        return

    def cached_download(
        url: str,
        *args: Any,
        cache_dir: Optional[str] = None,
        force_filename: Optional[str] = None,
        resume_download: bool = False,
        use_auth_token: Optional[str] = None,
        local_files_only: bool = False,
        legacy_cache_layout: bool = False,
        **kwargs: Any,
    ) -> str:
        """Minimal reimplementation backed by hf_hub_download."""

        if not url:
            raise ValueError("cached_download() requires a url")

        parsed = urlparse(url)
        path_parts = [part for part in parsed.path.split("/") if part]
        if "resolve" not in path_parts or len(path_parts) < 4:
            raise ValueError(f"Unexpected Hugging Face URL structure: {url}")

        resolve_idx = path_parts.index("resolve")
        repo_id = "/".join(path_parts[:resolve_idx])
        revision = path_parts[resolve_idx + 1]
        filename = "/".join(path_parts[resolve_idx + 2 :])

        # Bridge old parameter names to the new API
        token = kwargs.pop("token", None) or use_auth_token
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            force_filename=force_filename,
            resume_download=resume_download,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            proxies=proxies,
            **kwargs,
        )

        if cache_dir:
            relative_path = force_filename or filename
            target_path = os.path.join(cache_dir, relative_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            copy_needed = force_download or not os.path.exists(target_path)
            if copy_needed:
                shutil.copy2(downloaded_path, target_path)

            return target_path

        return downloaded_path

    hfhub.cached_download = cached_download  # type: ignore[attr-defined]
