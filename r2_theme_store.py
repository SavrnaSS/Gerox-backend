from pathlib import Path

import r2_storage

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def normalize_theme_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "")
        .replace("-", "_")
    )


def sync_theme_to_local(theme_name: str, local_root: Path) -> Path:
    """
    Download themes/<theme_name>/* from R2 into local_root/<theme_name>/...
    Skips download if ETag matches (stored in .etag file).
    """
    if not r2_storage.R2_ENABLED:
        raise RuntimeError("R2 is not enabled; cannot sync themes from R2")

    theme_key = normalize_theme_name(theme_name)
    prefix = f"themes/{theme_key}/"

    local_theme_dir = local_root / theme_key
    local_theme_dir.mkdir(parents=True, exist_ok=True)

    keys = r2_storage.list_keys(prefix)

    # Filter images only (skip DS_Store etc.)
    keys = [k for k in keys if k.lower().endswith(IMG_EXTS)]
    if not keys:
        raise RuntimeError(f"No theme images found in R2 at prefix: {prefix}")

    for key in keys:
        filename = key.split("/")[-1]
        local_path = local_theme_dir / filename
        etag_path = local_theme_dir / (filename + ".etag")

        remote_etag = r2_storage.head_etag(key)
        local_etag = etag_path.read_text().strip() if etag_path.exists() else None

        if local_path.exists() and local_etag == remote_etag:
            continue

        data = r2_storage.get_bytes(key)
        local_path.write_bytes(data)
        etag_path.write_text(remote_etag)

    return local_theme_dir
