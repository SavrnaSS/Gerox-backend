import os
import uuid
import mimetypes
from typing import Optional

import boto3
from botocore.client import Config


def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if v else None


R2_ACCOUNT_ID = _env("R2_ACCOUNT_ID")
R2_BUCKET = _env("R2_BUCKET")
R2_ACCESS_KEY_ID = _env("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = _env("R2_SECRET_ACCESS_KEY")
R2_PUBLIC_BASE_URL = _env("R2_PUBLIC_BASE_URL")  # optional (custom domain or r2.dev)

R2_ENABLED = all([R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY])

if R2_ENABLED:
    R2_ENDPOINT = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )
else:
    s3 = None


def ensure_r2():
    if not R2_ENABLED:
        raise RuntimeError(
            "R2 is not configured. Set R2_ACCOUNT_ID, R2_BUCKET, "
            "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY"
        )


def guess_content_type(key: str, default: str = "application/octet-stream") -> str:
    ct, _ = mimetypes.guess_type(key)
    return ct or default


def put_bytes(
    key: str,
    data: bytes,
    content_type: Optional[str] = None,
    cache_control: str = "public, max-age=31536000, immutable",
) -> str:
    """
    Upload bytes to R2 and return a URL.
    If R2_PUBLIC_BASE_URL exists -> stable public URL.
    Else -> presigned GET (1 hour).
    """
    ensure_r2()
    if content_type is None:
        content_type = guess_content_type(key)

    s3.put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
        CacheControl=cache_control,
    )

    # Return public URL if configured
    if R2_PUBLIC_BASE_URL:
        return f"{R2_PUBLIC_BASE_URL.rstrip('/')}/{key}"

    # Otherwise return presigned GET (private bucket)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET, "Key": key},
        ExpiresIn=3600,
    )


def get_bytes(key: str) -> bytes:
    ensure_r2()
    obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
    return obj["Body"].read()


def head_etag(key: str) -> str:
    ensure_r2()
    return s3.head_object(Bucket=R2_BUCKET, Key=key)["ETag"].strip('"')


def list_keys(prefix: str) -> list[str]:
    ensure_r2()
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": R2_BUCKET, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            keys.append(obj["Key"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


def new_key(folder: str, ext: str) -> str:
    folder = folder.strip("/")

    if not ext.startswith("."):
        ext = "." + ext

    return f"{folder}/{uuid.uuid4().hex}{ext}"
