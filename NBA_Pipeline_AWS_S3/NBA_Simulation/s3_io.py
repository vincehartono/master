import os
import io
from typing import Iterable, List

import boto3
import pandas as pd

_s3 = boto3.client("s3")


def _bucket_and_prefix() -> tuple[str, str]:
    bucket = os.environ["NBA_DATA_BUCKET"]
    prefix = os.environ.get("NBA_DATA_PREFIX", "").strip()
    return bucket, prefix


def _key(name: str) -> tuple[str, str]:
    bucket, prefix = _bucket_and_prefix()
    return bucket, f"{prefix}{name}" if prefix else name


def read_csv(name: str) -> pd.DataFrame:
    bucket, key = _key(name)
    obj = _s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def write_csv(df: pd.DataFrame, name: str) -> None:
    bucket, key = _key(name)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    _s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def read_text_lines(name: str) -> List[str]:
    bucket, key = _key(name)
    try:
        obj = _s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read().decode("utf-8")
        return data.splitlines(keepends=True)
    except _s3.exceptions.NoSuchKey:
        return []


def write_text(name: str, lines: Iterable[str]) -> None:
    bucket, key = _key(name)
    text = "".join(lines)
    _s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))


def append_text_line(name: str, line: str) -> None:
    lines = read_text_lines(name)
    if not line.endswith("\n"):
        line += "\n"
    lines.append(line)
    write_text(name, lines)

