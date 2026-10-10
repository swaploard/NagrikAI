from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def canonicalize_url(url: str) -> str:
    """Return a stable source URL without discarding query identity."""
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo = f"{userinfo}:{parsed.password}"
        userinfo = f"{userinfo}@"

    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"{userinfo}{hostname}{port}"
    path = parsed.path or "/"
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_id_for_url(url: str) -> str:
    return _sha256_text(canonicalize_url(url))


def content_hash_for_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def version_id_for_content(source_id: str, content_hash: str) -> str:
    return _sha256_text(f"{source_id}\n{content_hash}")


def document_id_for_identifier(authority: str, identifier: str) -> str:
    normalized_authority = " ".join(authority.casefold().split())
    normalized_identifier = " ".join(identifier.casefold().split())
    return _sha256_text(f"{normalized_authority}\n{normalized_identifier}")
