import hashlib


def hash_chunk(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def hash_file_bytes(data: bytes, prefix_len: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:prefix_len]
