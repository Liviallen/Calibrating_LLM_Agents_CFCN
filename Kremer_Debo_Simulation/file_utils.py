"""Atomic file writes used when importing fitted weights."""
import os
import tempfile


def atomic_write(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(data)
        temp = stream.name
    os.replace(temp, path)
