#!/usr/bin/env python3
"""Minimal untrusted process that maps trusted sensor messages to commands."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import struct
import sys
from pathlib import Path
from typing import BinaryIO

MAX_REQUEST_BYTES = 1_048_576
MAX_RESPONSE_BYTES = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--module-file", type=Path, required=True)
    parser.add_argument("--function", default="control")
    parser.add_argument("--response-fd", type=int, required=True)
    return parser.parse_args()


def read_exact(stream: BinaryIO, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def write_response(stream: BinaryIO, response: dict[str, object]) -> None:
    payload = json.dumps(response, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_RESPONSE_BYTES:
        payload = json.dumps({"ok": False, "error": "controller error message was too long"}).encode()
    stream.write(struct.pack("!I", len(payload)) + payload)
    stream.flush()


def main() -> None:
    args = parse_args()
    response_stream = os.fdopen(args.response_fd, "wb", buffering=0, closefd=False)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    submission = args.submission.resolve()
    module_file = args.module_file.resolve()
    for import_root in (submission, submission / "src", module_file.parent, module_file.parent.parent):
        import_root_text = str(import_root)
        if import_root_text not in sys.path:
            sys.path.append(import_root_text)

    controller = None
    load_error: str | None = None
    try:
        from racing.student.api import load_student_controller

        controller = load_student_controller(args.module_file, function_name=args.function)
    except BaseException as error:
        load_error = f"{type(error).__name__}: {error}"[:1000]
    write_response(
        response_stream,
        {"ok": True} if load_error is None else {"ok": False, "error": load_error},
    )

    while True:
        try:
            request_size = struct.unpack("!I", read_exact(sys.stdin.buffer, 4))[0]
            if request_size > MAX_REQUEST_BYTES:
                raise ValueError("sensor request exceeded size limit")
            sensors = pickle.loads(read_exact(sys.stdin.buffer, request_size))
        except EOFError:
            return
        except BaseException as error:
            write_response(response_stream, {"ok": False, "error": f"invalid sensor request: {error}"[:1000]})
            return

        if load_error is not None or controller is None:
            write_response(response_stream, {"ok": False, "error": load_error or "controller did not load"})
            continue
        try:
            command = controller(sensors)
            response = {
                "ok": True,
                "throttle": float(command.throttle),
                "steer": float(command.steer),
            }
            write_response(response_stream, response)
        except BaseException as error:
            write_response(response_stream, {"ok": False, "error": f"{type(error).__name__}: {error}"[:1000]})


if __name__ == "__main__":
    main()
