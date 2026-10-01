"""Extract one F1 starting-light beep from the decoded source documented with the audio assets."""

from __future__ import annotations

import argparse
import struct
import wave
from pathlib import Path


def extract_beep(source: Path, output: Path) -> None:
    with wave.open(str(source), "rb") as recording:
        if recording.getsampwidth() != 2:
            raise ValueError("source must be 16-bit PCM WAV")
        rate, channels = recording.getframerate(), recording.getnchannels()
        recording.setpos(round(0.735 * rate))
        frames = recording.readframes(round(0.370 * rate))
    values = struct.unpack(f"<{len(frames) // 2}h", frames)
    mono = [sum(values[i:i + channels]) / channels for i in range(0, len(values), channels)]
    offset = sum(mono) / len(mono)
    mono = [value - offset for value in mono]
    gain = 0.85 * 32767 / max(abs(value) for value in mono)
    samples = [
        round(value * gain * min(1.0, i / (0.003 * rate), (len(mono) - 1 - i) / (0.018 * rate)))
        for i, value in enumerate(mono)
    ]
    with wave.open(str(output), "wb") as beep:
        beep.setnchannels(1)
        beep.setsampwidth(2)
        beep.setframerate(rate)
        beep.writeframes(struct.pack(f"<{len(samples)}h", *samples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract_beep(args.source, args.output)
