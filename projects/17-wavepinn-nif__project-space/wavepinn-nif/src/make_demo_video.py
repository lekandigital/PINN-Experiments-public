#!/usr/bin/env python3
"""
Convert a Taichi frame export directory into MP4 and/or GIF.

This is a thin wrapper around ffmpeg so the demo pipeline stays reproducible.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path


def build_mp4_command(ffmpeg: str, pattern: str, fps: int, output_path: Path, overwrite: bool) -> list[str]:
    return [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        str(output_path),
    ]


def build_gif_commands(
    ffmpeg: str,
    pattern: str,
    fps: int,
    output_path: Path,
    overwrite: bool,
    width: int,
    palette_path: Path,
) -> tuple[list[str], list[str]]:
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    palettegen = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-vf",
        f"{scale},palettegen",
        str(palette_path),
    ]
    paletteuse = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-i",
        str(palette_path),
        "-lavfi",
        f"{scale}[x];[x][1:v]paletteuse",
        str(output_path),
    ]
    return palettegen, paletteuse


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create MP4/GIF files from exported Taichi frames")
    parser.add_argument("--frames-dir", required=True, help="Directory containing frame_0000.png style exports")
    parser.add_argument("--pattern", default="frame_%04d.png", help="ffmpeg frame pattern relative to --frames-dir")
    parser.add_argument("--fps", type=int, default=30, help="Playback fps for the output clip")
    parser.add_argument("--output-prefix", default="wave_demo", help="Output filename prefix inside --frames-dir")
    parser.add_argument("--gif-width", type=int, default=960, help="Width used for the GIF conversion")
    parser.add_argument("--mp4-only", action="store_true", help="Only create MP4")
    parser.add_argument("--gif-only", action="store_true", help="Only create GIF")
    parser.add_argument("--print-only", action="store_true", help="Print ffmpeg commands without running them")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    if args.mp4_only and args.gif_only:
        raise SystemExit("Choose at most one of --mp4-only or --gif-only.")

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    if not frames_dir.exists():
        raise SystemExit(f"Frames directory does not exist: {frames_dir}")

    pattern_path = str(frames_dir / args.pattern)
    mp4_path = frames_dir / f"{args.output_prefix}.mp4"
    gif_path = frames_dir / f"{args.output_prefix}.gif"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        ffmpeg = "ffmpeg"
        print("ffmpeg was not found on PATH. Commands are printed below.")
        args.print_only = True

    commands: list[list[str]] = []
    if not args.gif_only:
        commands.append(build_mp4_command(ffmpeg, pattern_path, args.fps, mp4_path, args.overwrite))

    gif_commands: tuple[list[str], list[str]] | None = None
    if not args.mp4_only:
        palette_path = Path(tempfile.gettempdir()) / f"{args.output_prefix}_palette.png"
        gif_commands = build_gif_commands(
            ffmpeg=ffmpeg,
            pattern=pattern_path,
            fps=args.fps,
            output_path=gif_path,
            overwrite=args.overwrite,
            width=args.gif_width,
            palette_path=palette_path,
        )
        commands.extend(list(gif_commands))

    for cmd in commands:
        print(format_command(cmd))

    if args.print_only:
        return 0

    for cmd in commands:
        subprocess.run(cmd, check=True)

    print(f"Wrote assets to {frames_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
