#!/usr/bin/env python3
"""
Encode the Prompt-3 hero media bundle for Project 12.

Pipeline per target:
    1. Invoke ``src/taichi_cloth4d_temporal_demo.py --offline`` in the chosen
       mode / checkpoint / volume combination. PNG frames land in
       ``artifacts/taichi_final/frames_<name>/``.
    2. Run ffmpeg on those frames with ``-crf 18 -preset slow -pix_fmt yuv420p``
       at the STUDIO playback framerate (30 * 0.72 = 21.6 fps).
    3. Move/copy the resulting MP4 to ``artifacts/taichi_final/<NAME>.mp4``.

Also renders ``poster.png`` as a single still via the viewer's --save_still.
The ablation track prepends a one-second title slide to make the "per-frame
query, no rollout" caveat visually explicit.

Prompt 3 deliverables produced by a full run:
  artifacts/taichi_final/
    nif_cloth4d_temporal_demo.mp4      (hero)
    baseline_demo.mp4                  (reference SDF; same STUDIO rig)
    comparison.mp4                     (side-by-side, supporting)
    temporal_ablation.mp4              (GRU-on per-frame, title-carded)
    poster.png                         (hero poster still)
    frames_nif/, frames_reference/, frames_compare/, frames_ablation/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIEWER = PROJECT_ROOT / "src" / "taichi_cloth4d_temporal_demo.py"
VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
ARTIFACTS = PROJECT_ROOT / "artifacts" / "taichi_final"

FRAMERATE = 30.0 * 0.72  # STUDIO playback_speed; 21.6 fps

PY = str(VENV_PY if VENV_PY.exists() else sys.executable)


def run(cmd: list[str], *, env: dict | None = None, log_prefix: str = "[run]") -> None:
    print(f"{log_prefix} $ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, text=True)
    dt = time.perf_counter() - t0
    print(f"{log_prefix} exit={proc.returncode} dt={dt:.1f}s", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def render_frames(
    name: str,
    mode: str,
    extra_args: list[str] | None = None,
) -> Path:
    frames_dir = ARTIFACTS / f"frames_{name}"
    # The viewer writes into <output_dir>/frames/; we pass <output_dir> and then
    # move that frames subdir so parallel renders don't collide.
    scratch = ARTIFACTS / f"_scratch_{name}"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    cmd = [
        PY, str(VIEWER),
        "--preset", "pitch",
        "--mode", mode,
        "--offline",
        "--output_dir", str(scratch),
        "--camera_preset", "3",
    ]
    if extra_args:
        cmd.extend(extra_args)
    run(cmd, log_prefix=f"[render:{name}]")

    src = scratch / "frames"
    if not src.exists():
        raise RuntimeError(f"Viewer did not produce {src}")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    shutil.move(str(src), str(frames_dir))
    shutil.rmtree(scratch)
    return frames_dir


def encode_mp4(frames_dir: Path, out_path: Path, title_card: str | None = None) -> None:
    """
    Encode frames_dir/frame_%04d.png to out_path with the canonical settings.

    When title_card is provided, a 1 s card is rendered first with the text
    centered, then concatenated with the main sequence via ffmpeg's concat
    demuxer. Title-card resolution matches the first frame's resolution.
    """
    main_mp4 = out_path.parent / f"_main_{out_path.stem}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vf_scale = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    # Main encode.
    cmd_main = [
        "ffmpeg", "-y",
        "-framerate", f"{FRAMERATE:.3f}",
        "-i", str(frames_dir / "frame_%04d.png"),
        "-vf", vf_scale,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        str(main_mp4),
    ]
    run(cmd_main, log_prefix=f"[encode:{out_path.name}]")

    if title_card is None:
        shutil.move(str(main_mp4), str(out_path))
        return

    # Determine the output resolution from the first frame so the title slide
    # matches exactly.
    first = sorted(frames_dir.glob("frame_*.png"))[0]
    from PIL import Image, ImageDraw, ImageFont
    w, h = Image.open(first).size

    # Generate the title-card PNG via Pillow (the local Homebrew ffmpeg was
    # built without libfreetype, so ffmpeg drawtext is unavailable).
    font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    font_path = next((p for p in font_candidates if Path(p).exists()), None)
    if font_path is None:
        raise RuntimeError("No suitable system font found for title-card rendering.")

    title_png = out_path.parent / f"_title_{out_path.stem}.png"
    title_mp4 = out_path.parent / f"_title_{out_path.stem}.mp4"

    card = Image.new("RGB", (w, h), (5, 7, 16))  # matches studio background
    draw = ImageDraw.Draw(card)
    font_size = max(int(h / 22), 20)
    # Wrap the text into 2-3 lines so it fits the 1600x1000 card.
    words = title_card.split(" - ")
    line_height = int(font_size * 1.35)
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except Exception:  # pragma: no cover
        font = ImageFont.load_default()
    total_h = line_height * len(words)
    y0 = (h - total_h) // 2
    for i, line in enumerate(words):
        tw = draw.textlength(line, font=font)
        draw.text(((w - tw) // 2, y0 + i * line_height),
                  line, fill=(224, 228, 234), font=font)
    card.save(title_png)

    cmd_title = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-framerate", f"{FRAMERATE:.3f}",
        "-t", "1.0",
        "-i", str(title_png),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-r", f"{FRAMERATE:.3f}",
        str(title_mp4),
    ]
    run(cmd_title, log_prefix=f"[title:{out_path.name}]")

    # Concat via concat-demuxer list file.
    list_path = out_path.parent / f"_concat_{out_path.stem}.txt"
    list_path.write_text(f"file '{title_mp4.name}'\nfile '{main_mp4.name}'\n")
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
        str(out_path),
    ]
    run(cmd_concat, log_prefix=f"[concat:{out_path.name}]")

    for tmp in (title_mp4, main_mp4, list_path, title_png):
        try:
            tmp.unlink()
        except FileNotFoundError:  # pragma: no cover
            pass


def render_poster(out_path: Path, cursor: float = 95.0) -> None:
    cmd = [
        PY, str(VIEWER),
        "--preset", "pitch",
        "--mode", "nif",
        "--camera_preset", "3",
        "--save_still", str(out_path),
        "--still_cursor", str(cursor),
    ]
    run(cmd, log_prefix=f"[poster]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip_render", action="store_true",
                    help="Skip viewer re-render; assume frames_*/ already exist.")
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated subset of targets to run "
                         "(nif,reference,compare,ablation,poster).")
    args = ap.parse_args()

    targets = {"nif", "reference", "compare", "ablation", "poster"}
    if args.only:
        targets = set(args.only.split(","))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # --- Hero (NIF) ---
    if "nif" in targets:
        if not args.skip_render:
            render_frames("nif", "nif")
        encode_mp4(
            ARTIFACTS / "frames_nif",
            ARTIFACTS / "nif_cloth4d_temporal_demo.mp4",
        )

    # --- Reference baseline ---
    if "reference" in targets:
        if not args.skip_render:
            render_frames("reference", "reference")
        encode_mp4(
            ARTIFACTS / "frames_reference",
            ARTIFACTS / "baseline_demo.mp4",
        )

    # --- Comparison (side-by-side) ---
    if "compare" in targets:
        if not args.skip_render:
            render_frames("compare", "compare")
        encode_mp4(
            ARTIFACTS / "frames_compare",
            ARTIFACTS / "comparison.mp4",
        )

    # --- Temporal ablation (GRU-on per-frame; title-carded) ---
    if "ablation" in targets:
        if not args.skip_render:
            render_frames(
                "ablation", "nif",
                extra_args=[
                    "--checkpoint", str(PROJECT_ROOT / "checkpoints" / "ablation" / "nif_cloth4d_temporal_gru.pt"),
                    "--use_precomputed_volume",
                    str(PROJECT_ROOT / "outputs" / "validation_model_temporal" / "field_sequence.npy"),
                ],
            )
        encode_mp4(
            ARTIFACTS / "frames_ablation",
            ARTIFACTS / "temporal_ablation.mp4",
            title_card=("GRU on - per-frame query, no rollout "
                        "- trust window = full clip"),
        )

    # --- Poster still ---
    if "poster" in targets:
        render_poster(ARTIFACTS / "poster.png")

    print("[encode_hero_media] done")


if __name__ == "__main__":
    main()
