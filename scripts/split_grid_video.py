"""Split an N×N grid-concatenated eval video into per-env videos.

RLinf saves eval videos as a single MP4 with all N_env rollouts tiled into
an image grid (e.g. 896×896 = 4×4 of 224×224). This script splits it into
per-env MP4s for inspection / writeup.
"""

from __future__ import annotations

import argparse
import os

import imageio.v2 as iio
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input_path", required=True, help="grid video (e.g. 0.mp4)")
    p.add_argument("--out", dest="output_dir", required=True, help="dir for per-env MP4s")
    p.add_argument("--grid", type=int, default=4, help="grid dimension (4 → 4×4 = 16 envs)")
    p.add_argument("--fps", type=float, default=30.0, help="output fps")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    reader = iio.get_reader(args.input_path)
    meta = reader.get_meta_data()
    W, H = meta["size"]
    cell_w = W // args.grid
    cell_h = H // args.grid

    frames = []
    for f in reader:
        frames.append(f)
    reader.close()

    if not frames:
        print("no frames decoded")
        return

    print(f"input: {len(frames)} frames, {W}×{H} → {args.grid * args.grid} cells of {cell_w}×{cell_h}")

    # Split each cell into its own stream
    for cell_idx in range(args.grid * args.grid):
        row = cell_idx // args.grid
        col = cell_idx % args.grid
        out_path = os.path.join(args.output_dir, f"env_{cell_idx:02d}.mp4")
        writer = iio.get_writer(out_path, fps=args.fps)
        for frame in frames:
            tile = frame[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w]
            writer.append_data(tile)
        writer.close()

    # Also build a simple side-by-side "before vs after" panel for quick scan
    panel_path = os.path.join(args.output_dir, "first_vs_last_panel.png")
    first, last = frames[0], frames[-1]
    panel = np.concatenate([first, last], axis=1)
    iio.imwrite(panel_path, panel)
    print(f"wrote {args.grid * args.grid} per-env videos + {panel_path}")


if __name__ == "__main__":
    main()
