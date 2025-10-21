"""Post-processing utilities for MSDZip bitrate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_resolution(value: str) -> Tuple[int, int]:
    if not value:
        raise ValueError("Resolution string must be provided")
    separators = ['x', 'X', ',', ' ']  # allow a few common separators
    for sep in separators:
        if sep in value:
            parts = [p for p in value.replace(' ', '').split(sep) if p]
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    raise ValueError(f"Could not parse resolution '{value}'. Use formats like 512x512 or 512,512.")


def write_pgm(path: Path, data: np.ndarray) -> None:
    data = np.clip(data, 0, 255).astype(np.uint8)
    header = f"P5\n{data.shape[1]} {data.shape[0]}\n255\n".encode('ascii')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(header)
        f.write(data.tobytes())


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Visualise MSDZip bitrate metrics.')
    parser.add_argument('--metrics', required=True, help='Path to the metrics npz file produced by compress.py.')
    parser.add_argument('--resolution', required=True,
                        help='Resolution of the original data as WIDTHxHEIGHT (e.g. 512x512).')
    parser.add_argument('--output-npy', help='Optional path to save the 2D bitrate map as a NumPy array.')
    parser.add_argument('--heatmap', help='Optional path to save a grayscale heatmap (PGM format).')
    parser.add_argument('--mask', help='Optional path to save a binary mask (PGM format).')
    parser.add_argument('--threshold', type=float,
                        help='Threshold (in bits) used when generating the binary mask.')
    parser.add_argument('--topk', type=int, default=0,
                        help='Print the top-K hardest locations (highest bitrate).')
    parser.add_argument('--topk-json', help='Optional path to export the top-K list as JSON.')
    args = parser.parse_args(argv)

    width, height = parse_resolution(args.resolution)
    data = np.load(args.metrics, allow_pickle=True)
    if 'bits' not in data:
        raise ValueError(f"The file {args.metrics} does not contain a 'bits' array.")
    bits = data['bits']
    if bits.ndim != 1:
        bits = bits.reshape(-1)
    expected = width * height
    if bits.size != expected:
        raise ValueError(f"Resolution {width}x{height} expects {expected} entries, found {bits.size}.")
    grid = bits.reshape(height, width)
    valid_mask = np.isfinite(grid)

    if not valid_mask.any():
        raise ValueError('No valid metrics found in the provided file.')

    stats = {
        'min_bits': float(np.nanmin(grid)),
        'max_bits': float(np.nanmax(grid)),
        'mean_bits': float(np.nanmean(grid)),
    }
    print('Bitrate statistics:', stats)

    if args.output_npy:
        np.save(args.output_npy, grid)
        print(f'Saved bitrate map to {args.output_npy}')

    if args.heatmap:
        scaled = np.zeros_like(grid, dtype=np.float32)
        valid_values = grid[valid_mask]
        min_val = float(valid_values.min())
        max_val = float(valid_values.max())
        if max_val > min_val:
            scaled[valid_mask] = (valid_values - min_val) / (max_val - min_val)
        heatmap = (scaled * 255.0).astype(np.uint8)
        write_pgm(Path(args.heatmap), heatmap)
        print(f'Saved heatmap to {args.heatmap}')

    if args.mask:
        if args.threshold is None:
            raise ValueError('A --threshold value must be provided when generating a mask.')
        mask = np.zeros_like(grid, dtype=np.uint8)
        mask[(grid >= args.threshold) & valid_mask] = 255
        write_pgm(Path(args.mask), mask)
        print(f'Saved mask to {args.mask}')

    if args.topk and args.topk > 0:
        flat_bits = grid.reshape(-1)
        flat_mask = valid_mask.reshape(-1)
        valid_indices = np.where(flat_mask)[0]
        values = flat_bits[flat_mask]
        order = np.argsort(values)[::-1]
        topk = []
        for idx in order[:args.topk]:
            linear_index = int(valid_indices[idx])
            row = linear_index // width
            col = linear_index % width
            topk.append({'index': linear_index, 'row': int(row), 'col': int(col), 'bits': float(values[idx])})
        print('Top locations:', topk)
        if args.topk_json:
            Path(args.topk_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.topk_json, 'w') as f:
                json.dump(topk, f, indent=2)
            print(f'Saved top-K list to {args.topk_json}')


if __name__ == '__main__':
    main()
