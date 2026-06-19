#!/usr/bin/env python3
"""
Convert training_progress/*.png snapshots into a timelapse video.

Usage:
    python make_video.py                        # default output: training_timelapse.mp4
    python make_video.py -o my_video.mp4        # custom output path
    python make_video.py --fps 10               # custom frame rate (default: 5)
    python make_video.py --input-dir some/path  # custom input directory
"""
import argparse
import glob
import os
import subprocess
import sys


def make_video(input_dir='training_progress', output='training_timelapse.mp4', fps=5):
    pattern = os.path.join(input_dir, 'step_*.png')
    frames = sorted(glob.glob(pattern))

    if not frames:
        print("No frames found matching: {}".format(pattern))
        sys.exit(1)

    print("Found {} frames in '{}'".format(len(frames), input_dir))
    print("Output: {}  ({} fps, ~{:.1f}s video)".format(output, fps, len(frames) / fps))

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', pattern,
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # ensure even dimensions for h264
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',  # broad compatibility (QuickTime, browsers, etc.)
        output,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg error:\n", result.stderr)
        sys.exit(1)

    print("Done: {}".format(os.path.abspath(output)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Turn training PNGs into a video.')
    parser.add_argument('--input-dir', default='training_progress',
                        help='Directory containing step_*.png files (default: training_progress)')
    parser.add_argument('-o', '--output', default='training_timelapse.mp4',
                        help='Output video path (default: training_timelapse.mp4)')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second (default: 5)')
    args = parser.parse_args()

    make_video(input_dir=args.input_dir, output=args.output, fps=args.fps)
