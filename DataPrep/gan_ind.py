#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

def collect_pairs(root: Path, particles):
    """
    Returns a dict:
        {
          "contained_proton": [(number1, number2), ...],
          "exiting_proton":   [(number1, number2), ...],
          "muons":            [(number1, number2), ...],
        }
    Only (number1, number2) are stored, both as ints.
    """
    meta = {p: {} for p in particles}

    for particle in particles:
        base = root / particle
        print(base)
        index = {}
        # Files should be at <root>/<particle>/<number1>/<number2>.npy
        # Use glob for exactly one directory level then .npy files
        for subdir in base.glob("*"):
            print(subdir)
            number1 = int(subdir.stem)
            for file in subdir.glob("*.npz"):
                try:
                    number2 = int(file.stem)
                except ValueError:
                    # Skip anything that isn't strictly numeric
                    continue
                index.setdefault(number1, []).append(number2)
            #sort and deduplicate
            #index[number1] = sorted(set(index[number1]))


        meta[particle] = index

    return meta

def main():
    parser = argparse.ArgumentParser(
        description="Recursively read <particle>/<number1>/<number2>.npy and pickle only (number1, number2)."
    )
    parser.add_argument("root", type=Path, help="Root directory containing particle folders")
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("gan_ind.pkl"),
        help="Output pickle path (default: gan_ind.pkl)"
    )
    parser.add_argument(
        "--particles", nargs="+",
        default=["proton_contained", "proton_exiting", "muon"],
        help="Particle folder names to scan (default: contained_proton exiting_proton muons)"
    )
    args = parser.parse_args()


    meta = collect_pairs(args.root, args.particles)

    # Save only the numbers as requested
    with args.out.open("wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Brief summary
    for p in args.particles:
        print(f"{p}: {len(meta[p])} pairs")
        for k, v in meta[p].items():
            print(f"p{p}: {k}: {len(v)}")
            
    print(f"Saved to: {args.out.resolve()}")

if __name__ == "__main__":
    main()


