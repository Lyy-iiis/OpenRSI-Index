"""Verify the pilot subset, optionally extracting it from the official ZIP first."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def pair_members(scene, i):
    for kind, frame, ext in (("final", i, "png"), ("final", i + 1, "png"),
                             ("flow", i, "flo"), ("invalid", i, "png"),
                             ("occlusions", i, "png")):
        yield f"training/{kind}/{scene}/frame_{frame:04d}.{ext}"


def members():
    names = set()
    for line in Path(__file__).with_name("pairs.tsv").read_text().splitlines():
        _, scene, index = line.split()
        names.update(pair_members(scene, int(index)))
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="parent of training/")
    parser.add_argument("--archive", type=Path, help="MPI-Sintel-complete.zip")
    args = parser.parse_args()
    files = members()
    if args.archive:
        with zipfile.ZipFile(args.archive) as archive:
            for name in files:
                out = args.root / name
                data = archive.read(name)  # Checks member CRC.
                if out.exists():
                    if out.read_bytes() != data:
                        raise ValueError(f"existing local file differs: {out}")
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode("ascii") + b"\x00")
        digest.update(hashlib.sha256((args.root / name).read_bytes()).digest())
    expected = Path(__file__).with_name("data.sha256").read_text().strip()
    if digest.hexdigest() != expected:
        raise ValueError(f"dataset fingerprint mismatch: {digest.hexdigest()}")
    print(f"Verified {len(files)} files. SHA-256 {expected}")


if __name__ == "__main__":
    main()
