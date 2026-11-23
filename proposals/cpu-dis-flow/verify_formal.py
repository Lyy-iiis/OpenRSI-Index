"""Verify the formal split manifests and required members of MPI-Sintel-complete.zip."""
import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from prepare_data import pair_members

HERE = Path(__file__).resolve().parent
SCENES = {
    "work": ("alley_1", "ambush_5", "bamboo_1", "market_2", "shaman_2"),
    "validation": ("bandage_2", "cave_4"),
    "test": ("sleeping_1", "temple_3"),
}


def inventory(archive):
    pairs = []
    required = set()
    for split, scenes in SCENES.items():
        for scene in scenes:
            for start in range(1, 50):
                pairs.append(f"{split}\t{scene}\t{start}\n")
                required.update(pair_members(scene, start))
    counts = Counter(archive.namelist())
    rows = ["member\tbytes\tcrc32\n"]
    for name in sorted(required):
        if counts[name] != 1:
            raise ValueError(f"{name}: expected one archive member, found {counts[name]}")
        item = archive.getinfo(name)
        rows.append(f"{name}\t{item.file_size}\t{item.CRC:08x}\n")
    return "".join(pairs), "".join(rows), sorted(required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--directory-only", action="store_true",
                        help="check exact inventory without reading compressed payloads")
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        pairs, members, names = inventory(archive)
        for filename, expected in (("formal-pairs.tsv", pairs), ("formal-members.tsv", members)):
            if (HERE / filename).read_bytes() != expected.encode("ascii"):
                raise ValueError(f"{filename}: differs from the formal split or archive metadata")
        if not args.directory_only:
            for name in names:
                # ZipFile.read reads through EOF and verifies the member CRC.
                if len(archive.read(name)) != archive.getinfo(name).file_size:
                    raise ValueError(f"{name}: member size mismatch")
    scope = "directory" if args.directory_only else "directory and payloads"
    print(f"Verified {len(pairs.splitlines())} pairs, {len(names)} distinct members ({scope}).")


if __name__ == "__main__":
    main()
