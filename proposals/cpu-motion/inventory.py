"""Compile or verify exact example and archive-member manifests from the official ZIP."""
import argparse
import csv
import io
from pathlib import Path
from zipfile import ZipFile

SCENES = {
    "dev": "alley_1 ambush_5 bamboo_1 market_2 shaman_2",
    "pilot": "bandage_1 cave_2 mountain_1 sleeping_2 temple_2",
    "validation": "bandage_2 cave_4",
    "test": "sleeping_1 temple_3",
}


def table(columns, rows):
    s = io.StringIO()
    writer = csv.writer(s, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return s.getvalue()


def inventory(z, task):
    # Work and pilot are fixed subsets. Val/test use all complete windows within
    # these predeclared 50-frame sequences. A missing official member is fatal.
    span = {"confidence": 1, "midpoint": 2, "tracking": 5}[task]
    samples, members = [], set()
    for split, names in SCENES.items():
        for scene in names.split():
            starts = range(1, 51-span)
            if split in ("dev", "pilot"):
                starts = {"confidence": (4,5,15,16,26,27,37,38),
                          "midpoint": (4,15,26,37), "tracking": (4,26)}[task]
            for start in starts:
                samples.append((split, scene, start, span))
                for i in range(start, start+span+1):
                    members.add(f"training/final/{scene}/frame_{i:04d}.png")
                    if i == start + span or task == "midpoint":
                        continue
                    kinds = ("flow","invalid","occlusions") if task == "tracking" else ("flow","invalid")
                    for kind in kinds:
                        ext = "flo" if kind == "flow" else "png"
                        members.add(f"training/{kind}/{scene}/frame_{i:04d}.{ext}")
    records = []
    for name in sorted(members):
        item = z.getinfo(name)
        records.append((name, item.file_size, f"{item.CRC:08x}"))
    return table(("split","scene","start","transitions"), samples), table(("member","bytes","crc32"), records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("task", choices=("confidence","midpoint","tracking"))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--extract", type=Path, help="output root for locally licensed dataset files")
    parser.add_argument("--splits", nargs="+", choices=tuple(SCENES), default=["dev","pilot"])
    args = parser.parse_args()
    folder = Path(__file__).parent / args.task
    with ZipFile(args.archive) as z:
        manifests = inventory(z, args.task)
    for name, content in zip(("manifest.tsv", "members.tsv"), manifests):
        path = folder / name
        if args.write:
            path.write_text(content)
        elif path.read_text() != content:
            raise ValueError(f"inventory mismatch: {path}")
        print(f"{name}: {len(content.splitlines())-1} rows")
    if args.extract:
        allowed = {scene for split in args.splits for scene in SCENES[split].split()}
        with ZipFile(args.archive) as z:
            for row in csv.DictReader(io.StringIO(manifests[1]), delimiter="\t"):
                name = row["member"]
                if name.split("/")[2] in allowed:
                    target = args.extract / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    data = z.read(name)
                    if target.exists() and target.read_bytes() != data:
                        raise ValueError(f"local file differs: {target}")
                    if not target.exists():
                        target.write_bytes(data)


if __name__ == "__main__":
    main()
