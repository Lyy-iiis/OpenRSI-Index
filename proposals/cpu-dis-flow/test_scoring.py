"""Exercise the benchmark CLI with known flow, invalid pixels and occlusion masks."""
import csv
import io
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib


def png(path, width, height, pixels):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    rows = b"".join(b"\0" + bytes(pixels(x, y) for x in range(width)) for y in range(height))
    header = struct.pack(">2I5B", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def test(binary, root):
    width, height = 128, 96
    for kind in ("final", "invalid", "occlusions", "flow"):
        (root / "training" / kind / "scene").mkdir(parents=True)

    def frame(kind, name):
        return root / "training" / kind / "scene" / name

    for index in (1, 2):
        png(frame("final", f"frame_{index:04d}.png"), width, height,
            lambda x, y: (17 * x + 13 * y) % 256)
    png(frame("invalid", "frame_0001.png"), width, height, lambda x, y: 255 if x == 0 else 0)
    png(frame("occlusions", "frame_0001.png"), width, height, lambda x, y: 255 if x >= 64 else 0)
    # Identical images give zero estimated motion. Vary the reference by region
    # so that swapping the occlusion mask or including invalid pixels fails.
    truth = b"".join(struct.pack("<ff", 1000 if x == 0 else (3 if x >= 64 else 0),
                                 4 if x >= 64 else 0)
                     for y in range(height) for x in range(width))
    frame("flow", "frame_0001.flo").write_bytes(struct.pack("<fii", 202021.25, width, height) + truth)
    manifest = root / "pairs.tsv"
    manifest.write_text("dev\tscene\t1\n")
    args = [binary, str(root), str(manifest), "dev", "3"]
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert {r["method"] for r in rows} == {"fast", "medium", "extra-iteration", "full-grid", "adaptive"}
    for row in rows:
        assert int(row["n"]) == 127 * height
        assert int(row["occ_n"]) == 64 * height
        assert int(row["boundary_n"]) == 20 * height
        for key, expected in (("epe_sum", 64 * height * 5), ("occ_sum", 64 * height * 5),
                              ("boundary_sum", 10 * height * 5)):
            assert abs(float(row[key]) - expected) < 0.1, (row["method"], key, row[key], expected)
    frame("flow", "frame_0001.flo").unlink()
    assert subprocess.run(args, capture_output=True).returncode != 0


def test_middlebury(binary, root):
    width, height = 128, 96
    images = root / "other-data" / "scene"
    labels = root / "other-gt-flow" / "scene"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in (10, 11):
        png(images / f"frame{index}.png", width, height,
            lambda x, y: (17 * x + 13 * y + (45 if index == 11 and x > 64 else 0)) % 256)
    # Each component can independently mark a missing vector.
    samples = b"".join(struct.pack("<ff", 1e10 if x == 0 else 3, -1e10 if x == 1 else 4)
                       for y in range(height) for x in range(width))
    (labels / "flow10.flo").write_bytes(struct.pack("<fii", 202021.25, width, height) + samples)
    manifest = root / "middlebury.tsv"
    manifest.write_text("middlebury\tscene\t10\n")
    args = [binary, str(root), str(manifest), "middlebury", "3", "allocation"]
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    rows = {r["method"]: r for r in csv.DictReader(io.StringIO(result.stdout))}
    assert set(rows) == {"fast", "adaptive", "random-17", "random-29", "random-43"}
    assert 0 < int(rows["adaptive"]["checked"]) < int(rows["adaptive"]["eligible"])
    for method, row in rows.items():
        assert int(row["n"]) == (width - 2) * height
        assert int(row["occ_n"]) == int(row["boundary_n"]) == 0
        if method != "fast":
            for field in ("eligible", "checked"):
                assert row[field] == rows["adaptive"][field]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        test(sys.argv[1], Path(folder))
        test_middlebury(sys.argv[1], Path(folder))
