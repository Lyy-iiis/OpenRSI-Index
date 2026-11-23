# Visibility-Weighted Midpoint Synthesis with Frozen CPU Motion Fields

[Formal OpenRSI proposal](proposal.md) · [Experimental results](pilot.csv) · [Exact scored examples](manifest.tsv)

When two video frames are separated by a missing frame, the two endpoints can project different appearances to the same midpoint. This task asks an agent to decide which projected intensity to trust, under fixed bidirectional CPU motion estimates. The answer is scored against the actual withheld image.

OpenCV provides the motion estimator; the pilot supplies midpoint reconstruction measurements. On 20 triplets from five pilot scenes, scene-mean squared error was **297.8677** for a cross-dissolve, **135.5366** for equal splatting and **134.9456** for cycle weighting. Cycle weighting won in three scenes and lost in two. The inconsistent scene effects provide a concrete test of learned visibility cues.

## Reproduction

From the parent `proposals/cpu-motion` directory, use Python 3.13. Obtain the official Sintel ZIP on the authorized local research host. Extraction defaults to the two public reproduction splits; `--splits dev` builds the development-only partition.

```sh
python -m pip install -r requirements.txt
python inventory.py /path/to/MPI-Sintel-complete.zip midpoint --extract /path/to/sintel
python midpoint/check.py
python midpoint/pilot.py /path/to/sintel dev dev-out.csv
python midpoint/pilot.py /path/to/sintel pilot pilot-out.csv
python report.py dev-out.csv pilot-out.csv
```

The complete input inventory covers 96 validation and 96 final three-frame windows. Formal Judge partitions have only been checked against the archive index in this pilot. Public pilot measurements are exploratory. No GPU, remote scoring service or dataset redistribution is required.
