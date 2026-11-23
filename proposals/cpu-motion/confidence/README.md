# Calibrated Failure Probabilities for Frozen CPU Optical Flow

[Formal OpenRSI proposal](proposal.md) · [Experimental results](ambiguity-pilot.csv) · [Exact scored examples](manifest.tsv)

The revised task scores the samples where the frozen cycle-and-appearance lookup gives a failure probability in [0.25, 0.75]. On these samples a third cue, local motion spread, reduced scene-macro Brier from 0.2091978 to 0.1520691 on the five-scene public pilot. The formal target is to improve on this three-cue reference in each final scene. The cohort is fixed before labels and candidate scores are read. The [pilot CSV](ambiguity-pilot.csv) contains eligible counts, failure counts and loss sums for each frame and method. Original all-grid results remain available in [pilot.csv](pilot.csv).

## Reproduction

From the parent `proposals/cpu-motion` directory, use Python 3.13. Obtain the official Sintel ZIP on the authorized local research host. Extraction defaults to the two public reproduction splits; `--splits dev` builds the development-only partition.

```sh
python -m pip install -r requirements.txt
python inventory.py /path/to/MPI-Sintel-complete.zip confidence --extract /path/to/sintel
python confidence/check.py
python confidence/ambiguity.py /path/to/sintel dev dev-out.csv
python confidence/ambiguity.py /path/to/sintel pilot pilot-out.csv
python report.py dev-out.csv pilot-out.csv
```

The complete input inventory covers 98 validation and 98 final two-frame pairs. Formal Judge partitions have only been checked against the archive index in this pilot. Public pilot measurements are exploratory. No GPU, remote scoring service or dataset redistribution is required.
