# Calibrated Failure Probabilities for Frozen CPU Optical Flow

[Formal OpenRSI proposal](proposal.md) · [Experimental results](pilot.csv) · [Exact scored examples](manifest.tsv)

A flow estimate and a probability that it is wrong are different outputs. This task freezes every motion vector and asks whether an agent can estimate the probability of a greater-than-three-pixel error on a new scene. Downstream consumers can then use probabilities for decisions at different operating points.

Upstream tests the released flow estimator, without reporting Brier score for this derived task. Our five-scene pilot has scene-mean Brier 0.1359284 for constant prevalence, 0.0620296 for cycle only and **0.0587952** for cycle+appearance. The two-signal model improved on cycle-only in all five pilot scenes. The formal task must beat the stronger two-signal model.

## Reproduction

From the parent `proposals/cpu-motion` directory, use Python 3.13. Obtain the official Sintel ZIP on the authorized local research host. Extraction defaults to the two public reproduction splits; `--splits dev` builds the development-only partition.

```sh
python -m pip install -r requirements.txt
python inventory.py /path/to/MPI-Sintel-complete.zip confidence --extract /path/to/sintel
python confidence/check.py
python confidence/pilot.py /path/to/sintel dev dev-out.csv
python confidence/pilot.py /path/to/sintel pilot pilot-out.csv
python report.py dev-out.csv pilot-out.csv
```

The complete input inventory covers 98 validation and 98 final two-frame pairs. Formal Judge partitions have only been checked against the archive index in this pilot. Public pilot measurements are exploratory. No GPU, remote scoring service or dataset redistribution is required.
