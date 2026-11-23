# Calibrated Failure Probabilities for Frozen CPU Optical Flow

[Formal OpenRSI proposal](proposal.md) · [Experimental results](log-pilot.csv) · [Exact scored examples](manifest.tsv)

The primary score is binary predictive log loss in nats on samples where the original two-cue table assigns a failure probability in [0.25, 0.75]. Eligibility is fixed before candidate predictions. On the public pilot, the frozen three-cue reference scored 0.4705348 nats. Adding a Work-fitted correction for motion speed yielded 0.4811669 nats, with gains in two of five scenes. Brier is a secondary diagnostic. The [pilot CSV](log-pilot.csv) contains the sample counts, failure counts, summed negative log likelihood and summed squared error. The original [whole-grid Brier results](pilot.csv) and [conditional Brier experiment](ambiguity-pilot.csv) are also available.

## Reproduction

From the parent `proposals/cpu-motion` directory, use Python 3.13. Obtain the official Sintel ZIP on the authorized local research host. Extraction defaults to the two public reproduction splits; `--splits dev` builds the development-only partition.

```sh
python -m pip install -r requirements.txt
python inventory.py /path/to/MPI-Sintel-complete.zip confidence --extract /path/to/sintel
python confidence/check.py
python confidence/log_score.py /path/to/sintel dev dev-out.csv
python confidence/log_score.py /path/to/sintel pilot pilot-out.csv
python report.py dev-out.csv pilot-out.csv
```

The complete input inventory covers 98 validation and 98 final two-frame pairs. Formal Judge partitions have only been checked against the archive index in this pilot. Public pilot measurements are exploratory. No GPU, remote scoring service or dataset redistribution is required.
