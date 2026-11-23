# When to Re-anchor a Sparse CPU Point Track

[Formal OpenRSI proposal](proposal.md) · [Experimental results](pilot.csv) · [Exact scored examples](manifest.tsv)

In a short video, matching each new frame to the previous one accumulates drift, while matching back to the first frame can fail when appearance changes. This task asks an agent to choose the temporal reference for each point, using only evidence available at that time.

OpenCV ships sparse LK and a tracking example; neither reports the score for this five-step task. On ten six-frame clips across five pilot scenes, final scene-mean capped endpoint loss was **3.3908 px** for incremental, **4.6112 px** for anchor preference and **3.2339 px** for patch choice. Patch choice beat incremental in all five scenes. The point detector used qualityLevel=0.01 after a coverage check showed too few eligible trajectories at 0.3; the pilot is exploratory.

## Reproduction

From the parent `proposals/cpu-motion` directory, use Python 3.13. Obtain the official Sintel ZIP on the authorized local research host. Extraction defaults to the two public reproduction splits; `--splits dev` builds the development-only partition.

```sh
python -m pip install -r requirements.txt
python inventory.py /path/to/MPI-Sintel-complete.zip tracking --extract /path/to/sintel
python tracking/check.py
python tracking/pilot.py /path/to/sintel dev dev-out.csv
python tracking/pilot.py /path/to/sintel pilot pilot-out.csv
python report.py dev-out.csv pilot-out.csv
```

The complete input inventory covers 90 validation and 90 final six-frame clips. Formal Judge partitions have only been checked against the archive index in this pilot. Public pilot measurements are exploratory. No GPU, remote scoring service or dataset redistribution is required.
