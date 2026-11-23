# Adaptive CPU optical flow under a tail-latency budget

Where should a CPU optical-flow algorithm spend its last few milliseconds? We compare refinement at high-residual pixels with refinement at the same number of randomly chosen locations. The residual selector lowers endpoint error (EPE, in pixels) on two datasets. Its absolute gains are small and consistent across the sampled scenes. All measurements use one CPU thread and zero GPUs.

## Location selection and transfer

| Evaluation set | FAST EPE | Random EPE¹ | Residual EPE | Grid sites checked² | Residual vs FAST, P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sintel Final, five held-out scenes / 40 pairs | 4.46632 | 4.45886 | 4.43773 | 9.37% | +6.8% |
| Middlebury, eight scenes / eight pairs | 0.81117 | 0.80736 | 0.80077 | 16.40% | +9.4%, +8.8% on repeat |

¹ Mean EPE across three fixed random seeds. ² The random and residual methods check exactly the same number of eligible centers **on every pair**. Random masks are supplied before timing. On Sintel, residual P95 was 5.868 ms and the random controls ranged from 5.852 to 5.885 ms. On Middlebury these were 3.729 ms and 3.697–3.855 ms on the first run. [Paired experiment protocol](allocation.md)

On Sintel, the random control recovered about one quarter of the residual method’s improvement over FAST. On Middlebury it recovered about one third. Comparing the two allocations accounts for the remaining 74% and 63% of the measured gain, respectively. The comparison tests the value of these selected locations under this particular patch correction. It leaves open whether a stronger confidence estimate can do better.

The candidate was frozen using Sintel development scenes. It beat both FAST and the three-seed random mean in each of the eight Middlebury scenes. Excluding RubberWhale, which was used for an early FAST timing probe, pooled EPE was 0.85685 for FAST, 0.85263 for random and 0.84557 for residual selection. Middlebury provides one labeled pair per scene here, so the P95 is the slowest of eight pair medians. Timing comes from one Apple M4 Max. [Scene-level measurements](results/middlebury.csv) · [Middlebury benchmark](https://vision.middlebury.edu/flow/data/)

## Experiment

The reference is OpenCV DIS FAST. The candidate samples its output on a four-pixel grid. Where a pixel's predicted destination has a high brightness residual, the algorithm tests the motion of four neighboring points. A five-sample patch cost decides whether to borrow a neighboring motion. The algorithm interpolates accepted corrections back to the original resolution. This requires only the two grayscale input frames. The frozen scene split, metrics, controls and success criteria are in [the protocol](protocol.md). The upstream DIS implementation is pinned to [OpenCV 4.12.0](https://github.com/opencv/opencv/blob/49486f61fb25722cbcf586b7f4320921d46fb38e/modules/video/src/dis_flow.cpp).

| Method | All EPE ↓ | Occluded EPE ↓ | Boundary band EPE ↓ | P95 ms, run 1 | P95 ms, run 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DIS FAST | 4.4663 | 30.8359 | 8.7691 | 5.553 | 5.721 |
| DIS MEDIUM | 3.8947 | 31.9198 | 8.1051 | 29.369 | 30.015 |
| FAST, one extra variational iteration | 4.4890 | 31.2232 | 8.8641 | 6.175 | 6.281 |
| FAST, check the full grid | 4.4188 | 30.7952 | 8.6352 | 6.219 | 6.390 |
| FAST, residual-gated checks | 4.4377 | 30.8026 | 8.6827 | 5.968 | 6.142 |

Both held-out processes produced identical error sums and valid pixel counts. Checking the full grid yields a larger accuracy gain and exceeds the 10% latency allowance on both runs. Residual gating kept part of that gain within budget. The gated method reduced overall EPE by 0.64% and boundary-band EPE by 0.99%. The measured improvement is small, particularly on sleeping_2. All five scene results and the development results can be regenerated from the [per-pair measurements](results/) with the command below.

Two development experiments also shaped the candidate. On the 40 development pairs, reducing FAST's variational iterations from five to two raised EPE from 2.3850 to 2.5319. Seeding that cheaper solver with confident coarse tile translations gave 2.5338. Adding residual-gated neighbor checks after the two-iteration solver gave 2.5107. We retained the original five iterations and froze the postprocessing method before scoring the holdout. Public training scenes, correlated adjacent frames and a single host limit conclusions about generalization and latency. The pilot measures two-frame inference and does not establish end-to-end video throughput.

## Reproduce

Requirements are CMake 3.20+, a C++17 compiler, Python 3.8+ (standard library) and the OpenCV 4.12.0 C++ development libraries for core, imgproc, imgcodecs and video. The measured build used Release mode, AppleClang 21 and an Apple M4 Max. OpenCV OpenCL was disabled. The single-thread limit applies to optical-flow inference.

Run these commands from this directory. To inspect the committed measurements without compiling or downloading data:

```sh
python3 summarize.py results/*.csv
```

Download the official complete dataset for your research use. The preparation command extracts and fingerprints only the 360 files named by the pilot. If the archive is already extracted, omit the archive option. Keep dataset files outside the checkout. See the [Sintel downloads](https://sintel.is.tuebingen.mpg.de/downloads) and [dataset terms](https://sintel.is.tuebingen.mpg.de/license).

```sh
python3 prepare_data.py /path/to/sintel --archive /path/to/MPI-Sintel-complete.zip
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
build/flow_bench /path/to/sintel pairs.tsv dev 9 > build/dev.csv
build/flow_bench /path/to/sintel pairs.tsv holdout 9 > build/holdout.csv
build/flow_bench /path/to/sintel pairs.tsv holdout 9 > build/holdout-repeat.csv
python3 summarize.py build/dev.csv build/holdout.csv build/holdout-repeat.csv
```

To reproduce the paired controls, download the Middlebury color two-frame and flow archives listed in [the dataset index](https://vision.middlebury.edu/flow/data/). With both ZIP files in the working directory, verify them and extract both into the same dataset root. The benchmark reads the native `other-data` and `other-gt-flow` directories. Images with no public flow labels are excluded by the manifest.

```sh
shasum -a 256 -c middlebury.sha256
unzip -q other-color-twoframes.zip -d /path/to/middlebury
unzip -q other-gt-flow.zip -d /path/to/middlebury
build/flow_bench /path/to/sintel pairs.tsv dev 9 allocation > build/allocation-dev.csv
build/flow_bench /path/to/sintel pairs.tsv holdout 9 allocation > build/allocation-holdout.csv
build/flow_bench /path/to/middlebury middlebury.tsv middlebury 9 allocation > build/middlebury.csv
build/flow_bench /path/to/middlebury middlebury.tsv middlebury 9 allocation > build/middlebury-repeat.csv
python3 summarize.py build/allocation-*.csv build/middlebury*.csv
```

On the measured host the Homebrew shared video module had an unresolved optional DNN dependency. The build used the matching static video archive. To reproduce that link configuration, add `-DDIS_VIDEO_LIBRARY=/opt/homebrew/opt/opencv/lib/libopencv_video.a` to the CMake configure command. Leave the machine idle during timing. The synthetic CTest checks all five methods against known pixel counts and error sums, including invalid pixels, the occlusion mask and the boundary band.

## OpenRSI scope

A concrete next research task is to design a refinement selector that improves on the frozen residual gate on new scene families. Keep FAST, the four neighboring motion hypotheses and the patch update fixed. Give the selector the same per-pair quota and a CPU latency cap of 1.10× FAST. Competing hypotheses include motion discontinuity, local texture ambiguity and forward-warp residual. Charge the selector for its own computations; expose validation EPE, boundary EPE and latency after each iteration. Judge the final candidate on separately held-back scenes. The two public pilot data sets provide baselines for development and replication. The initial reference and candidate are in [flow.cpp](flow.cpp); [benchmark.cpp](benchmark.cpp) supplies the evaluator for this pilot. A future judged run should compare against both the FAST reference and this pilot candidate. The formal proposal would also require a contributor-confirmed project background, a separately provisioned final split, and independent packaging and validation. The repository commit contains the proposal evidence and local benchmark.

## Formal split for OpenRSI #390

Each scene supplies the 49 adjacent frame pairs from 1→2 through 49→50. For every pair the scorer requires both Final images and the starting frame's forward flow, invalid mask and occlusion mask.

| Split | Sintel training sequences | Distinct pairs | Distinct required ZIP members |
| --- | --- | ---: | ---: |
| Work | alley_1, ambush_5, bamboo_1, market_2, shaman_2 | 245 | 985 |
| Validation | bandage_2, cave_4 | 98 | 394 |
| Final test | sleeping_1, temple_3 | 98 | 394 |

[Formal pairs](formal-pairs.tsv) is a headerless input to the benchmark with columns for split, scene and starting frame. The [member inventory](formal-members.tsv) records each required path, uncompressed length and CRC32 from the official ZIP directory. Adjacent pairs share an endpoint. Per scene this gives 50 Final images and 49 files of each of the three annotation types. Mask 50 is unused because frame 50 has no outgoing evaluated pair.

On an authorized installation, verify the full ZIP before provisioning. The command checks both pinned manifests, reads all 1,773 required members and verifies their CRCs. For a directory-only inventory audit, add `--directory-only`.

```sh
python3 verify_formal.py /path/to/MPI-Sintel-complete.zip
```

Run the allocation baseline with `formal-pairs.tsv` and the split argument `work`, `validation` or `test`. If any required pair is unavailable or excluded, stop provisioning without a score. Restore the exact inputs and repeat verification before restarting. The split, denominator and scene weights remain fixed. Existing pilot measurements use the separate pilot manifest.

## References

- Till Kroeger, Radu Timofte, Dengxin Dai and Luc Van Gool. *Fast Optical Flow Using Dense Inverse Search*. ECCV 2016. [Paper](https://arxiv.org/abs/1603.03590)
- Daniel J. Butler, Jonas Wulff, Garrett B. Stanley and Michael J. Black. *A Naturalistic Open Source Movie for Optical Flow Evaluation*. ECCV 2012. [MPI Sintel](https://sintel.is.tuebingen.mpg.de/)
