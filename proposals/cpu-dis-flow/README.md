# Adaptive CPU optical flow under a tail-latency budget

This OpenRSI task proposal studies how to spend a small amount of CPU time correcting dense optical flow. On 40 held-out Sintel Final pairs, a residual-gated correction reduced mean endpoint error from 4.4663 to 4.4377 pixels. Error decreased in all five held-out scenes. P95 inference time increased by 7.5% on the first run and 7.4% on the repeat. The pilot used one CPU thread and zero GPUs.

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
python3 summarize.py results/dev.csv results/holdout.csv results/holdout-repeat.csv
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

On the measured host the Homebrew shared video module had an unresolved optional DNN dependency. The build used the matching static video archive. To reproduce that link configuration, add `-DDIS_VIDEO_LIBRARY=/opt/homebrew/opt/opencv/lib/libopencv_video.a` to the CMake configure command. Leave the machine idle during timing. The synthetic CTest checks all five methods against known pixel counts and error sums, including invalid pixels, the occlusion mask and the boundary band.

## OpenRSI scope

A research agent could change how uncertainty is estimated, which alternative motions are proposed and how corrections are interpolated under the same single-thread budget. The initial reference and candidate are in [flow.cpp](flow.cpp); [benchmark.cpp](benchmark.cpp) supplies the evaluator for this pilot. A future judged run should compare against both the FAST reference and this pilot candidate. The formal proposal would also require a contributor-confirmed project background, a separately provisioned final split, and independent packaging and validation. The repository commit contains the proposal evidence and local benchmark.

## References

- Till Kroeger, Radu Timofte, Dengxin Dai and Luc Van Gool. *Fast Optical Flow Using Dense Inverse Search*. ECCV 2016. [Paper](https://arxiv.org/abs/1603.03590)
- Daniel J. Butler, Jonas Wulff, Garrett B. Stanley and Michael J. Black. *A Naturalistic Open Source Movie for Optical Flow Evaluation*. ECCV 2012. [MPI Sintel](https://sintel.is.tuebingen.mpg.de/)
