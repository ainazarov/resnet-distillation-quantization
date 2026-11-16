# Galaxy S24+ Food-101 benchmark

On a fresh checkout, first [export the model files](#regenerate-the-packaged-assets). Then open this directory in Android Studio, select the connected phone, choose **Run**, and tap **Run benchmark**. The app packages both models, their labels and 101 Food-101 test JPEGs for offline use. It displays median forward time, p95 JPEG-to-top-3 time and agreement with Python for each model. [Android device setup](https://developer.android.com/studio/run/device)

## Measured on the Samsung Galaxy S24+

The November 16, 2026 run used the physical SM-S926B (SoC identifier `s5e9945`), Android 16, with USB connected. Power saving was off and Android reported thermal status 0 at both ends of the pass. The app ran in the foreground without a debugger. The APK is installed on the phone.

| Model | Full Food-101 test top-3 | Median forward on phone | Phone JPEG-to-top-3 p95 | Ordered top-three agreement with Python |
|---|---:|---:|---:|---:|
| ResNet-50 teacher | 95.18% | 50.92 ms | 67.57 ms | 101/101 |
| Distilled ResNet-18 | 93.63% | 22.56 ms | 33.51 ms | 101/101 |

We gave up **1.55 percentage points of top-3 accuracy** and reduced **p95 JPEG-to-top-3 latency from 67.57 ms to 33.51 ms** on the phone, a 50.4% reduction. For example, on `fried_rice/3331345`, both Android and Python returned fried rice, paella and garlic bread for the teacher, and fried rice, paella and carrot cake for the student. [Phone measurements and predictions](../results/android/galaxy-s24-plus.json) · [Export and numerical verification](../results/android/export.json)

## What the app measures

The benchmark runs each model on the CPU in FP32, batch size one, with four XNNPACK threads. It warms each model on 20 photos and then times all 101 photos once per model. It alternates which model goes first on each photo. JPEGs are staged as local files before warmup.

- **Model forward** measures the synchronous ExecuTorch `Module.forward` call with a normalized input tensor ready.
- **JPEG to top three** starts before opening the JPEG and ends after looking up all three dish names. It includes decode, resize, center crop, normalization, tensor preparation, forward and ranking. Model loading and UI drawing are outside the interval.
- **Python agreement** compares the three predicted class indices in order on every image. **Test accuracy** is the separate measurement on the full 25,250-image Food-101 test split.

## Build and run over USB

The project uses JDK 17, Android SDK 35, Gradle 8.9, and the arm64 ExecuTorch 1.5.1 Android runtime. The Run button starts a worker thread. The app checks that no debugger is attached. [ExecuTorch Android guide](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)

From this directory in PowerShell, with the Android SDK configured and `adb` on the path:

```powershell
.\gradlew.bat :app:assembleDebug
adb -d install -r app/build/outputs/apk/debug/app-debug.apk
adb -d shell am start -S -n org.fooddiary.benchmark/.MainActivity --ez benchmark true
```

Once the two summaries appear, retrieve the phone's JSON report with the following command. The report contains the 202 per-request measurements, top-three predictions and Python references, model hashes, thread count, power-saving mode and thermal state.

```powershell
adb -d exec-out run-as org.fooddiary.benchmark cat files/benchmark.json | Set-Content -Encoding utf8 phone-benchmark.json
```

## Regenerate the packaged assets

The repository includes the trained student checkpoints, labels, 101 benchmark JPEGs and their Python predictions. Download the teacher checkpoint using the commands in the [project README](../README.md#reproduce-the-experiment). From the Android directory, run `python export_models.py` in an ExecuTorch 1.5.1 environment to generate the PTE files before building the APK. The export checks logits and ordered predictions against Python on five photos per model.

To regenerate the JPEG subset and reference predictions from the full dataset, run `python prepare_photos.py --images ../data/food-101/images` from this directory in the desktop PyTorch environment. The benchmark manifest selects one photo per class. The benchmark photos come from Food-101 and belong to their original photographers. [Food-101 dataset and licensing information](https://huggingface.co/datasets/ethz/food101#licensing-information)

## Distilled ResNet-18 precision test

The **Food-101 FP32 vs INT8** app compares the distilled student's FP32 export and its statically calibrated INT8 export. It uses the same 101 JPEGs, preprocessing, warmup, timing boundaries and XNNPACK thread count as the teacher/student benchmark. It takes FP32 image tensors for both exports. The reference contains predictions from each exported program running in Python. Each Android export is checked against its own reference and SHA-256. A tie in logit value goes to the lower class index.

### Measured on the Galaxy S24+

| Distilled ResNet-18 export | PTE size (decimal MB) | Test top-1 | Test top-3 | Phone median forward | Phone JPEG-to-top-3 p95 |
|---|---:|---:|---:|---:|---:|
| FP32 | 44.91 | 82.444% | 93.628% | 21.43 ms | 27.90 ms |
| Static INT8 | 11.31 | 82.372% | 93.620% | 4.99 ms | 11.60 ms |

**INT8 reduced p95 JPEG-to-top-3 latency by 58.4% and export size by 74.8%.** On the full test set INT8 put the correct dish in the top three for 23,639 photos versus 23,641 for FP32. It gained 60 top-three hits and lost 62. Third and fourth place had equal logits on 1,684 INT8 photos, so the deployed tie-break rule matters when reproducing these scores. [FP32 accuracy](../results/android_ptq/resnet18_fp32_test.json) · [INT8 accuracy](../results/android_ptq/resnet18_int8_test.json)

The table pools three paired passes (303 timed requests per export). Each pass started a fresh app process and warmed each model for 20 requests. There were 20 seconds between passes. Power saving was off, the debugger was detached, and Android reported thermal status 0 before and after every pass. Individual p95s were 27.55, 28.32 and 27.93 ms for FP32, and 11.06, 11.97 and 11.50 ms for INT8. All 303 ordered lists per model matched that export's Python runtime reference. [Pooled results](../results/android_ptq/galaxy-s24-plus/summary.json) · [Per-request timing and predictions](../results/android_ptq/galaxy-s24-plus/)

### Calibration and export

Both models use the same selected distilled checkpoint. The FP32 file is byte-identical to the previous student export. For static PTQ, the calibration set comprises 505 original fit-set JPEGs selected by seeded hash, five per dish. Each photo receives the checkpoint's 255-pixel short-side resize, 224 × 224 center crop and ImageNet normalization. `XNNPACKQuantizer` uses fixed activation scales calibrated by `prepare_pt2e` observers and per-output-channel INT8 weights. After `convert_pt2e` and lowering, inspection of the serialized graph verified all 20 convolutions and the final linear layer use INT8 in one XNNPACK partition. [Calibration manifest](../results/android_ptq/calibration.csv) · [Export audit](../results/android_ptq/export.json) · [ExecuTorch PT2E quantization](https://docs.pytorch.org/executorch/stable/backends/xnnpack/xnnpack-quantization.html)

With the original Food-101 dataset at `../data/food-101`, run the following from this directory. Use the ExecuTorch 1.5.1 export environment for quantization and evaluation. The full-test evaluator uses the actual serialized files.

```powershell
python prepare_calibration.py
python quantize_student.py
python evaluate_export.py app/src/precision/assets/models/resnet18_fp32.pte
python evaluate_export.py app/src/precision/assets/models/resnet18_int8.pte
```

In Android Studio, select the `precision` build variant, select the phone and run. Alternatively, use the following from this directory. The app installs alongside the original benchmark.

```powershell
.\gradlew.bat :app:assemblePrecision
adb -d install -r app/build/outputs/apk/precision/app-precision.apk
adb -d shell am start -S -n org.fooddiary.benchmark.ptq/org.fooddiary.benchmark.MainActivity --ez benchmark true
```

After both summaries appear, retrieve that run's per-photo measurements:

```powershell
adb -d exec-out run-as org.fooddiary.benchmark.ptq cat files/benchmark.json | Set-Content -Encoding utf8 phone-precision.json
```

To reproduce the three-pass measurement with the installed precision APK, keep the phone unlocked and run `python run_precision.py --out ../results/android_ptq/repeat` with a fresh output directory. The runner verifies the JPEG IDs, model and reference hashes, XNNPACK delegation, batch size and thread count before pooling the measurements.

### Supplemental Android stage profiling

The HW1 report uses one additional foreground pass comparing the FP32 and INT8 students. A timer after Android bitmap decoding, sRGB pixel extraction and bitmap release separates JPEG-to-RGB from resize, center crop, normalization and tensor preparation. Forward starts with the input ready. Output readback, ranking and label lookup end the request. Each row's four intervals sum to its end-to-end time. Stage shares in the [profile summary](../results/android_ptq/stages/summary.json) use the sum of each stage's durations divided by the sum of all request durations. The summary also records the exclusion of a teacher/student pass that moved to the background during warmup.

To capture a fresh precision profile, run the following and wait for the two model summaries to appear before retrieving the file. Use `org.fooddiary.benchmark` as the package for the teacher/student pair.

```powershell
adb -d shell am start -S -W -n org.fooddiary.benchmark.ptq/org.fooddiary.benchmark.MainActivity --ez benchmark true --ez stages true
adb -d exec-out run-as org.fooddiary.benchmark.ptq cat files/stage-profile.json | Set-Content -Encoding utf8 phone-stages.json
```

## Series C · CE and distilled students on the phone

The `control` APK compares the validation-selected plain ResNet-18 and distilled ResNet-18, both in FP32. It uses the same XNNPACK build, 101 original JPEGs, four CPU threads, 20 warmups, rotating per-photo order, and batch size one as series A. The plain model's labels come from the saved full-test predictions. `export_control.py` verifies their provenance and compares the eager and exported CE logits on five photos. The APK checks the selected PTE hashes before timing.

From this directory, use the ExecuTorch 1.5.1 environment, build and install the APK, and keep the activity visible for the entire benchmark.

```powershell
python export_control.py
.\gradlew.bat :app:assembleControl
adb -d install -r app/build/outputs/apk/control/app-control.apk
adb -d shell am start -S -W -n org.fooddiary.benchmark.ce/org.fooddiary.benchmark.MainActivity --ez benchmark true
```

After the two timing summaries appear, retrieve the result.

```powershell
adb -d exec-out run-as org.fooddiary.benchmark.ce cat files/benchmark.json | Set-Content -Encoding utf8 phone-ce-control.json
```

The accepted foreground run on 16 November 2026 used a Galaxy S24+ (SM-S926B), ExecuTorch 1.5.1 and XNNPACK. Power saving was off, thermal status was 0 at both ends and the debugger was detached. Each model ran once on every JPEG after warmup. The rows below use the 50th and 95th percentiles of the 101 individual requests for each model.

| Series C, FP32 | PTE size, decimal MB | Median forward, ms | p95 JPEG to top 3, ms | Ordered top 3 matches with Python |
|---|---:|---:|---:|---:|
| Plain CE ResNet-18 | 44.91 | 21.36 | 27.52 | 101/101 |
| Distilled ResNet-18 | 44.91 | 21.32 | 27.42 | 101/101 |

The [raw measurements](../results/android_ce/galaxy-s24-plus.json) contain all 202 requests. The [export manifest](../results/android_ce/export.json) identifies both checkpoints and PTEs by SHA-256. The [verification record](../results/android_ce/verification.json) documents the foreground interval and independent percentile checks. An [earlier attempt](../results/android_ce/excluded-background.json) was excluded because Android paused the benchmark activity during measurement. The reported run started after the user reopened the app and kept it visible.
