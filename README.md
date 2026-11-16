# ResNet distillation and INT8 quantization

**This repository includes an [accompanying PDF report (in Russian)](HW1_Report_Nazarov.pdf)** with architecture diagrams, profiling measurements and results from the Galaxy S24+.

ResNet-50 to ResNet-18 distillation and INT8 quantization for offline food recognition on Android. Includes Food-101 accuracy results and Galaxy S24+ latency benchmarks.

[Editable report](report/HW1_Food101_v5.md) · [Android benchmark app](android/README.md)

This experiment compares a Food-101 ResNet-50 teacher with two ImageNet-initialized ResNet-18 students. Top-3 accuracy measures how often the labeled dish appears among the three suggestions.

## Measured results

Accuracy uses all 25,250 official test photos. Latency uses 505 original JPEGs on the same laptop CPU, with warmed models, batch size one and FP32. End-to-end includes file access, decode, preprocessing, inference and lookup of the top three dish names.

| Selected model | Top-1 | Top-3 | Parameters | Median forward | End-to-end p95 |
|---|---:|---:|---:|---:|---:|
| ResNet-50 teacher | 85.90% | 95.18% | 23,714,981 | 98.93 ms | 146.25 ms |
| ResNet-18 plain | 81.33% | 92.98% | 11,228,325 | 41.93 ms | 65.84 ms |
| ResNet-18 distilled | 82.44% | 93.63% | 11,228,325 | 42.01 ms | 61.75 ms |

**Teacher versus distilled.** The distilled student returned the labeled dish among its suggestions on 23,641 photos, compared with 24,032 for the teacher. That costs 1.55 percentage points of top-3 coverage in exchange for 84.49 ms lower p95 latency (57.8%) and 52.7% fewer parameters. [Teacher scores](results/teacher-test.json) · [Final CPU measurements](results/final/summary.json)

**Plain versus distilled.** Distillation raised top-1 by 1.11 percentage points and top-3 by 0.65. On the same photos, it recovered 487 top-3 misses and lost 323 previously correct suggestions, for a net gain of 164. The two students have the same inference architecture and their median forwards differ by less than 0.1 ms. This comparison uses one training seed and one final latency pass. [Student scores](results/student_test.json) · [Per-request timings](results/final/timings_ms.csv)

**Galaxy S24+ deployment test.** On the physical phone with power saving off, the distilled student reduced median forward latency from 50.92 ms to 22.56 ms and p95 JPEG-to-top-3 latency from 67.57 ms to 33.51 ms. Both models matched Python's ordered top three on all 101 mobile test photos. [Mobile results and test app](android/README.md)

**Static INT8 on the distilled ResNet-18.** In a separate paired Galaxy S24+ test, static PTQ reduced the student's export size from 44.91 to 11.31 MB and its p95 JPEG-to-top-3 latency from 27.90 to 11.60 ms. Top-3 accuracy on all 25,250 official test photos went from 93.628% to 93.620%, a net loss of two correct top-three lists. [Calibration, accuracy and phone measurements](android/README.md#measured-on-the-galaxy-s24)

**Plain FP32 control on Galaxy S24+ (series C).** Both exports were 44,912,916 bytes. Median forward / p95 JPEG-to-top-3 were 21.36 / 27.52 ms for plain training and 21.32 / 27.42 ms for distillation. Each model processed the same 101 photos in the foreground with four XNNPACK CPU threads and batch size one, after 20 warmups. Each matched its own Python reference on all 101 ordered triples. The distillation gain is 164 additional correct top-three results on the full test set. [Per-request phone measurements](results/android_ce/galaxy-s24-plus.json) · [Export verification](results/android_ce/export.json)

## Data and teacher

The official Food-101 release provides 750 training photos and 250 test photos per dish. Both students use 700 training photos and 50 validation photos per dish. The official test set contains 25,250 photos. Training and accuracy evaluation use the original JPEGs. [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

The teacher has a dropout layer and a 2,048-to-101 linear head. Evaluation decodes to RGB, resizes the short side to 255 pixels, takes the center 224 × 224 crop, scales channels to [0, 1], and applies ImageNet mean and standard deviation. All 101 teacher labels match torchvision's actual dataset indices in order. The official `meta/classes.txt` contains the same names but lists `cheesecake` before `cheese_plate`. Torchvision and the teacher both use index 16 for `cheese_plate` and 17 for `cheesecake`. All reported scores and the diary CLI use that mapping. [Teacher model card and preprocessing](https://huggingface.co/ilian-hadzhidimitrov/food-classifier-resnet50#food-101-only-baseline-variant)

## Training objective

Both students start from the same saved ImageNet ResNet-18 with a freshly initialized 512-to-101 linear head. The plain student minimizes cross-entropy. The distilled student minimizes

$$
0.5\,\mathrm{CE}(y,z_s) + 0.5\,T^2\, D_{\mathrm{KL}}\!\left(\operatorname{softmax}(z_t/T)\,\Vert\,\operatorname{softmax}(z_s/T)\right),\qquad T=2.
$$

The frozen teacher runs in evaluation mode without gradients and receives the same augmented batch as the student. Each run fine-tunes all student layers for four epochs with AdamW, batch size 96, peak learning rate 0.0003, weight decay 0.01, and a cosine schedule after warmup. A fixed seed of 42 controls the split, initial head, epoch-specific shuffle, random resized crops and horizontal flips. Validation top-3 selects the checkpoint, with top-1 and then the earlier epoch breaking ties. Training uses BF16 autocast on the RTX 4070 Ti SUPER. Validation and test use FP32. [PyTorch knowledge distillation example](https://github.com/pytorch/tutorials/blob/main/beginner_source/knowledge_distillation_tutorial.py)

Both students selected epoch 4. The plain student's validation top-3 was 88.51% and the distilled student's was 90.00%. Four training and validation passes took 126 s for plain training and 230 s for distillation. The shared initialization and split hashes are recorded with every epoch's validation results. [Plain training](results/training/plain.json) · [Distilled training](results/training/distilled.json)

## CPU pilot and profiler

The original pilot recorded kernel shapes but did not label individual Conv2d modules. For v5 we profiled the trained teacher and distilled student on the same Intel CPU (Core i7-1165G7) with four CPU threads. Each model received 20 warmups and then the same 16 preprocessed FP32 input tensors, one at a time, in alternating model order. The full module name identifies the branch. Each named invocation encloses one timed `aten::mkldnn_convolution` event. The CSV gives the number of calls in the entire trace, calls per forward, and total backend milliseconds divided by the number of forwards. The trace contains 1,168 individually attributed convolution calls. [Per-layer CSV](results/pilot/layers/layers.csv) · [Group totals, hashes and input IDs](results/pilot/layers/summary.json) · [Named Chrome trace](results/pilot/layers/paired-trace.json.gz)

For example, in ResNet-50, `layer2.0.conv1` is a 1×1 convolution on the main path, while `layer2.0.downsample.0` is the 1×1 shortcut projection in the same block. The module names assign each `aten::mkldnn_convolution` event to its branch. The original pilot averaged 42.8 ms across all 36 of the teacher's 1×1 calls per image. In the separate, module-labeled run the 32 main-path calls took 48.61 ms and the four shortcut projections took 10.69 ms per image.

To repeat the measurement with the saved checkpoints, run `python report/profile_layers.py` from this directory. To redraw the JPEG-stage chart and convolution comparison, run `python report/plot_layer_groups.py`. The Android quantization section pairs the means from the [101-photo Android stage profile](results/android_ptq/stages/summary.json) with the end-to-end distribution from [three separate 101-photo passes](results/android_ptq/galaxy-s24-plus/). The [three benchmark screenshots](saved_screenshots/) show individual app runs on the Galaxy S24+.

The architecture diagram uses a paired run spanning the prepared input tensor through the 101 logits. Both trained networks use the same 16 tensors, warmup and Intel CPU settings as the convolution profile. The two shared input blocks (convolution + BatchNorm + ReLU, then MaxPool) display only the teacher's measured time. Each residual stage includes both paths, BatchNorm, ReLU and addition. The head spans global average pooling, flattening and the classifier. Stage changes use the student's mean minus the teacher's mean, divided by the teacher's mean for the percentage, before rounding. To repeat the component benchmark, run `python report/profile_components.py`. Redraw the diagram with `python report/draw_architecture.py --timings`. [Means for both networks](results/pilot/components/components.csv) · [Individual component timings](results/pilot/components/passes.csv) · [Measurement settings and input hashes](results/pilot/components/summary.json) · [Earlier four-stage capture](results/pilot/stages/stages.csv)

The parameter and MAC table separates the input block, four residual stages and 101-class classifier. To regenerate it from the named profile, run `python report/plot_architecture_counts.py`. The sums count BatchNorm's learned scale and bias as parameters and count one multiplication with accumulation as one MAC for convolutions and linear layers.

The deployment measurement runs on a Windows laptop with an Intel CPU (Core i7-1165G7), four intra-op threads and one inter-op thread. Each model processes the same 505 original test JPEGs, five per class, one photo at a time in FP32. File hashes are checked first, so the file cache is warm. Each model gets 20 warmup requests. Model order rotates for every measured photo. End-to-end timing starts at file open and stops when the three dish names are available.

| Before training | Median file open + RGB decode | Median resize, crop + normalize | Median forward | End-to-end p95 |
|---|---:|---:|---:|---:|
| Teacher ResNet-50 | 3.02 ms | 3.21 ms | 102.62 ms | 171.68 ms |
| Initial ResNet-18 | 2.97 ms | 3.18 ms | 42.56 ms | 75.15 ms |

Across the pilot requests, forward accounted for 93.6% of elapsed time for the teacher and 86.5% for the student. In three separately profiled forwards, the oneDNN convolution operator accounted for 73.1% and 65.8% of total self CPU time, respectively. ResNet-50 ran 53 convolution calls per photo and ResNet-18 ran 20. The median preprocessing cost changed by only 0.03 ms. The profiler tables and Chrome traces are in [results/pilot](results/pilot).

## Run the food diary model

From this directory, with PyTorch, torchvision and Pillow installed, run the following command using the selected distilled checkpoint.

```bash
python suggest.py path/to/photo.jpg
```

It prints the three dish names with the highest softmax scores. The scores refer to the 101 Food-101 classes.

## Reproduce the experiment

The repository includes the two trained ResNet-18 checkpoints, class labels, benchmark manifests and results. Training requires the full Food-101 dataset and a PyTorch environment with torchvision and Pillow. The reported training runs used an RTX 4070 Ti SUPER.

Run the following in Bash from the repository root on the training host. The official archive's SHA-256 is `d97d15e438b7f4498f96086a4f7e2fa42a32f2712e87d3295441b2b6314053a4`. The commands fetch the same teacher revision used here.

```bash
mkdir -p assets
source_url=https://huggingface.co/ilian-hadzhidimitrov/food-classifier-resnet50/resolve/d945125805c9748ce6a5d49e2bb2fb8fd738c2eb
curl -fL "$source_url/resnet50_food101_best.pt" -o assets/resnet50_food101_best.pt
curl -fL "$source_url/classes_food101.txt" -o assets/classes_food101.txt
python -c "from torchvision.datasets import Food101; Food101('data', download=True)"
python train.py --role init
python evaluate.py --teacher assets/resnet50_food101_best.pt --output results/teacher-test.json
```

For the CPU pilot, copy the teacher, labels and `checkpoints/student_init.pt` to the laptop and make the official Food-101 images available there. Run the first command below on the laptop and the second on the training host. Then copy `checkpoints/plain.pt` and `checkpoints/distilled.pt` back to the laptop for the final benchmark. Both CPU runs select the same 505 photos by ID and verify the JPEG hashes.

```bash
python benchmark.py pilot --images data/food-101/images
python train.py --role all
python benchmark.py final --images data/food-101/images
```

Copy the GPU-generated `results` files to the laptop alongside its benchmarks. Run `python compare.py` to join accuracy and latency by checkpoint hash and recount the gains and losses from [per-photo predictions](results/student_test_predictions.csv).

To rebuild the report on Windows with Segoe UI and Georgia installed, run `python -m pip install reportlab pypdf pillow` and then `python report/build_report.py` from the repository root. The build uses the checked-in figures and writes the same PDF to `HW1_Report_Nazarov.pdf` and `output/pdf/HW1_Food101_v5.pdf`.
