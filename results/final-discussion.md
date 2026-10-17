### Measured comparison and error analysis

| Dataset/model | Parameters | Best validation accuracy | Official test accuracy | Macro F1 | Training time | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|
| MNIST / DigitNetPlus | 435,754 | 99.70% | 99.65% | 99.65% | 24.67 min | 45 / 60 |
| CIFAR-10 / VisionNetWide | 5,849,050 | 94.12% | 93.32% | 93.33% | 7 h 27 min | 148 / 200 |

The MNIST model made 35 mistakes among 10,000 test images. Its largest directed confusion was 9→4 (5 images), followed by 5→3 and 4→9 (4 each). These pairs have similar strokes in ambiguous handwriting, but that explanation is a hypothesis supported by inspecting the displayed examples rather than a causal finding.

The CIFAR-10 model made 668 mistakes. Its most frequent directed confusions were dog→cat (82), cat→dog (64), truck→automobile (44), bird→cat (35), and airplane→ship (29). At 32×32 resolution, related animal shapes and related vehicle silhouettes often share much of the available visual evidence. The saved mistake panel and confusion matrix show the actual outcomes.

MNIST and CIFAR-10 accuracy values describe different tasks and are not a controlled comparison of architecture quality. CIFAR-10 contains greater variation in color, pose, texture, background, and within-class appearance, which is consistent with its lower measured accuracy and need for a wider residual model.
