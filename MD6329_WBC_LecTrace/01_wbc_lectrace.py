"""MD6329: four selected execution states, approximately ten minutes.

Run from the extracted package: lectrace serve 01_wbc_lectrace.py
Read README_FIRST.md for Colab instructions and the two execution modes.
"""
import numpy as np
from lectrace import text, note, image
from _wbc_support import DEMO, ASSETS, CLASSES, EVIDENCE, REFERENCES


def main():
    text("# How a ResNet18 classifies a blood cell")
    text("**MD6329 · Four-class white blood cell classification**\n\n"
         "Research question: Does final-block fine-tuning improve performance over a "
         "frozen backbone, and does light augmentation add a benefit?")
    note("Classifying white blood cells can support microscopic assessment. "
         "This project evaluates cell classification, not disease diagnosis.")
    image(str(ASSETS / "training_examples.png"), width=560)
    text("## Data and controlled comparison")
    text("Barcelona PBC-derived BloodMNIST+ **four-class subset**. "
         "64×64 RGB images, resized to 224×224. After exact-image deduplication: "
         "**6,349 train / 908 validation / 1,814 test**. Nine rows were removed.\n\n"
         "**A:** frozen backbone + augmentation. **B:** train layer4 and classifier + augmentation. "
         "**C:** train layer4 and classifier, no random augmentation.\n\n"
         "All runs start independently from the same ImageNet initialization and matched "
         "classifier initialization within each seed. Seeds: 42 and 2026. "
         "AdamW, up to 8 epochs, early-stopping patience 3; classifier LR 0.001, layer4 LR 0.0001.")
    note("A versus B tests the fine-tuning strategy under matching augmentation. "
         "B versus C tests augmentation under matching trainable layers. "
         "Frozen modules keep BatchNorm running statistics fixed; layer4 BatchNorm updates when unfrozen. "
         "Augmentation is random horizontal/vertical flips and rotation within ±15°. "
         "Best checkpoints and the method are selected using validation macro-F1, not test results.")
    text(f"**Demonstration mode:** {DEMO['mode']}")
    text("Fixed demonstration case: test-subset index 0, condition C, seed 42.")

    show_input()
    feature_maps = show_convolution()
    pooled = show_pooling(feature_maps)
    show_decision(pooled)

    text("## What did the experiments show?")
    image(str(ASSETS / "results_table.png"), width=800)
    text("The method selected by mean **validation** macro-F1 was C. "
         "Test results are means ± sample SD across two training seeds; SD is not a confidence interval.")
    image(str(ASSETS / "learning_curves.png"), width=1000)
    text(EVIDENCE)
    note("Under this training budget, fine-tuning improves macro-F1 substantially. "
         "The augmentation difference is small. Some best checkpoints occur at epoch 8, "
         "so we have not established fully converged performance or a universal law about augmentation.")
    text("## Examine a failure, not just a successful prediction")
    image(str(ASSETS / "error_cases.png"), width=650)
    text("These are the first two errors from the fixed-index ordering for C, seed 42. "
         "Both are labeled neutrophil and predicted monocyte. "
         "Grad-CAM targets the **predicted class**; each image has its own heatmap scale. "
         "Warm colors indicate relatively higher positive Grad-CAM attribution, not a diagnosis.")
    note("The cell image and overlay provide complementary views. A visually plausible heatmap "
         "does not establish causal or medically correct reasoning. "
         "These two examples are illustrative, not an estimate of error frequencies.")
    text("## Take-home message and limits")
    text("**Fine-tuning improved this four-class benchmark under the tested budget; "
         "light augmentation did not show an additional benefit.**\n\n"
         "Limits: one data source, low-resolution images, only four cell types and two seeds, "
         "no patient identifiers, no near-duplicate assessment, no external validation, "
         "and no probability-calibration experiment. Exact deduplication does not establish patient independence.")
    note("Course connections: Lecture 2, pp.107–117 and 123–124 "
         "(transfer learning and independence), and Lecture 11, pp.30–33 (Grad-CAM). "
         "AI assistance was used for drafting and presentation preparation.")
    text(REFERENCES)


def show_input():
    text("## State 1 · From pixels to model input")
    original_shape = DEMO['original_image'].shape
    normalized_input = DEMO['normalized_input']  # @inspect normalized_input
    input_shape = normalized_input.shape  # @inspect input_shape
    text(f"Original image: **{original_shape}**. Model input: **{input_shape}**. "
         "Axes are batch, color channels, height, width. "
         "The recorded input was resized and normalized with ImageNet channel means and standard deviations.")
    image(str(ASSETS / "state_1_input.png"), width=300)
    note("RGB pixel values become a normalized floating-point tensor. "
         "The pretrained model expects this scale. "
         "Negative normalized values are normal; resizing does not restore missing detail.")


def show_convolution():
    text("## State 2 · Learned spatial features")
    feature_maps = DEMO['conv_features']  # @inspect feature_maps
    feature_shape = feature_maps.shape  # @inspect feature_shape
    mean_feature_map = feature_maps[0].mean(axis=0)  # @inspect mean_feature_map
    text(f"Final residual block output: **{feature_shape}** = 512 feature maps of size 7×7. "
         "Each map responds to a learned pattern. "
         "The displayed average summarizes activation across channels; it is **not Grad-CAM** "
         "and is not a class-specific explanation.")
    image(str(ASSETS / "state_2_features.png"), width=650)
    note("The network represents the image using learned channels and "
         "coarser spatial positions. These features are the representation given "
         "to pooling and the classifier. Individual channels are not verified clinical concepts.")
    return feature_maps


def show_pooling(feature_maps):
    text("## State 3 · Compress each spatial feature map")
    pooled = feature_maps.mean(axis=(2, 3))  # @inspect pooled
    first_channel = feature_maps[0, 0]  # @inspect first_channel
    first_channel_mean = float(first_channel.mean())  # @inspect first_channel_mean
    saved_pooled = DEMO['pooled_features'].reshape(1, 512)
    matches_saved_pool = bool(np.allclose(pooled, saved_pooled, atol=1e-5))
    assert matches_saved_pool, "Recomputed global average pooling disagrees with the captured state."
    text("Global average pooling computes one average for each 7×7 channel: "
         "**[1,512,7,7] → [1,512]** after flattening. "
         "The displayed code actually recalculates pooling and checks the captured output.")
    image(str(ASSETS / "state_3_pooling.png"), width=750)
    note("For the first channel, 49 spatial values become one number. "
         "Repeating this for all channels creates 512 inputs for the classifier. "
         "Feature indices are learned dimensions, not named clinical variables.")
    return pooled


def show_decision(pooled):
    text("## State 4 · Scores become a prediction")
    if DEMO['classifier_weight'] is not None:
        logits = pooled @ DEMO['classifier_weight'].T + DEMO['classifier_bias']  # @inspect logits
    else:
        logits = DEMO['logits']  # @inspect logits
    shifted_logits = logits - logits.max(axis=1, keepdims=True)  # @inspect shifted_logits
    exponentials = np.exp(shifted_logits)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)  # @inspect probabilities
    predicted_index = int(probabilities[0].argmax())  # @inspect predicted_index
    predicted_class = CLASSES[predicted_index]  # @inspect predicted_class
    true_class = CLASSES[int(DEMO['true_label'][0])]  # @inspect true_class
    assert np.allclose(probabilities, DEMO['probabilities'], atol=1e-5), "Softmax verification failed."
    text(f"Predicted: **{predicted_class}**. Reference label: **{true_class}**. "
         f"Maximum softmax output: **{100 * probabilities.max():.4f}%**. "
         "This is a model output, not a calibrated clinical probability.")
    image(str(ASSETS / "state_4_decision.png"), width=750)
    note("In live-checkpoint mode the matrix multiplication computes four logits "
         "from the 512 pooled features and the trained classifier. In recorded mode the logits "
         "come from the captured forward pass. Softmax and argmax are computed here in both modes. "
         "Subtracting the maximum improves numerical stability without changing softmax. "
         "A confident correct case is useful for tracing, but says little about reliability on its own.")


if __name__ == '__main__':
    main()
