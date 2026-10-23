"""Setup and rendering, kept outside the presentation trace.

Convolution is recomputed only when the original selected checkpoint is available.
The portable fallback explicitly replays captured states; no weights are invented.
"""
from pathlib import Path
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
CLASSES = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']


class InspectableArray(np.ndarray):
    """Keep the actual array for computation; show a compact, truthful variable panel."""
    def __lectrace__(self):
        raw = np.asarray(self)
        result = {'shape': list(raw.shape), 'dtype': str(raw.dtype),
                  'min': float(raw.min()), 'max': float(raw.max())}
        if raw.size <= 512:
            result['values'] = raw.tolist()
        else:
            result['first_12_values'] = raw.reshape(-1)[:12].tolist()
            result['display_note'] = 'Compact preview; computations use the full array.'
        return result


def load_demo():
    with np.load(ASSETS / 'forward_states.npz', allow_pickle=False) as source:
        demo = {k: source[k].copy() for k in source.files}
    demo['classifier_weight'] = None
    demo['classifier_bias'] = None
    demo['mode'] = ('Recorded-state replay from the actual C/seed42 Colab forward pass; '
                    'pooling, softmax and argmax are recomputed. Convolution and logits are replayed.')
    explicit = os.environ.get('WBC_CHECKPOINT', '').strip()
    candidates = [Path(explicit)] if explicit else [
        ROOT / 'best.pt',
        Path('/content/drive/MyDrive/MD6329_WBC_ResNet18_v2_dedup/C_layer4_noaug_seed42/best.pt'),
    ]
    checkpoint = next((p for p in candidates if p.is_file()), None)
    if explicit and checkpoint is None:
        raise FileNotFoundError(f'WBC_CHECKPOINT does not exist: {explicit}')
    if checkpoint is not None:
        import torch
        from torch import nn
        from torchvision.models import resnet18
        torch.set_num_threads(2)
        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 4)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        captured = {}
        h1 = model.layer4.register_forward_hook(lambda m, i, o: captured.update(conv_features=o.detach().numpy().copy()))
        h2 = model.avgpool.register_forward_hook(lambda m, i, o: captured.update(pooled_features=o.detach().numpy().copy()))
        try:
            with torch.no_grad():
                scores = model(torch.from_numpy(demo['normalized_input']).float())
                probs = scores.softmax(1).numpy()
        finally:
            h1.remove()
            h2.remove()
        if not np.allclose(probs, demo['probabilities'], atol=1e-3):
            raise ValueError('Checkpoint does not reproduce the recorded example. Use C, seed42 best.pt.')
        demo.update(captured)
        demo['logits'] = scores.numpy()
        demo['probabilities'] = probs
        demo['classifier_weight'] = model.fc.weight.detach().numpy().copy()
        demo['classifier_bias'] = model.fc.bias.detach().numpy().copy()
        demo['mode'] = ('Live checkpoint inference on CPU: the original C/seed42 weights '
                        'recompute the convolutional and pooling states. No training or pretrained-weight download.')
    return {k: v.view(InspectableArray) if isinstance(v, np.ndarray) else v
            for k, v in demo.items()}


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def make_figures(demo):
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.imshow(demo['original_image']); ax.axis('off')
    ax.set_title('Fixed test example: eosinophil', fontsize=11)
    ax.text(0.5, -0.06, 'Original resolution: 64×64', transform=ax.transAxes,
            ha='center', fontsize=10)
    finish(fig, 'state_1_input.png')
    features = demo['conv_features'][0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    a = axes[0].imshow(features.mean(0), cmap='viridis')
    axes[0].set_title('Mean of 512 feature maps (not Grad-CAM)')
    fig.colorbar(a, ax=axes[0], shrink=.8)
    axes[1].imshow(features[0], cmap='viridis'); axes[1].set_title('Channel 0: 7 x 7 activations')
    finish(fig, 'state_2_features.png')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].imshow(features[0], cmap='viridis'); axes[0].set_title(f'Channel 0 mean = {features[0].mean():.4f}')
    axes[1].plot(features.mean(axis=(1, 2)), linewidth=.8)
    axes[1].set(xlabel='Learned feature index', ylabel='Mean activation', title='512 pooled features')
    finish(fig, 'state_3_pooling.png')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(CLASSES, demo['logits'][0], color='slateblue'); axes[0].set_title('Four unnormalized class scores')
    axes[1].bar(CLASSES, demo['probabilities'][0], color='steelblue'); axes[1].set(ylim=(0, 1.08), title='Softmax outputs')
    for ax in axes: ax.tick_params(axis='x', rotation=35)
    finish(fig, 'state_4_decision.png')
    # Original figure: eight rows, first four errors then four correct cases.
    original = Image.open(ASSETS / 'error_analysis_gradcam.png')
    original.crop((0, 0, original.width, round(original.height * 2 / 8))).save(ASSETS / 'error_cases.png')


def results_text():
    with (ASSETS / 'test_results_per_seed.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    lines = ['| Method | Test accuracy (%) | Test macro-F1 (%) |',
             '|---|---:|---:|']
    values = {}
    for name in ['A_frozen_aug', 'B_layer4_aug', 'C_layer4_noaug']:
        selected = [r for r in rows if r['experiment'] == name]
        if len(selected) != 2:
            raise ValueError('Expected exactly the two predefined seeds per method.')
        acc = 100 * np.array([float(r['accuracy']) for r in selected])
        f1 = 100 * np.array([float(r['macro_f1']) for r in selected])
        values[name] = f1.mean()
        lines.append(f'| {name} | {acc.mean():.2f} ± {acc.std(ddof=1):.2f} | {f1.mean():.2f} ± {f1.std(ddof=1):.2f} |')
    evidence = (f'With augmentation held fixed, B improves mean test macro-F1 over A by '
                f'**{values["B_layer4_aug"] - values["A_frozen_aug"]:.2f} percentage points**. '
                f'C exceeds B by only **{values["C_layer4_noaug"] - values["B_layer4_aug"]:.2f} percentage points**. '
                'Two seeds do not establish statistical significance. B has the highest mean AUROC, '
                'so C should not be described as best on every metric.')
    # Render the same table with explicit column widths for the presentation.
    cells = [[part.strip() for part in line.strip('|').split('|')]
             for line in lines[2:]]
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis('off')
    table = ax.table(cellText=cells,
                     colLabels=['Method', 'Test accuracy (%)', 'Test macro-F1 (%)'],
                     colWidths=[0.40, 0.30, 0.30], cellLoc='center',
                     loc='center', bbox=[0, 0.04, 1, 0.92])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    for (row, col), cell in table.get_celld().items():
        cell.PAD = 0.12
        cell.set_edgecolor('#cbd5e1')
        if row == 0:
            cell.set_facecolor('#e2e8f0')
            cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#f8fafc' if row % 2 else 'white')
        if col == 0:
            cell.set_text_props(ha='left')
    finish(fig, 'results_table.png')
    return '\n'.join(lines), evidence


REFERENCES = '''## Sources and course connections
- Acevedo et al. (2020). A dataset of microscopic peripheral blood cell images for development of automatic recognition systems. https://doi.org/10.1016/j.dib.2020.105474
- Yang et al. (2023). MedMNIST v2. https://doi.org/10.1038/s41597-022-01721-8
- He et al. (2016). Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
- Selvaraju et al. (2017). Grad-CAM. https://arxiv.org/abs/1610.02391
- MedMNIST data/code: https://github.com/MedMNIST/MedMNIST
- torchvision ResNet18 implementation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- LecTrace implementation: https://pypi.org/project/lectrace/1.1.30/
- Course: Lecture 2, PDF pp.107–117 and 123–124; Lecture 11, PDF pp.30–33.

This presentation uses actual exported experiment results.
'''

DEMO = load_demo()
make_figures(DEMO)
RESULT_TABLE, EVIDENCE = results_text()
