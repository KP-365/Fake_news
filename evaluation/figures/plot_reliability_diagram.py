from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[2]
arrays_dir = repo_root / 'evaluation' / 'mc_arrays'
output_path = Path(__file__).with_name('mc_reliability_diagram.pdf')

mean_probabilities = np.load(arrays_dir / 'mean_probabilities.npy')
true_labels = np.load(arrays_dir / 'true_labels.npy')

predicted_labels = mean_probabilities.argmax(axis=1)
confidence = mean_probabilities.max(axis=1)
correct = predicted_labels == true_labels

# These are the same 15 equal-width bins used by the executed evaluation notebook.
bin_edges = np.linspace(0.0, 1.0, 16)
bin_confidence = []
bin_accuracy = []
mc_ece = 0.0
for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
    in_bin = (confidence > lower) & (confidence <= upper)
    if in_bin.any():
        mean_confidence = confidence[in_bin].mean()
        accuracy = correct[in_bin].mean()
        bin_confidence.append(mean_confidence)
        bin_accuracy.append(accuracy)
        mc_ece += in_bin.mean() * abs(accuracy - mean_confidence)

assert f'{mc_ece:.4f}' == '0.0055'

fig, ax = plt.subplots(figsize=(5.4, 4.5))
ax.plot(
    [0.0, 1.0],
    [0.0, 1.0],
    color='#666666',
    linestyle='--',
    linewidth=1.2,
    label='Perfect calibration',
)
ax.plot(
    bin_confidence,
    bin_accuracy,
    color='#3572A5',
    marker='o',
    markersize=5,
    linewidth=1.6,
    label='MC Dropout',
)
ax.text(
    0.04,
    0.94,
    f'ECE = {mc_ece:.4f}',
    transform=ax.transAxes,
    ha='left',
    va='top',
    bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': '#999999'},
)
ax.set_xlabel('Mean confidence per bin')
ax.set_ylabel('Accuracy per bin')
ax.set_title('MC Dropout reliability diagram (15 bins)')
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_aspect('equal', adjustable='box')
ax.grid(color='#D9D9D9', linewidth=0.6)
ax.legend(loc='lower right', frameon=False)

fig.tight_layout()
fig.savefig(output_path, format='pdf', bbox_inches='tight')
plt.close(fig)

print(f'mc ece (15 bins) = {mc_ece:.4f}')
print(f'non-empty bins = {len(bin_confidence)}')
print(f'figure = {output_path}')
