from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

repo_root = Path(__file__).resolve().parents[2]
arrays_dir = repo_root / 'evaluation' / 'mc_arrays'
output_path = Path(__file__).with_name('predictive_entropy_boxplot.pdf')

mean_probabilities = np.load(arrays_dir / 'mean_probabilities.npy')
true_labels = np.load(arrays_dir / 'true_labels.npy')
predictive_entropy = np.load(arrays_dir / 'predictive_entropy.npy')

predicted_labels = mean_probabilities.argmax(axis=1)
correct = predicted_labels == true_labels
correct_mean = predictive_entropy[correct].mean()
incorrect_mean = predictive_entropy[~correct].mean()

assert f'{correct_mean:.4f}' == '0.0333'
assert f'{incorrect_mean:.4f}' == '0.4816'

# The notebook used 15 equal-width bins, so this repeats its ECE calculation exactly.
confidence = mean_probabilities.max(axis=1)
bin_edges = np.linspace(0.0, 1.0, 16)
mc_ece = 0.0
for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
    in_bin = (confidence > lower) & (confidence <= upper)
    if in_bin.any():
        bin_accuracy = correct[in_bin].mean()
        mc_ece += in_bin.mean() * abs(bin_accuracy - confidence[in_bin].mean())
assert f'{mc_ece:.4f}' == '0.0055'

fig, ax = plt.subplots(figsize=(5.4, 4.2))
colours = ['#3572A5', '#C44E52']
# The executed notebook hid outliers, which keeps the small incorrect group readable.
boxes = ax.boxplot(
    [predictive_entropy[correct], predictive_entropy[~correct]],
    tick_labels=['Correct', 'Incorrect'],
    showfliers=False,
    showmeans=True,
    patch_artist=True,
    medianprops={'color': 'black', 'linewidth': 1.2},
    meanprops={
        'marker': 'D',
        'markerfacecolor': 'white',
        'markeredgecolor': 'black',
        'markersize': 5,
    },
)
for box, colour in zip(boxes['boxes'], colours):
    box.set_facecolor(colour)
    box.set_alpha(0.78)

legend_items = [
    Patch(facecolor=colours[0], alpha=0.78, label=f'Correct mean = {correct_mean:.4f}'),
    Patch(facecolor=colours[1], alpha=0.78, label=f'Incorrect mean = {incorrect_mean:.4f}'),
]
ax.set_xlabel('Prediction outcome')
ax.set_ylabel('Predictive entropy (nats)')
ax.set_title('Predictive entropy by classification outcome')
ax.grid(axis='y', color='#D9D9D9', linewidth=0.6)
ax.legend(handles=legend_items, loc='upper left', frameon=False)

fig.tight_layout()
fig.savefig(output_path, format='pdf', bbox_inches='tight')
plt.close(fig)

print(f'correct entropy mean = {correct_mean:.4f}')
print(f'incorrect entropy mean = {incorrect_mean:.4f}')
print(f'mc ece (15 bins) = {mc_ece:.4f}')
print(f'figure = {output_path}')
