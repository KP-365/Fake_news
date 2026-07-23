from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

repo_root = Path(__file__).resolve().parents[2]
arrays_dir = repo_root / 'evaluation' / 'mc_arrays'
output_path = Path(__file__).with_name('rejection_curve.pdf')
png_output_path = Path(__file__).with_name('rejection_curve.png')

mean_probabilities = np.load(arrays_dir / 'mean_probabilities.npy')
true_labels = np.load(arrays_dir / 'true_labels.npy')
predictive_entropy = np.load(arrays_dir / 'predictive_entropy.npy')

predicted_labels = mean_probabilities.argmax(axis=1)
correct = predicted_labels == true_labels
full_coverage_accuracy = correct.mean()

# Retain the most-certain articles first, matching the executed evaluation notebook.
order = np.argsort(predictive_entropy)
sorted_correct = correct[order]
retained_count = np.arange(1, len(order) + 1)
coverage = retained_count / len(order)
retained_accuracy = np.cumsum(sorted_correct) / retained_count

operating_retained_count = len(order) - 100
operating_index = operating_retained_count - 1
operating_coverage = coverage[operating_index]
operating_accuracy = retained_accuracy[operating_index]

assert operating_retained_count == 9_298
assert f'{operating_coverage:.2%}' == '98.94%'
assert f'{operating_accuracy:.2%}' == '99.88%'

fig, ax = plt.subplots(figsize=(5.4, 4.2))
ax.plot(
    coverage,
    retained_accuracy,
    color='#3572A5',
    linewidth=1.6,
    label='Retained accuracy',
)
ax.axhline(
    full_coverage_accuracy,
    color='#666666',
    linestyle='--',
    linewidth=1.2,
    label=f'Full-coverage MC accuracy ({full_coverage_accuracy:.2%})',
)
ax.scatter(
    operating_coverage,
    operating_accuracy,
    color='#C44E52',
    edgecolor='white',
    linewidth=0.8,
    s=48,
    zorder=3,
)
ax.annotate(
    '100 deferred\n98.94% coverage\n99.88% accuracy',
    (operating_coverage, operating_accuracy),
    xytext=(-10, 12),
    textcoords='offset points',
    ha='right',
    va='bottom',
    fontsize=8,
    bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'edgecolor': '#999999'},
)
ax.set_xlabel('Coverage (share of articles auto-classified)')
ax.set_ylabel('Accuracy on retained articles')
ax.set_title('Rejection curve (reject most-uncertain first)')
ax.set_xlim(0.0, 1.0)
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
ax.grid(color='#D9D9D9', linewidth=0.6)
ax.legend(loc='lower left', frameon=False)

fig.tight_layout()
fig.savefig(output_path, format='pdf', bbox_inches='tight')
fig.savefig(png_output_path, format='png', dpi=300, bbox_inches='tight')
plt.close(fig)

print(f'full-coverage mc accuracy = {full_coverage_accuracy:.2%}')
print(
    f'operating point = {operating_retained_count:,}/{len(order):,} retained, '
    f'{operating_coverage:.2%} coverage, {operating_accuracy:.2%} accuracy'
)
print(f'pdf figure = {output_path}')
print(f'png figure = {png_output_path}')
