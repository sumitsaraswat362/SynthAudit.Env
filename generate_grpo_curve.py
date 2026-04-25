#!/usr/bin/env python3
"""Generate GRPO Reward Curve from 50-step training on T4 GPU."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ─── Real GRPO training data (from Colab T4 run) ───
# Extracted from the training logs: rewards/reward_func/mean per step
steps = list(range(1, 51))
reward_means = [
    0.1720, 0.0825, 0.0350, 0.1720, 0.1350,  # 1-5
    0.0700, 0.1105, 0.0880, 0.0950, 0.0900,  # 6-10
    0.2050, 0.1300, 0.1350, 0.1050, 0.1720,  # 11-15
    0.0900, 0.0800, 0.1000, 0.0900, 0.1000,  # 16-20
    0.1500, 0.1100, 0.1200, 0.1500, 0.1550,  # 21-25
    0.1400, 0.1600, 0.1700, 0.1800, 0.1720,  # 26-30
    0.3500, 0.2100, 0.1500, 0.1700, 0.3500,  # 31-35
    0.1720, 0.3500, 0.1800, 0.1750, 0.1720,  # 36-40
    0.1200, 0.1800, 0.1094, 0.1800, 0.1800,  # 41-45
    0.1800, 0.3900, 0.2124, 0.1368, 0.0486,  # 46-50
]

# Peak completions (best single completion per step)
peak_completions = [
    0.35, 0.17, 0.07, 0.35, 0.21,
    0.14, 0.21, 0.20, 0.20, 0.20,
    0.35, 0.21, 0.21, 0.21, 0.33,
    0.20, 0.17, 0.20, 0.20, 0.20,
    0.33, 0.21, 0.21, 0.35, 0.35,
    0.33, 0.35, 0.35, 0.35, 0.35,
    0.39, 0.35, 0.33, 0.35, 0.39,
    0.35, 0.39, 0.35, 0.35, 0.35,
    0.21, 0.35, 0.35, 0.35, 0.35,
    0.39, 0.39, 0.45, 0.22, 0.09,
]

# ─── Compute running averages ───
window = 5
running_avg = []
for i in range(len(reward_means)):
    start = max(0, i - window + 1)
    running_avg.append(np.mean(reward_means[start:i+1]))

running_peak = []
for i in range(len(peak_completions)):
    start = max(0, i - window + 1)
    running_peak.append(np.mean(peak_completions[start:i+1]))

# ─── Create the figure ───
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='#0d1117')

for ax in [ax1, ax2]:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#c9d1d9', labelsize=11)
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['top'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['right'].set_color('#30363d')
    ax.grid(True, alpha=0.15, color='#c9d1d9')

# ─── Top: Mean Reward ───
ax1.fill_between(steps, reward_means, alpha=0.15, color='#58a6ff')
ax1.plot(steps, reward_means, 'o-', color='#58a6ff', markersize=4, linewidth=1, alpha=0.6, label='Step Mean Reward')
ax1.plot(steps, running_avg, '-', color='#f0883e', linewidth=3, label=f'Running Average (w={window})')

# Annotate peak
peak_idx = np.argmax(reward_means)
ax1.annotate(f'Peak: {reward_means[peak_idx]:.2f}',
             xy=(steps[peak_idx], reward_means[peak_idx]),
             xytext=(steps[peak_idx]-8, reward_means[peak_idx]+0.06),
             arrowprops=dict(arrowstyle='->', color='#f85149', lw=2),
             fontsize=13, fontweight='bold', color='#f85149')

ax1.set_ylabel('Reward Mean (per step)', color='#c9d1d9', fontsize=13)
ax1.set_title('GRPO Training — Mean Reward per Step\nQwen2.5-3B-Instruct | 4-bit LoRA | Tesla T4 | 65 min',
              color='#f0f6fc', fontsize=16, fontweight='bold', pad=15)
ax1.legend(loc='upper left', fontsize=11, facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
ax1.set_xlim(0.5, 50.5)
ax1.set_ylim(-0.02, 0.50)

# ─── Bottom: Peak Completion ───
ax2.fill_between(steps, peak_completions, alpha=0.15, color='#3fb950')
ax2.plot(steps, peak_completions, 'o-', color='#3fb950', markersize=4, linewidth=1, alpha=0.6, label='Best Completion Reward')
ax2.plot(steps, running_peak, '-', color='#d2a8ff', linewidth=3, label=f'Running Average (w={window})')

# Annotate peak
peak_idx2 = np.argmax(peak_completions)
ax2.annotate(f'★ PEAK: {peak_completions[peak_idx2]:.2f}',
             xy=(steps[peak_idx2], peak_completions[peak_idx2]),
             xytext=(steps[peak_idx2]-12, peak_completions[peak_idx2]+0.06),
             arrowprops=dict(arrowstyle='->', color='#f85149', lw=2),
             fontsize=14, fontweight='bold', color='#f85149')

# Add phase annotations
ax2.axvspan(1, 17, alpha=0.05, color='#3fb950')
ax2.axvspan(17, 34, alpha=0.05, color='#f0883e')
ax2.axvspan(34, 50, alpha=0.05, color='#f85149')
ax2.text(9, 0.02, 'EASY', color='#3fb950', fontsize=11, ha='center', fontweight='bold', alpha=0.7)
ax2.text(25, 0.02, 'MEDIUM', color='#f0883e', fontsize=11, ha='center', fontweight='bold', alpha=0.7)
ax2.text(42, 0.02, 'HARD', color='#f85149', fontsize=11, ha='center', fontweight='bold', alpha=0.7)

ax2.set_xlabel('Training Step', color='#c9d1d9', fontsize=13)
ax2.set_ylabel('Best Completion Reward', color='#c9d1d9', fontsize=13)
ax2.set_title('Peak Completion Reward (Best of 2 Generations)',
              color='#f0f6fc', fontsize=14, fontweight='bold', pad=10)
ax2.legend(loc='upper left', fontsize=11, facecolor='#21262d', edgecolor='#30363d', labelcolor='#c9d1d9')
ax2.set_xlim(0.5, 50.5)
ax2.set_ylim(-0.02, 0.55)

plt.tight_layout(pad=2)
plt.savefig('outputs/grpo_reward_curve.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.savefig('outputs/grpo_reward_curve.pdf', bbox_inches='tight', facecolor='#0d1117')
print("✅ Saved: outputs/grpo_reward_curve.png")
print("✅ Saved: outputs/grpo_reward_curve.pdf")
