# eval_viz.py -- matplotlib comparison of the random vs trained evaluation runs.
# Called from evaluate.py: plot_eval(r, t, it, n=N)
#
# r, t are the tuples run_episodes() returns:
#   (raw_rets, shaped_rets, lens, lines, hist, terminated)
# one entry per game, except hist which is the 12-bin action count over all games.

import numpy as np
import matplotlib.pyplot as plt

_ACTION_LABELS = ['NOOP', 'A', 'B', 'right', 'right+A', 'right+B',
                  'left', 'left+A', 'left+B', 'down', 'down+A', 'down+B']
_C_RAND  = '#8899a6'   # grey-blue  = random baseline
_C_TRAIN = '#e0662b'   # orange     = trained policy


def _strip_box(ax, series_r, series_t, ylabel, title):
    """Box plot + jittered per-game dots for random (left) vs trained (right)."""
    bp = ax.boxplot([series_r, series_t], widths=0.55, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='black',
                                   markeredgecolor='black', markersize=5),
                    medianprops=dict(color='black'), patch_artist=True)
    for patch, c in zip(bp['boxes'], (_C_RAND, _C_TRAIN)):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    for i, s in enumerate((series_r, series_t), start=1):
        if len(s):
            jitter = np.random.normal(i, 0.05, size=len(s))
            ax.scatter(jitter, s, s=22, color=(_C_RAND, _C_TRAIN)[i - 1],
                       edgecolor='white', linewidth=0.5, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['random', 'trained'])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', alpha=0.25)


def plot_eval(r, t, iteration, n=None, step_cap=5000,
              out_path='eval_comparison.png', show=True):
    raw_r, shp_r, len_r, lin_r, hist_r, term_r = r
    raw_t, shp_t, len_t, lin_t, hist_t, term_t = t
    term_r = np.asarray(term_r, dtype=bool)
    term_t = np.asarray(term_t, dtype=bool)
    if n is None:
        n = len(len_r)

    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.5, wspace=0.25)
    ax_steps = fig.add_subplot(gs[0, 0])
    ax_lines = fig.add_subplot(gs[0, 1])
    ax_shp   = fig.add_subplot(gs[1, 0])
    ax_term  = fig.add_subplot(gs[1, 1])
    ax_hist  = fig.add_subplot(gs[2, :])

    # steps survived -- probably the strongest learning signal
    _strip_box(ax_steps, len_r, len_t, 'agent steps  (x4 = NES frames)',
               f'Steps survived    random {len_r.mean():.0f}   |   trained {len_t.mean():.0f}')

    # lines cleared per game, the metric we actually care about
    _strip_box(ax_lines, lin_r, lin_t, 'lines cleared',
               f'Lines per game    random {lin_r.mean():.2f}   |   trained {lin_t.mean():.2f}')

    # cumulative shaped reward -- kind of biased, see the caption below
    _strip_box(ax_shp, shp_r, shp_t, 'cumulative shaped reward',
               f'Shaped return    random {shp_r.mean():.1f}   |   trained {shp_t.mean():.1f}')
    ax_shp.text(0.5, -0.22,
                'longer games score lower here: sum of (Phi_now - Phi_prev) telescopes to '
                'Phi(final board),\nwhich is more negative the fuller the board is at top-out',
                transform=ax_shp.transAxes, ha='center', va='top',
                fontsize=8, style='italic', color='#666')

    # how games ended -- topped out vs hit the step cap
    cap_r, cap_t = float((~term_r).mean()), float((~term_t).mean())
    top = [1 - cap_r, 1 - cap_t]
    cap = [cap_r, cap_t]
    ax_term.bar(['random', 'trained'], top, color=(_C_RAND, _C_TRAIN), label='topped out')
    ax_term.bar(['random', 'trained'], cap, bottom=top, color='#d9d9d9',
                label=f'hit {step_cap}-step cap')
    for i, (tp, cp) in enumerate(zip(top, cap)):
        ax_term.text(i, tp / 2, f'{tp*100:.0f}%', ha='center', va='center', fontsize=9)
        if cp > 0.02:
            ax_term.text(i, tp + cp / 2, f'{cp*100:.0f}%', ha='center', va='center', fontsize=9)
    ax_term.set_ylim(0, 1)
    ax_term.set_ylabel('fraction of games')
    ax_term.set_title('How games ended', fontsize=10)
    ax_term.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))

    # action distribution -- did it collapse? what is it actually pressing
    fr = hist_r / max(hist_r.sum(), 1)
    ft = hist_t / max(hist_t.sum(), 1)
    x = np.arange(len(_ACTION_LABELS))
    w = 0.4
    ax_hist.bar(x - w / 2, fr, w, label='random', color=_C_RAND)
    ax_hist.bar(x + w / 2, ft, w, label='trained', color=_C_TRAIN)
    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels(_ACTION_LABELS, rotation=45, ha='right')
    ax_hist.set_ylabel('fraction of actions chosen')
    ax_hist.set_title('Action distribution', fontsize=10)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(axis='y', alpha=0.25)

    fig.suptitle(f'Trained vs Random   --   checkpoint iteration {iteration},   '
                 f'N = {n} games each', fontsize=13, y=0.997)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'saved {out_path}')
    if show:
        try:
            plt.show()
        except Exception as e:
            print(f'(could not open a window: {e})')
    plt.close(fig)
