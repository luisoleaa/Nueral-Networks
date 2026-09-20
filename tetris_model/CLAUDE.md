# tetris_model

A from-scratch (pure NumPy, no ML framework) attempt to get an agent to play
NES Tetris via `gym_tetris`/`nes_py`. Started as "train a neural net with
PPO," ended up as "a hand-coded search bot works, and evolving its weights
with a GA beats hand-tuning it." Read this before assuming any particular
approach is still the active one -- several were tried and abandoned.

## TL;DR: what actually works right now

`heuristic_bot.py` -- a pure-NumPy exhaustive search bot. For each falling
piece it simulates every legal (rotation, column) placement, scores each
resulting board, and executes the best one against the real emulator. It
reliably clears 70-150+ lines/game. This is the only thing in the whole
project that plays Tetris well.

Its scoring weights were hand-tuned at first (`[20.0, -0.51, -0.36,
-0.18]` for `[w_lines, w_agg_height, w_holes, w_bumpiness]`), then
`evolve_bot.py` (a genetic algorithm) found a different set of weights
(`[15.16, -1.06, -0.29, -0.07]`) that beat the hand-tuned ones live: 121.7
vs 109.4 mean lines/game over 10 episodes each. See "GA weight evolution"
below for status and what's unfinished there.

**Every neural-network approach (PPO from scratch, BC, BC+PPO fine-tune,
BC+DAgger across 56 rounds) failed to clear a single line live.** This is
a load-bearing fact for this project -- see "why the NN never worked"
below before spending more time trying to fine-tune it further. The
current, uncontested working strategy is the search bot (optionally with
evolved weights), not a learned policy.

## File map

| File | What it does |
|---|---|
| `grid.py` | Board/state feature extraction: `binary_board`, `column_heights`, `count_holes`, `board_features`/`board_potential` (evolvable-weight version), `placement_score`, `row_progress_score`, `get_state_vector` (262-dim NN input, only used by the NN scripts). |
| `heuristic_bot.py` | **The thing that works.** Exhaustive placement search + execution against the real emulator. `search_best_placement(env, weights=None)` and `run_bot_episode(...)` both take an optional `weights` override (defaults to `grid.DEFAULT_PLACEMENT_WEIGHTS`), used by `evolve_bot.py`. |
| `evolve_bot.py` | Genetic algorithm that evolves `heuristic_bot`'s placement-scoring weights. Offline simulator (no emulator, no crash risk) for fitness, live `validate_live()` at the end to sanity-check against the real game. Resumable via `evolve_ckpt.npz`. |
| `model.py` | Hand-rolled `Layer` (forward/backward), `Relu`, `Adam` optimizer, `softmax`, `gradient_check`. The NN building blocks -- no autodiff, everything is manually derived backprop. |
| `functions.py` | `compute_gae`, `compute_ppo_loss`, `get_minibatches`, `save_checkpoint`/`load_checkpoint` (atomic `.npz` write pattern used everywhere). |
| `main.py` | PPO training loop for the NN. **Never got the policy to clear a line**, despite reward-shaping fixes, value-function warmup, and starting from a BC-pretrained checkpoint. Not the recommended path forward -- see below. |
| `evaluate.py` / `eval_viz.py` | Loads an NN checkpoint, plays N games greedily, compares to a random baseline, plots a matplotlib comparison. Useful if you ever revisit the NN, not used by the bot/GA path. |
| `heuristic_bot.py` generates demos -> `generate_demos.py` -> `pretrain_bc.py` -> `dagger_train.py` | The abandoned BC/DAgger pipeline for bootstrapping the NN from the bot's play. All working code, all ultimately a dead end (see below). |
| `record_human_play.py` | Records your own keyboard play as demo pairs, same format as the bot's demos. Never actually used (no `human_demos.npz` was ever generated). |
| `play.py` | Tiny original script, just launches `play_human` on the env. Predates everything else here. |

Generated/derived files are gitignored (see `.gitignore`): `*.npz` (except
`evolve_ckpt.npz` and `evolved_weights.npz`, which are force-added since
they're the GA's actual useful output), `*.log`, `*.csv`, `*.png`,
`__pycache__/`.

## Why the NN never worked (don't redo this without reading this first)

Four different attempts, all ending in 0 lines cleared live:

1. **From-scratch PPO** (`main.py`) -- trial-and-error RL directly. Reward
   is too sparse/delayed (a line clear needs ~10-40 correct button presses
   in a row, repeated over many pieces) for random exploration to ever
   stumble into it. Policy tended to collapse onto 1-2 actions.
2. **BC pretraining** (`pretrain_bc.py`) -- imitation learning from the
   bot's demos, hit 75% validation accuracy, still 0 lines live (greedy or
   stochastic). Classic distributional-shift problem: BC only learns what
   the expert does on states the expert visits; any small deviation lands
   the network somewhere it's never seen, with no idea what to do.
3. **BC -> PPO fine-tune** -- tried to fix #2 with more RL. Made it worse
   (catastrophic forgetting): a freshly-initialized value head produced
   near-noise advantages early on, pushing the already-decent BC policy
   toward randomness before the critic had anything real to learn from.
   Guarded with value-function warmup / lower LR / lower entropy coef /
   row-completion reward shaping -- slowed the erosion, didn't reverse it.
4. **BC -> DAgger** (`dagger_train.py`) -- the standard fix for #2's
   specific failure mode: roll the current policy out live, query the bot
   for a corrective label at every state it actually visits, aggregate,
   retrain. Ran 56 rounds total (~350k pairs). Train accuracy capped at
   72-81% and actually drifted down as more self-generated data came in.
   Steps-survived plateaued (~650-830). **Still 0 lines, every single
   round.**

Leading hypothesis (untested): the 2-hidden-layer 128/64-unit MLP lacks
the representational capacity/structure (no convolution for board
locality, no memory across pieces) for precise multi-step placement
control. If picking this back up, the next real experiment is trying a
meaningfully bigger/different architecture on the existing aggregated
DAgger dataset, or reframing the NN's job as scoring placements (like the
bot does) instead of predicting raw button sequences.

**This is why the project pivoted to evolving the search bot's weights
instead** -- it sidesteps the whole "learn long-horizon control from
scratch" problem by keeping the proven-working exhaustive search and only
learning/evolving the few numbers that rank candidate boards.

## GA weight evolution -- current status

`evolve_bot.py` evolves `[w_lines, w_agg_height, w_holes, w_bumpiness]`
(4 genes) with tournament selection, arithmetic crossover, and per-gene
Gaussian mutation with sigma decay. Fitness is the mean lines cleared
across `GAMES_PER_EVAL` offline-simulated games (no emulator -- built from
`heuristic_bot.PIECE_SHAPES`/`simulate_drop`, piece sequences uniform
random over the 7 families). Individual 0 of generation 0 is always seeded
as the exact current hand-tuned baseline, so the GA can never report a
"best" worse than what's already in production.

Arithmetic crossover is safe here (unlike for NN weights) because each of
the 4 genes has a fixed, named meaning across every individual -- there's
no hidden-unit permutation ambiguity to worry about.

**Last run**: 20 generations, pop 40, `MAX_PIECES_SIM=400`,
`GAMES_PER_EVAL=3`. Result: evolved weights
`[15.16, -1.06, -0.29, -0.07]`, live validation 121.7 vs baseline's 109.4
mean lines/game (10 episodes each) -- about +11%, and this is a real
emulator result, not just the offline simulator's number.

**Known issue, not yet fixed**: offline fitness saturated across
generations (best stayed ~155-158 the whole run) even though live results
were clearly improving. Likely cause: `MAX_PIECES_SIM=400` caps games
before a "better" genome's advantage shows up -- once a genome is good
enough to survive 400 pieces, the simulator can't distinguish it from an
even-better one. **Next step, if resuming this**: raise `MAX_PIECES_SIM`
(e.g. 800-1000) and/or `GAMES_PER_EVAL` to un-saturate the fitness signal,
then resume from `evolve_ckpt.npz` for more generations. Per-generation
cost was ~90-105s at the current settings (pop 40, `MAX_PIECES_SIM=400`,
3 games/genome) -- benchmark a single `simulate_game()` call before
committing to a much longer run, cost scales roughly linearly with
`MAX_PIECES_SIM * GAMES_PER_EVAL * POP_SIZE`.

To resume: just rerun `python evolve_bot.py` -- it auto-detects
`evolve_ckpt.npz` and continues from the last completed generation. Raise
`N_GENERATIONS` first if you want it to keep going past where it stopped.

## Key technical gotchas (learned the hard way, don't rediscover these)

- **nes_py native crash**: `env.step()` can raise `OSError: access
  violation` (native C extension bug) after a large number of accumulated
  steps on one persistent env instance. Not recoverable in place -- every
  long-running script (`main.py`, `dagger_train.py`, `evolve_bot.py`'s
  `validate_live`) wraps `env.step()`/rollout loops in
  `try/except OSError` that closes and recreates the env, ending the
  current rollout/episode early. `heuristic_bot.py` itself has no guard
  (never needed one across its own validation runs) -- add one if you see
  this there too.
- **Edge-triggered inputs, NES DAS**: rotation (`A`) needs a genuine
  release before a repeat press registers -- no auto-repeat just from
  holding a macro-step action. Left/right follows NES DAS (shifts once
  after a short lag, then needs many more held frames before repeating).
  Every script that presses buttons re-verifies against `ram[0x40]`
  (column) / `_current_piece` (orientation) rather than assuming a press
  worked, and inserts a NOOP macro-step after each press to force a real
  release.
- **`board_potential` is bounded above by 0** (it's `-weight*feature`
  terms with a negative-weighted feature that's always >= 0). Reward
  shaping that multiplies it by `gamma` incorrectly
  (`gamma*phi_now - phi_prev` instead of `phi_now - phi_prev`) silently
  injects free positive reward for doing nothing, since
  `(gamma-1)*phi >= 0` unconditionally when `phi <= 0`. Already fixed
  everywhere it appears, but if you add new potential-based shaping,
  remember potential-based shaping's invariance theorem only holds for
  `F = gamma*Phi(s') - Phi(s)`, applied exactly that way.
- **Checkpoint convention**: atomic write (`np.savez` to
  `path + ".tmp.npz"`, then `os.replace`), no pickle ever, plain arrays
  only. `functions.py`'s `save_checkpoint`/`load_checkpoint` for NN layer
  lists, `evolve_bot.py`'s `save_genome_checkpoint`/`load_genome_checkpoint`
  and `save_best_weights`/`load_best_weights` mirror the same pattern for
  GA state. Follow this pattern for any new persisted state.
- **RAM addresses**: `ram[0x40]` = piece column, `ram[0x41]` = piece row,
  `env.unwrapped._current_piece` / `_next_piece` = orientation names (one
  of the 19 in `grid.piece_table`), `env.unwrapped._board` = raw tile IDs
  (239 = empty).
- **`PIECE_SHAPES`** in `heuristic_bot.py` were captured empirically (a
  calibration script let each orientation lock naturally and recorded its
  cells relative to the pivot `(ram[0x41], ram[0x40])`), not hand-typed --
  trust them, they're cross-checked against standard tetromino geometry.

## How to run things

```
python heuristic_bot.py       # 5-episode self-test of the search bot, prints lines/game
python evolve_bot.py          # runs (or resumes) the GA, then live-validates baseline vs evolved
python generate_demos.py      # regenerate bc_demos.npz from the bot (only needed for the NN path)
python pretrain_bc.py         # BC pretrain (only needed for the NN path)
python dagger_train.py        # DAgger training (only needed for the NN path, not recommended -- see above)
python main.py                # PPO training (not recommended -- see above)
python evaluate.py            # compare a trained NN checkpoint vs random, needs Tetris_ppo.npz
```

All of these use the Python at
`/c/Users/luiso/AppData/Local/Programs/Python/Python312/python.exe` in
this environment (adjust if picking this up on a different machine --
that's literally why this file exists).

## Git / repo notes

- Remote: `https://github.com/luisoleaa/Nueral-Networks.git`, this project
  lives under `tetris_model/` in that repo, branch `dev`.
- `.gitignore` in this directory excludes generated checkpoints/logs/csvs/
  pngs/pycache -- only source + the two GA artifact files
  (`evolve_ckpt.npz`, `evolved_weights.npz`) are tracked. If you add new
  persisted state you want kept, force-add it explicitly
  (`git add -f <path>`) and note why in the `.gitignore` comment.
- Local git identity for this repo (not global) is set to
  `luisoleaa` / `luisolea097@gmail.com`.

## If you're picking this up fresh

1. Run `python heuristic_bot.py` to confirm the bot still works
   (70-150+ lines/game). If it doesn't, something broke -- start there,
   not with the GA or NN.
2. If you want to keep improving play quality, resume `evolve_bot.py`
   (raise `MAX_PIECES_SIM`/`N_GENERATIONS` first, see "GA weight
   evolution" above) rather than going back to the NN approaches.
3. Only revisit `main.py`/`dagger_train.py` if you have a specific new
   idea for why the NN would work now that it didn't before (e.g. a
   genuinely different architecture) -- re-running the same approach
   longer is not expected to help, that was already tried.
