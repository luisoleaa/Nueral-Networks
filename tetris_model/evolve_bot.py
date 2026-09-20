# evolve_bot.py -- tries to evolve heuristic_bot's placement-scoring weights
# with a genetic algorithm instead of me just guessing/hand-tuning them.
#
# fitness gets evaluated completely offline (no live emulator, so no nes_py
# crash risk either) with a full-game simulator built out of heuristic_bot's
# own PIECE_SHAPES/simulate_drop. whatever genome wins gets validated live at
# the very end thru run_bot_episode, just to make sure the offline sim
# actually matches how the real game behaves.
#
# genome is [w_lines, w_agg_height, w_holes, w_bumpiness] -- same order as
# grid.DEFAULT_PLACEMENT_WEIGHTS / grid.placement_score, the exact formula
# heuristic_bot.search_best_placement already uses live.

import csv
import os
import time

import numpy as np

from grid import DEFAULT_PLACEMENT_WEIGHTS, placement_score
from heuristic_bot import PIECE_SHAPES, make_env, run_bot_episode, simulate_drop

FAMILIES = np.array(['T', 'J', 'Z', 'S', 'O', 'L', 'I'])

N_GENES = len(DEFAULT_PLACEMENT_WEIGHTS)   # 4

POP_SIZE = 40
N_GENERATIONS = 20      # sanity-check budget; raise once timing is known
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.7
ELITISM = 2
GAMES_PER_EVAL = 3
MAX_PIECES_SIM = 400    # matches the max_pieces convention used by heuristic_bot.py's
                        # own self-test and dagger_train.py's live rollouts; a single
                        # un-capped offline game can run 2000+ pieces without topping
                        # out, which made a "sanity check" generation take ~11 minutes
BASE_SEED = 10           # matches SEED convention in main.py/dagger_train.py

SIGMA0 = np.array([4.0, 0.15, 0.10, 0.06])   # ~20-25% of |DEFAULT_PLACEMENT_WEIGHTS| per gene
SIGMA_DECAY = 0.97
SIGMA_FLOOR_RATIO = 0.05

GENOME_CKPT_PATH = "evolve_ckpt.npz"
BEST_WEIGHTS_PATH = "evolved_weights.npz"
LOG_PATH = "evolve_log.csv"

LOG_FIELDS = [
    "generation", "best_fitness", "mean_fitness", "std_fitness", "worst_fitness",
    "best_genome_w_lines", "best_genome_w_height", "best_genome_w_holes", "best_genome_w_bump",
    "sigma_mean", "elapsed_sec",
]


# offline simulator

def best_placement_offline(board, family, weights):
    # same brute force scan as heuristic_bot.search_best_placement (same
    # tie-break too). returns None if nothing fits anymore -- thats our
    # offline stand in for the emulator's `done` flag (a simulated top out)
    best = None   # (score, new_board, n_cleared)
    for name, shape in PIECE_SHAPES.items():
        if name[0] != family:
            continue
        for col in range(board.shape[1]):
            result = simulate_drop(board, shape, col)
            if result is None:
                continue
            new_board, n_cleared = result
            score = placement_score(new_board, n_cleared, weights)
            if best is None or score > best[0]:
                best = (score, new_board, n_cleared)
    return None if best is None else (best[1], best[2])


def simulate_game(weights, piece_sequence):
    board = np.zeros((20, 10), dtype=np.int8)
    total_lines, pieces_placed = 0, 0
    for family in piece_sequence:
        result = best_placement_offline(board, family, weights)
        if result is None:
            break   # simulated top-out
        board, n_cleared = result
        total_lines += n_cleared
        pieces_placed += 1
    return total_lines, pieces_placed


def make_piece_sequences(seed, n_games, max_pieces):
    r = np.random.RandomState(seed)
    return [r.choice(FAMILIES, size=max_pieces) for _ in range(n_games)]


def evaluate_fitness(genome, piece_sequences):
    lines = [simulate_game(genome, seq)[0] for seq in piece_sequences]
    return float(np.mean(lines))


# GA operators

def tournament_select(population, fitnesses, k, rng):
    idx = rng.choice(len(population), size=k, replace=False)
    return population[idx[np.argmax(fitnesses[idx])]]


def arithmetic_crossover(p1, p2, rng):
    alpha = rng.uniform(0.0, 1.0, size=N_GENES)
    return alpha * p1 + (1 - alpha) * p2


def mutate(genome, sigma, rng):
    return genome + rng.normal(0.0, sigma, size=N_GENES)


# checkpointing, same pattern as functions.py's save/load_checkpoint

def save_genome_checkpoint(path, generation, best_genome, best_fitness, population, sigma):
    arrays = {
        "generation": np.array(generation),
        "best_genome": best_genome,
        "best_fitness": np.array(best_fitness),
        "population": population,
        "sigma": sigma,
    }
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_genome_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
    except (ValueError, OSError) as e:
        print(f"could not read {path} ({e}); starting fresh")
        return None
    return {
        "generation": int(data["generation"]),
        "best_genome": data["best_genome"],
        "best_fitness": float(data["best_fitness"]),
        "population": data["population"],
        "sigma": data["sigma"],
    }


def save_best_weights(path, generation, best_genome, best_fitness):
    # smaller standalone file -- just the weights, none of the population/sigma
    # stuff, since thats all a live-play caller actually cares about
    tmp = path + ".tmp.npz"
    np.savez(tmp, generation=np.array(generation), weights=best_genome, fitness=np.array(best_fitness))
    os.replace(tmp, path)


def load_best_weights(path):
    try:
        data = np.load(path)
    except (ValueError, OSError, FileNotFoundError) as e:
        print(f"could not read {path} ({e})")
        return None
    return data["weights"]


# live validation

def validate_live(weights, n_episodes=10, max_pieces=400, label="evolved"):
    env = make_env()
    results = []
    for i in range(n_episodes):
        try:
            result = run_bot_episode(env, record=False, max_pieces=max_pieces, weights=weights)
        except OSError as e:
            print(f"  [engine fault during {label} validation ep {i}: {e}; recreating env]")
            env.close()
            env = make_env()
            continue
        results.append(result)
        print(f"  {label} live ep {i}: pieces={result['pieces']:4d}  lines={result['lines']}")
    env.close()
    return results


# main GA loop

def run_evolution():
    rng = np.random.RandomState(BASE_SEED)

    ckpt = load_genome_checkpoint(GENOME_CKPT_PATH)
    if ckpt is None:
        population = np.tile(DEFAULT_PLACEMENT_WEIGHTS, (POP_SIZE, 1))
        population[1:] += rng.normal(0.0, SIGMA0, size=(POP_SIZE - 1, N_GENES))
        # individual 0 stays the exact current hard-coded baseline, so the GA
        # cant ever report a "best" thats actually worse than what we already have
        start_gen = 0
        best_genome, best_fitness = DEFAULT_PLACEMENT_WEIGHTS.copy(), -np.inf
        print("starting evolution fresh")
    else:
        population = ckpt["population"]
        start_gen = ckpt["generation"] + 1
        best_genome, best_fitness = ckpt["best_genome"], ckpt["best_fitness"]
        print(f"resuming evolution from {GENOME_CKPT_PATH} at generation {start_gen}")

    _log_is_new = (not os.path.exists(LOG_PATH)) or os.path.getsize(LOG_PATH) == 0
    log_file = open(LOG_PATH, "a", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
    if _log_is_new:
        log_writer.writeheader()
        log_file.flush()

    for generation in range(start_gen, N_GENERATIONS):
        t0 = time.time()
        piece_sequences = make_piece_sequences(BASE_SEED * 1000 + generation, GAMES_PER_EVAL, MAX_PIECES_SIM)
        fitnesses = np.array([evaluate_fitness(ind, piece_sequences) for ind in population])

        order = np.argsort(-fitnesses)
        population, fitnesses = population[order], fitnesses[order]

        if fitnesses[0] > best_fitness:
            best_fitness, best_genome = fitnesses[0], population[0].copy()
            save_best_weights(BEST_WEIGHTS_PATH, generation, best_genome, best_fitness)

        sigma = SIGMA0 * max(SIGMA_DECAY ** generation, SIGMA_FLOOR_RATIO)

        new_population = [population[i].copy() for i in range(ELITISM)]
        while len(new_population) < POP_SIZE:
            if rng.random_sample() < CROSSOVER_RATE:
                p1 = tournament_select(population, fitnesses, TOURNAMENT_K, rng)
                p2 = tournament_select(population, fitnesses, TOURNAMENT_K, rng)
                child = arithmetic_crossover(p1, p2, rng)
            else:
                child = tournament_select(population, fitnesses, TOURNAMENT_K, rng).copy()
            new_population.append(mutate(child, sigma, rng))
        population = np.array(new_population)

        row = {
            "generation": generation,
            "best_fitness": fitnesses[0],
            "mean_fitness": fitnesses.mean(),
            "std_fitness": fitnesses.std(),
            "worst_fitness": fitnesses.min(),
            "best_genome_w_lines": population[0][0],
            "best_genome_w_height": population[0][1],
            "best_genome_w_holes": population[0][2],
            "best_genome_w_bump": population[0][3],
            "sigma_mean": sigma.mean(),
            "elapsed_sec": time.time() - t0,
        }
        log_writer.writerow({k: (round(float(v), 5) if isinstance(v, float) else v) for k, v in row.items()})
        log_file.flush()

        print(f"gen {generation:4d} | best {row['best_fitness']:7.1f} | mean {row['mean_fitness']:7.1f} "
              f"+/- {row['std_fitness']:5.1f} | sigma {row['sigma_mean']:.3f} | {row['elapsed_sec']:.1f}s")

        save_genome_checkpoint(GENOME_CKPT_PATH, generation, best_genome, best_fitness, population, sigma)

    log_file.close()
    return best_genome, best_fitness


if __name__ == "__main__":
    best_genome, best_fitness = run_evolution()
    print(f"\nbest genome: {best_genome}  fitness(offline mean lines)={best_fitness:.1f}")

    print("\nvalidating baseline (hard-coded) weights live...")
    baseline_results = validate_live(DEFAULT_PLACEMENT_WEIGHTS, n_episodes=10, label="baseline")

    print("\nvalidating evolved weights live...")
    evolved_results = validate_live(best_genome, n_episodes=10, label="evolved")

    baseline_lines = np.mean([r['lines'] for r in baseline_results]) if baseline_results else float('nan')
    evolved_lines = np.mean([r['lines'] for r in evolved_results]) if evolved_results else float('nan')
    print(f"\nbaseline mean lines: {baseline_lines:.1f}")
    print(f"evolved  mean lines: {evolved_lines:.1f}")
