# dagger_train.py -- DAgger (dataset aggregation), trying to fix the
# compounding error problem that plain BC ran into.
#
# the issue: pretrain_bc.py only ever trained on the expert bot's OWN
# trajectories -- states the bot actually visited while playing well. it
# never had to demonstrate recovering from a bad placement because it never
# made one. and we confirmed live that Tetris_bc_pretrained.npz clears 0
# lines when you actually roll it out (greedy or stochastic), even though it
# hits 75% "matches the demonstrator" accuracy on the validation set. this is
# the classic distributional shift problem w/ behavior cloning -- small
# per-step mistakes stack up over the ~10-20 macro steps each piece takes,
# and once its off the demonstrated distribution the network just has
# nothing to go on.
#
# the fix: each round, roll the CURRENT learner out live (so it actually
# visits its own mistakes), ask the expert bot what it should have done at
# every single decision point (not just once per piece), throw those
# (state, expert_action) pairs into the training set, retrain. that
# teaches recovery, which plain BC just can't.
#
# trains its own separate checkpoint (Tetris_dagger.npz), doesn't touch
# Tetris_bc_pretrained.npz or Tetris_ppo.npz.

import os
import numpy as np
from model import Layer, Relu, Adam, softmax
from functions import save_checkpoint, load_checkpoint
from grid import get_state_vector
from heuristic_bot import (make_env, step_macro, search_best_placement,
                            NOOP_IDX, A_IDX, LEFT_IDX, RIGHT_IDX, DOWN_IDX)

SEED = 10
np.random.seed(SEED)

BC_CKPT_PATH = "Tetris_bc_pretrained.npz"   # starting point, read-only
OUT_CKPT_PATH = "Tetris_dagger.npz"         # this script's own checkpoint
DEMO_PATH = "bc_demos.npz"                  # original bot demos, kept in the mix every round
DAGGER_DEMO_PATH = "dagger_demos.npz"       # growing aggregated corrective-label dataset

N_ROUNDS = 30
EPISODES_PER_ROUND = 15
MAX_PIECES_PER_EPISODE = 400
MAX_MACRO_STEPS_PER_PIECE = 80   # safety bound; expert alone needs at most ~4*2 + 15*2 + a few drops
RETRAIN_EPOCHS_PER_ROUND = 8
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 0.5
LIVE_EVAL_EPISODES = 10
LIVE_EVAL_MAX_STEPS = 1500


class ExpertController:
    # basically a query-only version of heuristic_bot.execute_placement's
    # control logic -- given a live env, tells you the single action the
    # expert bot would press right now to carry out its (target_name,
    # target_col) plan for the current piece, without actually stepping the
    # env. has to mirror execute_placement's press-then-release pattern
    # exactly (rotation/movement are edge triggered here, no auto repeat
    # unless you actually release -- see heuristic_bot.py's docstring at the
    # top), otherwise the network would learn to just spam a button and fall
    # apart on anything needing 2+ rotations.
    #
    # re-reads the live state every call, so even if the learner does
    # something different than what the expert suggested, the next label
    # still gets computed off the real resulting state -- not assuming its
    # own suggestion actually happened

    def __init__(self, target_name, target_col):
        self.target_name = target_name
        self.target_col = target_col
        self.phase = 'rotate'
        self.awaiting_release = False
        self.rotate_tries = 0
        self.steer_tries = 0
        self.last_px = None

    def next_action(self, env):
        u = env.unwrapped

        if self.awaiting_release:
            self.awaiting_release = False
            return NOOP_IDX

        if self.phase == 'rotate':
            if u._current_piece == self.target_name or self.rotate_tries >= 4:
                self.phase = 'steer'
                return self.next_action(env)
            self.rotate_tries += 1
            self.awaiting_release = True
            return A_IDX

        if self.phase == 'steer':
            px = u.ram[0x40]
            if (px == self.target_col or self.steer_tries >= 15
                    or (self.last_px is not None and px == self.last_px)):
                self.phase = 'drop'
                return self.next_action(env)
            self.steer_tries += 1
            self.last_px = px
            self.awaiting_release = True
            return RIGHT_IDX if px < self.target_col else LEFT_IDX

        return DOWN_IDX   # phase == 'drop'


def build_network(state_size, num_actions):
    Layer1 = Layer(state_size, 128)
    Layer2 = Layer(128, 64)
    Relu1, Relu2 = Relu(), Relu()
    policy_head = Layer(64, num_actions)
    value_head = Layer(64, 1)   # untouched by DAgger, same as pretrain_bc.py
    return Layer1, Layer2, Relu1, Relu2, policy_head, value_head


def forward_probs(net, states):
    Layer1, Layer2, Relu1, Relu2, policy_head, _ = net
    out1 = Layer1.forward(states)
    r1 = Relu1.forward(out1)
    out2 = Layer2.forward(r1)
    r2 = Relu2.forward(out2)
    logits = policy_head.forward(r2)
    return softmax(logits)


def collect_dagger_round(env, net, num_actions, n_episodes, max_pieces):
    # rolls the current learner out live and labels every decision point with
    # what the expert would've done. returns the new (state, action) pairs +
    # some stats, and the env (which might be a recreated one if it crashed)
    new_states, new_actions = [], []
    total_lines, total_pieces = 0, 0

    for _ in range(n_episodes):
        try:
            env.reset()
        except OSError:
            env.close()
            env = make_env()
            env.reset()

        ep_lines, prev_lines = 0, 0
        target_name, target_col = search_best_placement(env)
        controller = ExpertController(target_name, target_col)
        piece_before = env.unwrapped._current_piece
        done, crashed = False, False
        pieces_done = 0

        for _piece_i in range(max_pieces):
            for _ in range(MAX_MACRO_STEPS_PER_PIECE):
                state = get_state_vector(env)
                expert_action = controller.next_action(env)
                new_states.append(state[0])
                new_actions.append(expert_action)

                probs = forward_probs(net, state)[0]
                learner_action = int(np.random.choice(num_actions, p=probs))

                try:
                    done, info = step_macro(env, learner_action)
                except OSError as e:
                    print(f"    [engine fault, ending episode early: {e}]")
                    crashed, done, info = True, True, {}

                if done:
                    ep_lines = info.get('number_of_lines', prev_lines)
                    break

                cur_piece = env.unwrapped._current_piece
                if cur_piece is not None and cur_piece != piece_before:
                    piece_before = cur_piece
                    prev_lines = info.get('number_of_lines', prev_lines)
                    target_name, target_col = search_best_placement(env)
                    controller = ExpertController(target_name, target_col)
                    break

            pieces_done += 1
            if done:
                break

        total_lines += ep_lines
        total_pieces += pieces_done

        if crashed:
            env.close()
            env = make_env()

    return new_states, new_actions, total_lines, total_pieces, env


def retrain(net, opt, states, actions, epochs, batch_size, max_grad_norm):
    Layer1, Layer2, Relu1, Relu2, policy_head, _ = net
    n = len(states)
    idx = np.arange(n)
    for epoch in range(epochs):
        np.random.shuffle(idx)
        losses = []
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            bs, ba = states[batch_idx], actions[batch_idx]
            N = len(ba)

            probs = forward_probs(net, bs)
            loss = -np.mean(np.log(probs[np.arange(N), ba] + 1e-8))
            losses.append(loss)

            one_hot = np.zeros_like(probs)
            one_hot[np.arange(N), ba] = 1.0
            dL_logits = (probs - one_hot) / N

            dL_r2 = policy_head.backward(dL_logits)
            dL_out2 = Relu2.backward(dL_r2)
            dL_r1 = Layer2.backward(dL_out2)
            dL_out1 = Relu1.backward(dL_r1)
            Layer1.backward(dL_out1)

            grads = [Layer1.dweights, Layer1.dbiases, Layer2.dweights, Layer2.dbiases,
                     policy_head.dweights, policy_head.dbiases]
            total_norm = np.sqrt(sum(np.sum(g * g) for g in grads))
            if total_norm > max_grad_norm:
                scale = max_grad_norm / (total_norm + 1e-8)
                for g in grads:
                    g *= scale
            opt.step(grads)
        print(f"    epoch {epoch}: loss {np.mean(losses):.4f}")


def accuracy(net, states, actions):
    probs = forward_probs(net, states)
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == actions))


def live_eval(env, net, num_actions, n_episodes, max_steps):
    lens, lines_list = [], []
    for _ in range(n_episodes):
        try:
            env.reset()
        except OSError:
            env.close()
            env = make_env()
            env.reset()
        done, length, info = False, 0, {}
        while not done and length < max_steps:
            state = get_state_vector(env)
            probs = forward_probs(net, state)[0]
            a = int(np.argmax(probs))
            try:
                done, info = step_macro(env, a)
            except OSError:
                done = True
            length += 1
        lens.append(length)
        lines_list.append(info.get('number_of_lines', 0))
    return float(np.mean(lens)), float(np.mean(lines_list)), env


if __name__ == "__main__":
    env = make_env()
    state_size = get_state_vector(env).shape[1]
    num_actions = 12

    net = build_network(state_size, num_actions)
    Layer1, Layer2, Relu1, Relu2, policy_head, value_head = net
    layers_for_ckpt = [Layer1, Layer2, policy_head, value_head]

    if os.path.exists(OUT_CKPT_PATH):
        it = load_checkpoint(OUT_CKPT_PATH, layers_for_ckpt)
        print(f"resuming DAgger from {OUT_CKPT_PATH} (marker iteration {it})")
    else:
        it = load_checkpoint(BC_CKPT_PATH, layers_for_ckpt)
        print(f"starting DAgger fresh from {BC_CKPT_PATH} (marker iteration {it})")

    opt = Adam([Layer1.weights, Layer1.biases, Layer2.weights, Layer2.biases,
                policy_head.weights, policy_head.biases], lr=LEARNING_RATE)

    bc = np.load(DEMO_PATH)
    bc_states = bc['states'].astype(np.float32)
    bc_actions = bc['actions'].astype(np.int64)
    print(f"seed dataset: {len(bc_states)} bot-demo pairs")

    if os.path.exists(DAGGER_DEMO_PATH):
        dg = np.load(DAGGER_DEMO_PATH)
        dagger_states = list(dg['states'])
        dagger_actions = list(dg['actions'])
        print(f"resuming with {len(dagger_states)} previously-aggregated DAgger pairs")
    else:
        dagger_states, dagger_actions = [], []

    for round_i in range(N_ROUNDS):
        print(f"\n=== DAgger round {round_i} ===")
        new_states, new_actions, lines, pieces, env = collect_dagger_round(
            env, net, num_actions, EPISODES_PER_ROUND, MAX_PIECES_PER_EPISODE)
        print(f"  collected {len(new_states)} labeled steps over {EPISODES_PER_ROUND} episodes "
              f"({pieces} pieces, {lines} total lines cleared during rollout)")

        dagger_states.extend(new_states)
        dagger_actions.extend(new_actions)
        np.savez_compressed(DAGGER_DEMO_PATH,
                             states=np.array(dagger_states, dtype=np.float32),
                             actions=np.array(dagger_actions, dtype=np.int64))

        agg_states = np.concatenate([bc_states, np.array(dagger_states, dtype=np.float32)])
        agg_actions = np.concatenate([bc_actions, np.array(dagger_actions, dtype=np.int64)])

        print(f"  retraining on {len(agg_states)} total pairs "
              f"({len(bc_states)} bot-demo + {len(dagger_states)} DAgger)...")
        retrain(net, opt, agg_states, agg_actions, RETRAIN_EPOCHS_PER_ROUND, BATCH_SIZE, MAX_GRAD_NORM)
        print(f"  train accuracy: {accuracy(net, agg_states, agg_actions):.3f}")

        save_checkpoint(OUT_CKPT_PATH, -1, layers_for_ckpt)

        eval_steps, eval_lines, env = live_eval(env, net, num_actions,
                                                 LIVE_EVAL_EPISODES, LIVE_EVAL_MAX_STEPS)
        print(f"  live eval (greedy): steps {eval_steps:.0f}  lines {eval_lines:.2f}")

    env.close()
