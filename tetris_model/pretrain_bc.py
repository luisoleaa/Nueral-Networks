# pretrain_bc.py -- behavior cloning pretraining. warm starts the same
# network architecture main.py uses by having it imitate the heuristic bot's
# demo data (see heuristic_bot.py / generate_demos.py) before PPO takes over.
#
# only trains the shared trunk + policy_head (plain softmax cross entropy,
# "predict the demonstrator's action"). value_head stays at its random init
# -- value learning is comparatively easy so PPO should pick it up fast once
# fine tuning starts, and theres nothing to imitate there anyway since the
# bot never produced value estimates to begin with.

import os
import numpy as np
from model import Layer, Relu, Adam, softmax
from functions import save_checkpoint
from grid import get_state_vector
import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace

DEMO_PATH = "bc_demos.npz"
HUMAN_DEMO_PATH = "human_demos.npz"   # optional -- see record_human_play.py
OUT_CKPT_PATH = "Tetris_bc_pretrained.npz"   # separate file, doesnt touch
                                              # Tetris_ppo.npz or whatever PPO
                                              # run is in progress. copy it over
                                              # manually (back up the current
                                              # checkpoint first!) if you want
                                              # to start PPO fresh from this
SEED = 10
np.random.seed(SEED)

learning_rate = 1e-3
n_epochs = 15
batch_size = 128
max_grad_norm = 0.5
val_frac = 0.05

# build the exact same architecture main.py uses
env = gym_tetris.make('TetrisA-v1')
env = JoypadSpace(env, MOVEMENT)
state_size = get_state_vector(env).shape[1]
num_actions = 12
env.close()

Layer1 = Layer(state_size, 128)
Layer2 = Layer(128, 64)
Relu1, Relu2 = Relu(), Relu()
policy_head = Layer(64, num_actions)
value_head  = Layer(64, 1)   # untouched by BC, saved as-is (random init)

layers = [Layer1, Layer2, policy_head, value_head]
# always starts from a fresh random init -- the whole point of BC pretraining
# is to replace the cold start, not build on top of an already PPO'd policy

bc_params = [Layer1.weights, Layer1.biases, Layer2.weights, Layer2.biases,
             policy_head.weights, policy_head.biases]
opt = Adam(bc_params, lr=learning_rate)

# load demos
data = np.load(DEMO_PATH)
states, actions = data['states'], data['actions']
print(f"bot demos: {len(states)} pairs")

if os.path.exists(HUMAN_DEMO_PATH):
    human = np.load(HUMAN_DEMO_PATH)
    print(f"human demos: {len(human['states'])} pairs (merging in)")
    states = np.concatenate([states, human['states'].astype(states.dtype)])
    actions = np.concatenate([actions, human['actions'].astype(actions.dtype)])
else:
    print(f"no {HUMAN_DEMO_PATH} found -- training on bot demos only "
          f"(run record_human_play.py first if you want to add your own play)")

n = len(states)
idx = np.random.permutation(n)
n_val = max(1, int(n * val_frac))
val_idx, train_idx = idx[:n_val], idx[n_val:]
print(f"loaded {n} demo pairs -> {len(train_idx)} train / {len(val_idx)} val")


def forward(batch_states):
    out1 = Layer1.forward(batch_states)
    r1 = Relu1.forward(out1)
    out2 = Layer2.forward(r1)
    r2 = Relu2.forward(out2)
    logits = policy_head.forward(r2)
    return softmax(logits)


def accuracy(idxs):
    probs = forward(states[idxs])
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == actions[idxs]))


for epoch in range(n_epochs):
    np.random.shuffle(train_idx)
    losses = []
    for start in range(0, len(train_idx), batch_size):
        batch_idx = train_idx[start:start + batch_size]
        bs = states[batch_idx]
        ba = actions[batch_idx]
        N = len(ba)

        probs = forward(bs)
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

    train_acc = accuracy(train_idx)
    val_acc = accuracy(val_idx)
    print(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")

# save as iteration -1 so main.py's `start_iteration = load_checkpoint(...) + 1`
# would resume PPO fine-tuning at iteration 0, with these pretrained weights
# -- once you copy this file over Tetris_ppo.npz yourself.
save_checkpoint(OUT_CKPT_PATH, -1, layers)
print(f"saved pretrained weights to {OUT_CKPT_PATH}")
print("this does NOT touch Tetris_ppo.npz -- to start PPO from these weights,")
print("back up your current checkpoint/log, then copy this file over Tetris_ppo.npz")
