import gym_tetris
from gym_tetris.actions import MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import os
import csv
import random
from model import Layer, Relu, Adam, softmax, gradient_check
from functions import compute_gae, compute_ppo_loss, get_minibatches, save_checkpoint, load_checkpoint
from grid import get_state_vector, column_heights, count_holes, binary_board, board_potential, row_progress_score
CKPT_PATH = "Tetris_ppo.npz"
SEED = 10
np.random.seed(SEED)

env = gym_tetris.make('TetrisA-v1')
env = JoypadSpace(env, MOVEMENT)


state = get_state_vector(env)
state_size = state.shape[1]
learning_rate = 5e-5

N_WARMUP_ITERS = 30      # rollouts spent fitting the critic before the policy moves at all
VALUE_WARMUP_LR = 1e-3   # isolated to value_head, can't destabilize the policy

W_lines = 5.0
W_ROW_PROGRESS = 0.05   # weight on the dense row-completion shaping term

gamma = 0.99

W_TOPOUT = 3


num_actions = 12

Layer1 = Layer(state_size, 128)
Layer2 = Layer(128, 64)

Relu1 = Relu()
Relu2 = Relu()

policy_head = Layer(64, num_actions) # outputs one score per action
value_head  = Layer(64, 1)             # outputs one scalar

layers = [Layer1, Layer2, policy_head, value_head]
start_iteration = 0
if os.path.exists(CKPT_PATH):
    start_iteration = load_checkpoint(CKPT_PATH, layers) +1
    print(f"resumed at iteration {start_iteration}")
max_grad_norm = 0.5
opt = Adam([p for l in layers for p in (l.weights, l.biases)], lr=learning_rate)
value_warmup_opt = Adam([value_head.weights, value_head.biases], lr=VALUE_WARMUP_LR)

done = True
terminated = False
truncated = False

n_iterations = 3000   # how many times we collect a fresh rollout and train on it
n_steps = 2048         # environment steps collected per rollout
n_epochs = 4            # PPO passes over each rollout before it's discarded
batch_size = 64

# training log -- one row per iteration, appended so a resumed run just
# keeps adding to the file. if you restart training from a fresh checkpoint
# delete train_log.csv first or the old and new rows will mix together
LOG_PATH = "train_log.csv"
LOG_FIELDS = [
    "iteration", "env_steps", "episodes",
    "ep_return_mean", "ep_return_max", "ep_len_mean", "reward_per_step",
    "return_mean", "adv_mean", "adv_std", "explained_var",
    "policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm",
]
_log_is_new = (not os.path.exists(LOG_PATH)) or os.path.getsize(LOG_PATH) == 0
log_file = open(LOG_PATH, "a", newline="")
log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
if _log_is_new:
    log_writer.writeheader()
    log_file.flush()

for iteration in range(start_iteration, n_iterations):
    is_warmup = iteration < N_WARMUP_ITERS
    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []

    if done:
        state = env.reset()
        phi_prev = board_potential(binary_board(env))
        row_prog_prev = row_progress_score(binary_board(env))
        prev_lines = 0

    for step in range(n_steps):
        if done:
            state = env.reset()
            phi_prev = board_potential(binary_board(env))
            row_prog_prev = row_progress_score(binary_board(env))
            prev_lines = 0

        state = get_state_vector(env)
        out1 = Layer1.forward(state)
        Relu1_output = Relu1.forward(out1)
        out2 = Layer2.forward(Relu1_output)
        Relu2_output = Relu2.forward(out2)

        action_logits = policy_head.forward(Relu2_output)   # shape (1, 12)
        state_value = value_head.forward(Relu2_output)

        action_probs = softmax(action_logits) # shape (1,12), sums to 1 for whole probability
        action = np.random.choice(num_actions, p = action_probs[0]) # picking 1 action based on probability
        # saving logs
        log_prob = np.log(action_probs[0, action] + 1e-8)
        

        k = 4
        crashed = False
        for j in range(k):
            try:
                next_obs, reward, done, info = env.step(action)
            except OSError as e:
                # nes_py's c extension randomly throws a native access
                # violation after enough accumulated env.step() calls on one
                # env instance. cant recover it in place so just recreate the
                # env and end this rollout early (still trains on whatever we
                # collected so far), next iteration's reset starts clean
                print(f"  [engine fault at iteration {iteration}, step {step}: {e}; "
                      f"recreating env, ending this rollout early]")
                env.close()
                env = gym_tetris.make('TetrisA-v1')
                env = JoypadSpace(env, MOVEMENT)
                done = True
                crashed = True
                break
            if done:
                break
        if crashed:
            break
        # saving states, actions, logs, state_values, and rewards
        states.append(state[0])
        
        actions.append(action)
        log_probs.append(log_prob)
        values.append(state_value[0,0])


        bb_now = binary_board(env)
        phi_now = board_potential(bb_now)
        row_prog_now = row_progress_score(bb_now)
        lines_cleared = info['number_of_lines'] - prev_lines
        prev_lines = info['number_of_lines']
        if done:
            shaped = W_lines * lines_cleared - W_TOPOUT
        elif lines_cleared > 0:
            # a clear just happened: the board-shift from removing full
            # rows makes the row-progress delta a noisy, sometimes-negative
            # artifact here (the cleared row's large contribution vanishes)
            # -- skip it this step, the W_lines bonus already rewards the
            # clear correctly on its own.
            shaped = W_lines * lines_cleared + (phi_now - phi_prev)
        else:
            shaped = (W_lines * lines_cleared
                      + (phi_now - phi_prev)
                      + W_ROW_PROGRESS * (row_prog_now - row_prog_prev))
        phi_prev = phi_now
        row_prog_prev = row_prog_now

        rewards.append(shaped)          # <- was: rewards.append(reward)

        dones.append(done)

        # env.render()


    rewards = np.array(rewards, dtype=np.float64)
    values  = np.array(values,  dtype=np.float64)
    dones   = np.array(dones,   dtype=np.float64)
    states    = np.array(states)          # (n_steps, 238)
    actions   = np.array(actions)         # (n_steps,)
    log_probs = np.array(log_probs)       # (n_steps,)
    rewards   = np.array(rewards, dtype=np.float64)
    values    = np.array(values,  dtype=np.float64)
    dones     = np.array(dones,   dtype=np.float64)


    # one extra step forward
    next_state = get_state_vector(env)
    x = Relu1.forward(Layer1.forward(next_state))
    x = Relu2.forward(Layer2.forward(x))
    bootstrap_value = value_head.forward(x)[0, 0]

    advantages, returns = compute_gae(rewards, values, bootstrap_value, dones, gamma, lam=0.95)

    adv_std = advantages.std()
    if adv_std > 1e-6:
        advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)



    # collect PPO-update diagnostics across every minibatch this iteration
    _pl, _vl, _ent, _kl, _clip, _gn = [], [], [], [], [], []

    for epoch in range(n_epochs):
        for batch in get_minibatches(states, actions, log_probs, advantages, returns, batch_size= batch_size ):
            
            out1 = Layer1.forward(batch['states'])
            Relu1_output = Relu1.forward(out1)
            out2 = Layer2.forward(Relu1_output)
            Relu2_output = Relu2.forward(out2)
            new_action_logits = policy_head.forward(Relu2_output)   # shape (1, 12)
            new_state_value = value_head.forward(Relu2_output)

            new_action_probs = softmax(new_action_logits) # shape (1,12), sums to 1 for whole probability

            

            # batch['actions'] holds the actions actually taken during rollout, for this minibatch
            new_log_probs = np.log(new_action_probs[np.arange(len(batch['actions'])), batch['actions']] + 1e-8)

            new_values = new_state_value[:, 0]      # shape (64,), not [ (64,) ]


            total_loss, policy_loss, value_loss, entropy_loss, dL_logits, dL_values = compute_ppo_loss(new_log_probs, batch['old_log_probs'], batch['actions'] ,new_action_probs, batch['advantages'], new_values, batch['returns'], clip_eps = .2, value_coef = .5, entropy_coef = .004)

            # now that we have our loss we loop that backward to get our new weights and biases
            dL_dRelu2_from_policy = policy_head.backward(dL_logits)
            dL_dRelu2_from_value  = value_head.backward(dL_values[:,None]) # changes shape when doing dot product

            # merging the two outputs
            dL_dRelu2 = dL_dRelu2_from_policy + dL_dRelu2_from_value

            dL_out2 = Relu2.backward(dL_dRelu2)
            dL_Relu1 = Layer2.backward(dL_out2)
            dL_out1 = Relu1.backward(dL_Relu1)
            dL_dstate = Layer1.backward(dL_out1)

            # Layer1.weights -= learning_rate * Layer1.dweights
            # Layer1.biases -= learning_rate * Layer1.dbiases
            # Layer2.weights -= learning_rate * Layer2.dweights
            # Layer2.biases -= learning_rate * Layer2.dbiases
            # policy_head.weights -= learning_rate * policy_head.dweights
            # policy_head.biases -= learning_rate * policy_head.dbiases
            # value_head.weights -= learning_rate * value_head.dweights
            # value_head.biases -= learning_rate * value_head.dbiases

            # Adam optimizer param update
            if is_warmup:
                # value warmup -- only fit the critic here, leave the
                # BC-pretrained trunk/policy alone so a still-random critic
                # cant shove the good policy around with noisy advantages
                grads = [value_head.dweights, value_head.dbiases]
            else:
                grads = [g for l in layers for g in (l.dweights, l.dbiases)]

            # gradient-norm clip
            total_norm = np.sqrt(sum(np.sum(g *g) for g in grads))
            if total_norm > max_grad_norm:
                scale = max_grad_norm / (total_norm + 1e-8)
                for g in grads:
                    g *= scale

            (value_warmup_opt if is_warmup else opt).step(grads)

            # --- diagnostics only: nothing below here changes training ---
            _ratios = np.exp(new_log_probs - batch['old_log_probs'])
            _pl.append(policy_loss)
            _vl.append(value_loss)
            _ent.append(-entropy_loss)                       # raw policy entropy
            _kl.append(np.mean((_ratios - 1.0) - np.log(_ratios + 1e-8)))
            _clip.append(np.mean(np.abs(_ratios - 1.0) > 0.2))
            _gsq = sum(np.sum(l.dweights ** 2) + np.sum(l.dbiases ** 2) for l in layers)
            _gn.append(np.sqrt(_gsq))

    save_checkpoint(CKPT_PATH, iteration, layers)

    # write one training-log row for this iteration 
    # split the rollout's rewards into finished episodes -> per-game return
    ep_returns, ep_lengths, _cr, _cl = [], [], 0.0, 0
    for _rw, _dn in zip(rewards, dones):
        _cr += _rw
        _cl += 1
        if _dn:
            ep_returns.append(_cr)
            ep_lengths.append(_cl)
            _cr, _cl = 0.0, 0

    row = {
        "iteration": iteration,
        "env_steps": (iteration + 1) * n_steps,
        "episodes": len(ep_returns),
        "ep_return_mean": np.mean(ep_returns) if ep_returns else float("nan"),
        "ep_return_max": np.max(ep_returns) if ep_returns else float("nan"),
        "ep_len_mean": np.mean(ep_lengths) if ep_lengths else float("nan"),
        "reward_per_step": rewards.mean(),
        "return_mean": returns.mean(),
        "adv_mean": advantages.mean(),
        "adv_std": advantages.std(),
        "explained_var": 1.0 - np.var(returns - values) / (np.var(returns) + 1e-8),
        "policy_loss": np.mean(_pl),
        "value_loss": np.mean(_vl),
        "entropy": np.mean(_ent),
        "approx_kl": np.mean(_kl),
        "clip_frac": np.mean(_clip),
        "grad_norm": np.mean(_gn),
    }
    log_writer.writerow({k: (round(float(v), 5) if isinstance(v, float) else v)
                         for k, v in row.items()})
    log_file.flush()

    print(f"it {iteration:4d} | ep_ret {row['ep_return_mean']:.1f} "
          f"(max {row['ep_return_max']:.1f}, n={row['episodes']}) | "
          f"ent {row['entropy']:.3f} | kl {row['approx_kl']:.4f} | "
          f"clip {row['clip_frac']:.2f} | EV {row['explained_var']:.2f} | "
          f"|g| {row['grad_norm']:.2f}")

env.close()
log_file.close()





        