import numpy as np
import os

CKPT_PATH = "Tetris_ppo.npz"

def compute_gae(rewards, values, next_value, dones, gamma, lam):
    t = len(rewards)
    advantages = np.zeros(t)
    gae = 0.0

    for x in reversed(range(t)):
        if x == t - 1:
            next_non_terminal = 1.0 - dones[x]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[x]
            next_val = values[x+1]

        # calculate TD Error
        delta = rewards[x] + gamma * next_val * next_non_terminal - values[x]
        # accumulte gae
        gae = delta + gamma  * lam * next_non_terminal * gae
        advantages[x] = gae

    returns = advantages + values
    # the advantage is how much better or worse a specific action is compared to the average expected behavior at a state
    return advantages, returns

def get_minibatches(states, actions, old_log_probs, advantages, returns, batch_size):
    t = len(states)
    indicies = np.arange(t)
    np.random.shuffle(indicies)
    # chooses batch index from start of randomly shuffled epochs    
    for start in range(0,t,batch_size):
        end = start + batch_size
        batch_idx = indicies[start:end] 

        yield{
            'states': states[batch_idx],
            'actions': actions[batch_idx],
            'old_log_probs': old_log_probs[batch_idx],
            'advantages': advantages[batch_idx],
            'returns': returns[batch_idx],
        }



def compute_ppo_loss(new_log_probs, log_probs, actions, new_action_probs, advantages, new_values, returns, clip_eps, value_coef, entropy_coef):

    N = len(advantages)
    # forard pass policy, value and entropy losses
    ratios = np.exp(new_log_probs - log_probs)

    surr1 = ratios * advantages
    surr2 = np.clip(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

    policy_loss = -np.minimum(surr1,surr2).mean()

    value_loss = np.mean((new_values - returns) ** 2)
    
    dL_values = value_coef * 2 * (new_values - returns) / N

    entropy = -np.sum(new_action_probs *np.log(new_action_probs + 1e-8), axis = 1)
    entropy_loss = -np.mean(entropy)

    total_loss = policy_loss + value_loss * value_coef + entropy_loss * entropy_coef
    
    unclipped_mask = (surr1 <= surr2)  # True -> gradient flows /False -> clipped
    g = np.where(unclipped_mask, -advantages * ratios, 0.0) / N #shape (N,)


    # spreading g across all 12 logits
    dL_logits = np.zeros_like(new_action_probs)
    for sample_idx in range(len(g)):
        a = actions[sample_idx]
        dL_logits[sample_idx] = g[sample_idx] * (-new_action_probs[sample_idx])
        dL_logits[sample_idx,a] += g[sample_idx]

    log_probs_all = np.log(new_action_probs + 1e-8)
    dL_entropy_dlogits = new_action_probs * (log_probs_all + entropy[:, None]) / N
    dL_logits += entropy_coef * dL_entropy_dlogits
    
    
    return total_loss, policy_loss, value_loss, entropy_loss, dL_logits, dL_values


def save_checkpoint(path, iteration, layers):
    arrays = {"iteration": np.array(iteration)}
    for i, layer in enumerate(layers):
        arrays[f"w{i}"] = layer.weights
        arrays[f"b{i}"] = layer.biases
    tmp = path + ".tmp.npz"       # has to end in .npz or np.savez renames it on us
    np.savez(tmp, **arrays)
    os.replace(tmp, path)          # atomic -- a crash mid save wont corrupt the old file

def load_checkpoint(path, layers):
    try:
        data = np.load(path)      # never allow_pickle=True
    except (ValueError, OSError) as e:
        print(f"could not read {path} ({e}); starting fresh")
        return -1                 # +1 in main.py -> start_iteration 0
    for i, layer in enumerate(layers):
        assert layer.weights.shape == data[f"w{i}"].shape, f"layer {i} shape changed"
        layer.weights = data[f"w{i}"]
        layer.biases  = data[f"b{i}"]
    return int(data["iteration"])
