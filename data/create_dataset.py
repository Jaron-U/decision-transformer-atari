import math
import numpy as np

import gzip
import pickle
from data.process_data import process_data

game_name = "Pong"
source_data_dir = f"downloaded_game_data/{game_name}/1/replay_logs"
dest_dir = "game_dataset"
step_size = 50
chunk_size = 10000  # meta data will be chunked by each 10000 rows
stack_size = 4      # frame stacking size (So stack_size - 1 frames are added in each states)

# initialize
cur_chunk = 0
cur_row = 0
actions_chunk = []
rtgs_chunk = []
timesteps_chunk = []
states_meta_chunk = []

# run loop for the downloaded buffer
for buffer_num in range(5):
    # load data
    acts_file = f"{source_data_dir}/$store$_action_ckpt.{buffer_num}.gz"
    obss_file = f"{source_data_dir}/$store$_observation_ckpt.{buffer_num}.gz"
    rews_file = f"{source_data_dir}/$store$_reward_ckpt.{buffer_num}.gz"
    trms_file = f"{source_data_dir}/$store$_terminal_ckpt.{buffer_num}.gz"
    invs_file = f"{source_data_dir}/invalid_range_ckpt.{buffer_num}.gz"
    with gzip.GzipFile(filename=acts_file) as f:
        acts = np.load(f, allow_pickle=False)
    with gzip.GzipFile(filename=obss_file) as f:
        obss = np.load(f, allow_pickle=False)
    with gzip.GzipFile(filename=rews_file) as f:
        rews = np.load(f, allow_pickle=False)
    with gzip.GzipFile(filename=trms_file) as f:
        trms = np.load(f, allow_pickle=False)
    with gzip.GzipFile(filename=invs_file) as f:
        invalid_idxs = np.load(f, allow_pickle=False)

    # remove invalid indices
    for i in reversed(np.sort(invalid_idxs)):
        acts = np.delete(acts, i)
        obss = np.delete(obss, i, axis=0)
        rews = np.delete(rews, i)
        trms = np.delete(trms, i)

    # record games
    is_first = True
    is_done = True
    for a, o, r, t in zip(acts, obss, rews, trms):
        # initialize array of game data
        if is_done:
            actions = []
            states = []
            rewards = []
            is_done = False
        # save step data in buffer
        actions.append(int(a))
        states.append(o.tolist())
        rewards.append(float(r))
        # when an episode is over, the buffer is saved.
        if t == 1:
            is_done = True
            if is_first:
                # dispose the first game
                # (because it starts in the middle of game ...)
                is_first = False
            else:
                # to numpy
                states = np.array(states, dtype=np.uint8)
                actions = np.array(actions)
                # get step number in this game
                num_steps = len(states)
                # add return-to-go (rtg) rewards
                rtgs = np.zeros(num_steps, dtype=int)
                rtgs[-1] = rewards[-1]
                for i in reversed(range(num_steps - 1)):
                    rtgs[i] = rtgs[i] + rtgs[i + 1]
                # add timesteps
                timesteps = list(range(num_steps))
                timesteps = np.array(timesteps)
                # reshape (if not fitted, fill with zeros)
                row_num = math.ceil(timesteps.shape[0] / step_size)
                states.resize(row_num, step_size, 84 * 84)
                actions.resize(row_num, step_size)
                rtgs.resize(row_num, step_size)
                timesteps.resize(row_num, step_size)
                # stack_size - 1 frames are added in each states
                append_states = states[:,-(stack_size - 1):,:]                       # (row_num,3,84*84)
                append_states = append_states[:-1,:,:]                               # (row_num-1,3,84*84)
                first_state = np.expand_dims(states[0,0,:], axis=0)                  # (1,84*84)
                first_state = np.repeat(first_state, stack_size - 1, axis=0)         # (3,84*84)
                first_state = np.expand_dims(first_state, axis=0)                    # (1,3,84*84)
                append_states = np.concatenate((first_state, append_states), axis=0) # (row_num,3,84*84)
                states = np.concatenate((append_states, states), axis=1)             # (row_num,step_size+3,84*84)
                # reshaped as image format
                states = states.reshape(row_num, (step_size+stack_size-1) * 84, 84)
                # save
                for i in range(row_num):
                    # save state as single file for each state for memory efficiency.
                    np.save(f"{dest_dir}/states/{cur_row}.npy", states[i])

                    # add meta data for states
                    states_meta_chunk.append(cur_row)
                    # save others in chunk (meta data)
                    actions_chunk.append(actions[i])
                    rtgs_chunk.append(rtgs[i])
                    timesteps_chunk.append(timesteps[i])
                    # if row reaches chunk limit, save chunk (as meta data)
                    cur_row += 1
                    if cur_row % chunk_size == 0:
                        with open(f"{dest_dir}/meta{cur_chunk}.pkl","wb") as f:
                            pickle.dump({
                                "actions": np.stack(actions_chunk, axis=0),
                                "rtgs": np.stack(rtgs_chunk, axis=0),
                                "timesteps": np.stack(timesteps_chunk, axis=0),
                                "states_meta": np.stack(states_meta_chunk, axis=0)
                            }, f)
                        cur_chunk += 1
                        actions_chunk = []
                        rtgs_chunk = []
                        timesteps_chunk = []
                        states_meta_chunk = []
                print(f"Processed packs:{buffer_num + 1}/50 rows:{cur_row}", end="\r")

# last (remaining) chunk
if len(actions_chunk) != 0:
    with open(f"{dest_dir}/meta{cur_chunk}.pkl","wb") as f:
        pickle.dump({
            "actions": np.stack(actions_chunk, axis=0),
            "rtgs": np.stack(rtgs_chunk, axis=0),
            "timesteps": np.stack(timesteps_chunk, axis=0),
            "states_meta": np.stack(states_meta_chunk, axis=0),
        }, f)

print("")
print(f"Saved {cur_row} rows in {dest_dir}")

print("")
print("Shuffle data and remove bias...")
process_data(dest_dir, step_size, chunk_size)
print("Done.")