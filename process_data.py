import os
import pickle
import numpy as np

game_name = "Pong"
source_data_dir = f"downloaded_game_data/{game_name}/1/replay_logs"
dest_dir = "game_dataset"
step_size = 50
chunk_size = 10000  # meta data will be chunked by each 10000 rows
stack_size = 4      # frame stacking size (So stack_size - 1 frames are added in each states)

# In the original data (the downloaded data), the latter part tends to have high rtgs. 
# (i.e, the original data has bias in order.) For this reason, we now shuffle all the data and remove bias.

# load all data into a single numpy array
def process_data():
    all_data = None
    for chunk_num in range(1000):
        # file check
        if not os.path.exists(f"{dest_dir}/meta{chunk_num}.pkl"):
            break
        # read chunk
        with open(f"{dest_dir}/meta{chunk_num}.pkl", "rb") as f:
            chunk = pickle.load(f)
        # append
        rtgs = chunk["rtgs"]
        states_meta = chunk["states_meta"]
        actions = chunk["actions"]
        timesteps = chunk["timesteps"]
        c = np.c_[
            rtgs.reshape(len(rtgs), -1),
            states_meta.reshape(len(states_meta), -1),
            actions.reshape(len(actions), -1),
            timesteps.reshape(len(timesteps), -1),
        ]
        if all_data is None:
            all_data = c
        else:
            all_data = np.concatenate((all_data, c), axis=0)

    # shuffle all data
    np.random.shuffle(all_data)
    col = 0
    all_rtgs = all_data[:,col:col+step_size].reshape((-1, rtgs.shape[1]))
    col += step_size
    all_states_meta = all_data[:,col]
    col += 1
    all_actions = all_data[:,col:col+step_size].reshape((-1, actions.shape[1]))
    col += step_size
    all_timesteps = all_data[:,col:col+step_size].reshape((-1, timesteps.shape[1]))

    # overwrite by new data
    for i, start in enumerate(range(0, len(all_states_meta), chunk_size)):
        with open(f"{dest_dir}/meta{i}.pkl","wb") as f:
            pickle.dump({
                "actions": all_actions[start:start+chunk_size,:],
                "rtgs": all_rtgs[start:start+chunk_size,:],
                "timesteps": all_timesteps[start:start+chunk_size,:],
                "states_meta": all_states_meta[start:start+chunk_size]
            }, f)

def get_max_timesteps():
    max_timesteps = 0
    for chunk_num in range(1000):
        # file check
        if not os.path.exists(f"{dest_dir}/meta{chunk_num}.pkl"):
            break
        # read chunk
        with open(f"{dest_dir}/meta{chunk_num}.pkl", "rb") as f:
            chunk = pickle.load(f)
        max_timesteps = max(np.max(chunk["timesteps"]), max_timesteps)
    return max_timesteps
