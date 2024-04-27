from torch.utils.data import IterableDataset
import torch
import numpy as np
import os
import pickle
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# convert the np array data to torch tensor

# get state array (step_size+stack_size-1) h, w) by file idx
def get_state_from_index(dest_dir, step_size, stack_size, idx):
    arr = np.load(f"{dest_dir}/states/{idx}.npy")
    arr = arr.reshape(step_size + stack_size - 1, 84, 84)
    return arr

# array of single frame (batch, step_size+stack_size-1, h, w)
#     to frame stacking (batch, step_size, stack_size, h, w)
def toFrameStack(states, stack_size):
    new_states = states
    new_states = [np.roll(new_states, i, axis=1) for i in reversed(range(stack_size))]
    new_states = np.stack(new_states, axis=2)
    new_states = new_states[:,stack_size - 1:,:,:]
    return new_states

class AtariPongDataset(IterableDataset):
    def __init__(self, files_dir, batch_size, step_size, stack_size):
        super(AtariPongDataset).__init__()

        self.files_dir = files_dir
        self.step_size = step_size
        self.batch_size = batch_size
        self.stack_size = stack_size
        
        self.max_chunk = 1000

    def __len__(self):
        length = 0
        for chunk_num in range(self.max_chunk):
            # file check
            if not os.path.exists(f"{self.files_dir}/meta{chunk_num}.pkl"):
                break
            # read chunk
            with open(f"{self.files_dir}/meta{chunk_num}.pkl", "rb") as f:
                chunk = pickle.load(f)
            states_meta = chunk["states_meta"]
            # add batch count
            length += math.ceil(len(states_meta) / self.batch_size)
        return length

    def __iter__(self):
        for chunk_num in range(self.max_chunk):
            # file check
            if not os.path.exists(f"{self.files_dir}/meta{chunk_num}.pkl"):
                break
            # read chunk
            with open(f"{self.files_dir}/meta{chunk_num}.pkl", "rb") as f:
                chunk = pickle.load(f)
            rtgs = chunk["rtgs"]
            states_meta = chunk["states_meta"]
            actions = chunk["actions"]
            timesteps = chunk["timesteps"]
            # shuffle in chunk
            c = np.c_[
                rtgs.reshape(len(rtgs), -1),
                states_meta.reshape(len(states_meta), -1),
                actions.reshape(len(actions), -1),
                timesteps.reshape(len(timesteps), -1),
            ]
            np.random.shuffle(c)
            col = 0
            rtgs = c[:,col:col+self.step_size].reshape(rtgs.shape)
            col += self.step_size
            states_meta = c[:,col]
            col += 1
            actions = c[:,col:col+self.step_size].reshape(actions.shape)
            col += self.step_size
            timesteps = c[:,col:col+self.step_size].reshape(timesteps.shape)
            # process batch
            for i in range(0, len(states_meta), self.batch_size):
                # get rtgs, actions, timesteps
                rtgs_batch = rtgs[i:i+self.batch_size,:]
                actions_batch = actions[i:i+self.batch_size,:]
                timesteps_batch = timesteps[i:i+self.batch_size,:]
                # get states
                states_batch = [get_state_from_index(self.files_dir, self.step_size, self.stack_size, j) 
                                for j in states_meta[i:i+self.batch_size]]
                states_batch = np.stack(states_batch, axis=0)
                states_batch = toFrameStack(states_batch, self.stack_size)
                # transform
                rtgs_batch = np.expand_dims(rtgs_batch.astype(float), axis=-1)
                states_batch = states_batch.astype(float) / 255.0
                # yield return
                yield torch.tensor(rtgs_batch, dtype=torch.float32).to(device), torch.tensor(states_batch, dtype=torch.float32).to(device), torch.tensor(actions_batch, dtype=torch.int64).to(device), torch.tensor(timesteps_batch, dtype=torch.int64).to(device)