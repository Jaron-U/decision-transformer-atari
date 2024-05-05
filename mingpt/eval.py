import torch
from torch import nn
from torch.nn import functional as F
from mingpt.train_atari import Env
import numpy as np
from mingpt.train_atari import TrainConfig

@torch.no_grad()
def eval_game(trainConfig: TrainConfig, model):
    target_rtg = trainConfig.target_rtg
    seed = trainConfig.seed
    game_name = trainConfig.game_name
    device = trainConfig.device
    stack_size = trainConfig.stack_size
    max_timesteps = trainConfig.max_timesteps
    step_size = trainConfig.step_size

    env = Env(game_name, seed, device, stack_size)

    model.eval()
    
    total_reward = []
    for i in range(10):
        state = env.reset()
        state = state.type(torch.float32).to(device).unsqueeze(0)
        rtgs = [target_rtg]

        states_para = state.unsqueeze(0)
        rtgs_para = torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1).type(torch.float32)
        timesteps_para = torch.zeros((1, 1), dtype=torch.int64).to(device)
        # get the action
        sampled_action = model.select_action(states = states_para, actions = None, 
                                             rtgs = rtgs_para, timesteps = timesteps_para)

        all_states = state
        time_start = 0
        actions = []
        rewards = 0
        while True:
            action = sampled_action.cpu().numpy()[0,-1]
            actions.append(sampled_action)
            next_state, reward, done = env.step(action)
            rewards += reward

            if done:
                total_reward.append(rewards)
                break
            
            next_state = next_state.unsqueeze(0).to(device)
            all_states = torch.cat([all_states, next_state], dim=0)
            rtgs += [rtgs[-1] - reward]

            # get last step_size size in sequence
            for t in range(time_start, len(all_states), step_size):
                batch_states = all_states.unsqueeze(0)[:, t:t+step_size]
                batch_actions = torch.tensor(actions+[0], dtype=torch.long).to(device).unsqueeze(0)[:, t:t+step_size]
                batch_rtgs = torch.tensor(rtgs, dtype=torch.long).to(
                    device).unsqueeze(0).unsqueeze(-1)[:, t:t+step_size].type(torch.float32)
            time_start = t

            batch_timestep = torch.arange(min(t, max_timesteps), min(t, max_timesteps) + 
                                          len(batch_states)).unsqueeze(0).to(device)
            # get the next action
            sampled_action = model.select_action(states = batch_states, actions = batch_actions, 
                                                 rtgs = batch_rtgs, timesteps = batch_timestep)
            
        print(f"Round {i+1}/10 reward: {rewards}", end="\r")
        
    total_returns = np.sum(total_reward)
    eval_return = np.mean(total_reward)
    print(f"Total return of 10 round: {total_returns}, Average return: {eval_return}")




