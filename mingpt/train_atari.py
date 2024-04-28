"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import numpy as np
import torch
import random
import atari_py
from collections import deque
import cv2

class Env():
    def __init__(self, game, seed, device, stack_size):
        self.device = device
        self.stack_size = stack_size # Number of frames to concatenate
        max_episode_length = 108e3  # to avoid infinite loop

        self.ale = atari_py.ALEInterface()
        self.ale.setInt("random_seed", seed)
        self.ale.setInt("max_num_frames_per_episode", max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(game.lower()))

        actions = self.ale.getMinimalActionSet()
        self.actions = {i: actions[i] for i in range(len(actions))}
        
        self.state_buffer = deque([], maxlen=stack_size)
    
    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        state = torch.tensor(state, device=self.device, dtype=torch.float32).div_(255)
        return state

    def _reset_buffer(self):
        for _ in range(self.stack_size):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))
    
    def reset(self):
        self._reset_buffer()

        self.ale.reset_game()
        # Perform up to 30 random no-ops before starting
        for _ in range(random.randrange(30)):
            self.ale.act(0)  # Assumes raw action 0 is always no-op
            if self.ale.game_over():
                self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), dim=0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(dim=0)[0] # Max pool over last 2 frames
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), dim=0), reward, done



