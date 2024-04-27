import io
import math
import argparse
import numpy as np
import torch
import gzip
import pickle
from data.process_data import get_max_timesteps, process_data
from data.load_data import AtariPongDataset





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--stack_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--chunk_size', type=int, default=10000)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--game', type=str, default='Pong')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dest_dir', type=str, default='game_dataset')
    args = parser.parse_args()

    scr_dir = f"downloaded_game_data/{args.game}/1/replay_logs"

    # get max timesteps
    max_timesteps = get_max_timesteps(args.dest_dir)
    print(f"max_timesteps: {max_timesteps}")

    # load dataset
    dataset = AtariPongDataset(args.dest_dir, 3, args.step_size, args.stack_size)
    
    