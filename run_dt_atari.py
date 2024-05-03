import io
import math
import torch
import gzip
import pickle
import random
import argparse
import numpy as np
from data.process_data import get_max_timesteps, process_data
from data.load_data import AtariPongDataset
from mingpt.model_atari import GPT, GPTConfig, Embeddings_Atari
from mingpt.train_atari import TrainConfig, Trainer
from mingpt.eval import eval_game
import atari_py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--stack_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--chunk_size', type=int, default=10000)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--game', type=str, default='Pong')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dest_dir', type=str, default='game_dataset')
    args = parser.parse_args()

    scr_dir = f"downloaded_game_data/{args.game}/1/replay_logs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    game_name = args.game.lower()
    ale = atari_py.ALEInterface()
    ale.loadROM(atari_py.get_game_path(game_name))
    ale.reset_game()
    legal_actions = ale.getMinimalActionSet()
    num_actions = len(legal_actions) # 6

    # get max timesteps
    max_timesteps = get_max_timesteps(args.dest_dir)

    # load dataset
    dataset = AtariPongDataset(args.dest_dir, args.batch_size, args.step_size, args.stack_size)

    gptConfig = GPTConfig(step_size=args.step_size, max_timestep=max_timesteps, 
                          vocab_size=num_actions, n_head=8, n_layer=6, n_embd=128)
    
    model = GPT(gptConfig).to(device)

    # different game has different target_rtg
    trainConfig = TrainConfig(target_rtg = 20.0, seed = seed, 
                              game_name = game_name, model = model, 
                              device=device, stack_size=args.stack_size, 
                              max_timesteps=max_timesteps, step_size=args.step_size,
                              weight_decay=0.1, epochs=args.epochs,
                              learning_rate=args.learning_rate, betas=(0.9, 0.999),
                              batch_size=args.batch_size, lr_decay=False)
       
    # train the model
    trainer = Trainer(trainConfig, dataset)
    losses = trainer.train_game()

    # # save the model
    # torch.save(model.state_dict(), f"{game_name}_model.pth")

    # model.load_state_dict(torch.load(f"{game_name}_model.pth"))
    # evaluate the model
    eval_game(trainConfig, model)

    
    