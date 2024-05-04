## Introduction
This is a project to train a model to play Atari games(`Pong` in the example) using Decision Transformer Reinforcement Learning.

## Code Reference
The code is based on the following repositories:  
official repository: https://github.com/kzl/decision-transformer  
minGPT: https://github.com/karpathy/minGPT/tree/master/mingpt  
decision-transformer-tsmatz: https://github.com/tsmatz/decision-transformer  

## How to run
### Requirements
```bash
$ sudo apt-get update
$ sudo apt install -y gcc
$ sudo apt-get install -y make
```

Install `gsutil`
```bash
# sudo install
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli

# pip install
pip install gsutil
```

Set a vertual environment
And download these packages
```bash
pip install torch numpy matplotlib opencv-python atari_py
```

Install pre-trained data
```bash
sudo apt-get install unrar
wget http://www.atarimania.com/roms/Roms.rar
unrar x -r Roms.rar
python3 -m atari_py.import_roms ROMS
```

```bash
mkdir downloaded_game_data
gsutil -m cp -R gs://atari-replay-datasets/dqn/Pong downloaded_game_data
mkdir game_dataset
```

### Run the code
In this example, we loaded 5/50 files of the dataset.
Totol 50 files required 1.5TB storage.
```bash
# get and process the dataset, it takes about 1 hour(Intel 14700)
python3 ./data/create_dataset.py
```

```bash
# train and evaluate the model
# you can add the some arguments(see the run_dt_atari.py) with the command
python3 ./run_dt_atari.py
```