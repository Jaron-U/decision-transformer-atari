## Introduction
This is a project to train a model to play Atari games(`Pong` in the example) using Decision Transformer Reinforcement Learning.

## Code Reference
The code is based on the following repositories:  
official repository: https://github.com/kzl/decision-transformer/tree/  
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
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli
```

Set a vertual environment
And download these packages
```bash
pip3 install torch numpy matplotlib opencv-python atari_py
```

Install pre-trained data
```bash
sudo apt-get install unrar
wget http://www.atarimania.com/roms/Roms.rar
unrar x -r Roms.rar
python3 -m atari_py.import_roms ROMS
```
###
Run the code
```bash
# get and process the dataset
python3 ./data/create_dataset.py
```