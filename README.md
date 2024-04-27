## How to run
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

Run the code
```bash
# get and process the data
python3 create_dataset.py
```