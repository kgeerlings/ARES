# ARES

## Description

The ARES (Aerial Recon Evasion System) project aims to design a simulated environment for testing and training autonomous drone strategies in aerial reconnaissance missions. The project name refers both to the drone's capabilities and to the Greek god of war, highlighting the tactical and strategic aspects of the system.

ARES is based on reinforcement learning to enable an agent to navigate autonomously in a dynamic and potentially hostile environment, to accomplish a reconnaissance mission while avoiding or confronting enemy drones.

## Prerequisites

- Python 3.12
- Linux (The environment parallelisation doesn't work on Windows)

## Installation

To install the project, first clone the repository:
```bash
git clone https://github.com/kgeerlings/ARES.git
cd ARES
```

Then, create your virtual environment:
```bash
python3 -m venv ares_env
source ares_env/bin/activate
```

Finally, install the project requirements, and the project library itself.
```bash
pip install -r requirements.txt
pip install -e .
```

## Command-lines options

- `--type` Type of operation to perform: train, finetune, or eval.
- `--config` Configuration model to use (0, 1, 2, or 3). Corresponds to config, config_model_1, config_model_2, or config_model_3.
- `--checkpoint-path` Path to checkpoint file (for eval/finetune).
- `--num-episodes` Number of episodes for evaluation.
- `--max-steps` Maximum steps per episode.
- `--record` Record videos during evaluation.
- `--save-video` Save videos during evaluation.
- `render` Render environment during evaluation.


## Test the project

If you want to test the differents models with evaluation mode, enter these commands:
```bash
python3 ares/main/main.py --type eval --checkpoint-path ares/models/1_ally_go_to_target.pt --config 1
```
(for the first model)

```bash
python3 ares/main/main.py --type eval --checkpoint-path ares/models/2_ally_go_to_target_and_return_to_base.pt --config 2
```
(for the second model)

```bash
python3 ares/main/main.py --type eval --checkpoint-path ares/models/3.2_ally_dodges_enemies.pt --config 3
```
(for the third model)

If you want to test the training:
```bash
python3 ares/main/main.py --type train
```
(you can modify the hyperparameters in config/config.json)

If you want to test the finetuning:
```bash
python3 ares/main/main.py --type finetune
```
(you can modify the hyperparameters in config/config.json)

For the training and the finetuning, you can run this command in an other terminal to see the tensorboard logs:
```bash
tensorboard --logdir=runs
```