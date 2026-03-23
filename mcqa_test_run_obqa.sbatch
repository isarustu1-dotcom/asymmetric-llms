#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
source ~/.bashrc
mamba activate adl-asymmetric-lm-duos

pip install -e .

python tools/evidential_ft.py --config_base configs/obqa_qwen2_7b/map_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_05b/map_obqa_qwen2_05b.yaml -w workdirs/ib_edl/obqa/ -n obqa_qwen2_7b_obqa_qwen2_05b --seed 1
python tools/evidential_ft.py --config_base configs/obqa_qwen2_7b/map_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_05b/map_obqa_qwen2_05b.yaml -w workdirs/ib_edl/obqa/ -n obqa_qwen2_7b_obqa_qwen2_05b --seed 2
python tools/evidential_ft.py --config_base configs/obqa_qwen2_7b/map_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_05b/map_obqa_qwen2_05b.yaml -w workdirs/ib_edl/obqa/ -n obqa_qwen2_7b_obqa_qwen2_05b --seed 3
python tools/evidential_ft.py --config_base configs/obqa_qwen2_7b/map_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_05b/map_obqa_qwen2_05b.yaml -w workdirs/ib_edl/obqa/ -n obqa_qwen2_7b_obqa_qwen2_05b --seed 4
python tools/evidential_ft.py --config_base configs/obqa_qwen2_7b/map_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_05b/map_obqa_qwen2_05b.yaml -w workdirs/ib_edl/obqa/ -n obqa_qwen2_7b_obqa_qwen2_05b --seed 5