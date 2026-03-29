#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 20:00:00
source ~/.bashrc
mamba activate adl-asymmetric-lm-duos

pip install -e .

python tools/evidential_ft.py --config_base configs/arc_c_qwen2_7b/map_arc_c_qwen2_7b.yaml --config_sidekick configs/arc_c_qwen2_15b/map_arc_c_qwen2_15b.yaml -w workdirs/ib_edl/arc_c/ -n arc_c_qwen2_7b_arc_c_qwen2_15b --seed 1
python tools/evidential_ft.py --config_base configs/arc_c_qwen2_7b/map_arc_c_qwen2_7b.yaml --config_sidekick configs/arc_c_qwen2_15b/map_arc_c_qwen2_15b.yaml -w workdirs/ib_edl/arc_c/ -n arc_c_qwen2_7b_arc_c_qwen2_15b --seed 2
python tools/evidential_ft.py --config_base configs/arc_c_qwen2_7b/map_arc_c_qwen2_7b.yaml --config_sidekick configs/arc_c_qwen2_15b/map_arc_c_qwen2_15b.yaml -w workdirs/ib_edl/arc_c/ -n arc_c_qwen2_7b_arc_c_qwen2_15b --seed 3
python tools/evidential_ft.py --config_base configs/arc_c_qwen2_7b/map_arc_c_qwen2_7b.yaml --config_sidekick configs/arc_c_qwen2_15b/map_arc_c_qwen2_15b.yaml -w workdirs/ib_edl/arc_c/ -n arc_c_qwen2_7b_arc_c_qwen2_15b --seed 4
python tools/evidential_ft.py --config_base configs/arc_c_qwen2_7b/map_arc_c_qwen2_7b.yaml --config_sidekick configs/arc_c_qwen2_15b/map_arc_c_qwen2_15b.yaml -w workdirs/ib_edl/arc_c/ -n arc_c_qwen2_7b_arc_c_qwen2_15b --seed 5
