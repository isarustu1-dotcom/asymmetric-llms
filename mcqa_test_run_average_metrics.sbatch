#!/bin/bash
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --qos=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
source ~/.bashrc
mamba activate adl-asymmetric-lm-duos

pip install -e .

python tools/average_metrics.py --config_base configs/arc_c_qwen2_7b/map_arc_c_qwen2_7b.yaml --config_sidekick configs/arc_c_qwen2_15b/map_arc_c_qwen2_15b.yaml -w workdirs/ib_edl/arc_c/ --dataset_name arc_c
python tools/average_metrics.py --config_base configs/csqa_qwen2_7b/map_csqa_qwen2_7b.yaml --config_sidekick configs/csqa_qwen2_15b/map_csqa_qwen2_15b.yaml -w workdirs/ib_edl/csqa/ --dataset_name csqa
python tools/average_metrics.py --config_base configs/obqa_qwen2_7b/map_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/map_obqa_qwen2_15b.yaml -w workdirs/ib_edl/obqa/ --dataset_name obqa