#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
source ~/.bashrc
mamba activate adl-asymmetric-lm-duos

pip install -e .

python tools/average_metrics.py --config_base configs/arc_c_qwen2_7b/ib_arc_c_qwen2_7b.yaml --config_sidekick configs/arc_c_qwen2_15b/ib_arc_c_qwen2_15b.yaml -w workdirs/ib_edl_qwen/arc_c/ --dataset_name arc_c
python tools/average_metrics.py --config_base configs/csqa_qwen2_7b/ib_csqa_qwen2_7b.yaml --config_sidekick configs/csqa_qwen2_15b/ib_csqa_qwen2_15b.yaml -w workdirs/ib_edl_qwen/csqa/ --dataset_name csqa
python tools/average_metrics.py --config_base configs/obqa_qwen2_7b/ib_obqa_qwen2_7b.yaml --config_sidekick configs/obqa_qwen2_15b/ib_obqa_qwen2_15b.yaml -w workdirs/ib_edl_qwen/obqa/ --dataset_name obqa
