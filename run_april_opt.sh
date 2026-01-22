#!/bin/bash
# source ./activate_env.sh
python main_opt.py --iterations 800 \
                   --lora_rank 0 \
                   --lr_opt 0.1 \
                   --use_layernorm \
                   --use_residual \
                   --dataset mnist \
                   --batch_size 1 \
                   --alpha 0.01
