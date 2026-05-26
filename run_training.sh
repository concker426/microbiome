#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2
export TORCH_FLASH_ATTN_ENABLED=0
export TF_CPP_MIN_LOG_LEVEL=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /hd/liujx/microbiome_llm_project
/hd/liujx/ProCyon/.venv/bin/python3 run_microbiome_nl_7b.py > training_output.log 2>&1
