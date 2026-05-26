#!/bin/bash
# Launcher for NL microbiome training
export PYTHONPATH="/hd/liujx/ProCyon/.venv/lib/python3.11/site-packages:$PYTHONPATH"
export PATH="/hd/liujx/ProCyon/.venv/bin:$PATH"
export TORCH_FLASH_ATTN_ENABLED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run with torchrun for FSDP multi-GPU
/hd/liujx/ProCyon/.venv/bin/torchrun \
    --nproc_per_node=3 \
    --master_port=29501 \
    /hd/liujx/microbiome_llm_project/run_microbiome_nl_7b.py
