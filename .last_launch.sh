#!/bin/bash
docker run \
    -d --name vllm-fn-tp1 \
    --gpus all --network host --ipc host \
    --cap-add SYS_NICE --cap-add SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
    --memory 99g --memory-swap 99g \
    --log-opt max-size=50m --log-opt max-file=3 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e VLLM_PLE_CPU_OFFLOAD=1 \
    -e VLLM_PLE_PACKED_TABLE_DIR=/root/.cache/vllm/ple_cache/Mia-AiLab--Qwen3.8-Flash-Next-NVFP4 \
    -e VLLM_PLE_OFFLOAD_STEP_TIMEOUT=300 \
    -e MAX_JOBS=2 \
    -e FLASHINFER_NVCC_THREADS=1 \
     \
     \
     \
    -v /home/bharat/Projects/qwen3.8-flash-next-spark/files/draft_vocab_en_code_47k.txt:/root/draft_vocab.txt:ro \
    -e VLLM_MTP_DRAFT_VOCAB=/root/draft_vocab.txt \
     \
    -e HF_HOME=/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    -v /home/bharat/Projects/qwen3.8-flash-next-spark/files/ple_layer_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
    -v /home/bharat/Projects/qwen3.8-flash-next-spark/files/modelopt_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py:ro \
    -v /home/bharat/Projects/qwen3.8-flash-next-spark/files/qsa_ops_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py:ro \
    -v /home/bharat/Projects/qwen3.8-flash-next-spark/files/qsa_nvidia_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/qsa.py:ro \
    -v /home/bharat/Projects/qwen3.8-flash-next-spark/files/mtp_patched.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/mtp.py:ro \
    -v /home/bharat/Projects/qwen3.8-flash-next-spark/files/ple_offload/ple_offload_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/ple_offload_layer.py:ro \
    -v /home/bharat/Projects/qwen3.8-flash-next-spark/files/ple_offload/connector.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/connector.py:ro \
    -v /home/bharat/Projects/qwen3.8-flash-next-spark/files/ple_offload/worker.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py:ro \
    -v /home/bharat/Projects/qwen3.8-flash-next-spark/files/ple_offload/protocol.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/protocol.py:ro \
    -v /home/bharat/.cache/huggingface:/root/.cache/huggingface \
    -v /home/bharat/.cache/vllm:/root/.cache/vllm \
     \
    vllm/vllm-openai:qwen38-flash-next \
    Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 \
    --enable-prompt-tokens-details --served-model-name qwen3.8-flash-next --tensor-parallel-size 1 --gpu-memory-utilization 0.780 --max-num-seqs 4 --max-num-batched-tokens 2048 --max-model-len 262144 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 --load-format safetensors --safetensors-load-strategy lazy --enable-chunked-prefill --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --distributed-executor-backend mp --speculative-config '{"method":"mtp","num_speculative_tokens":3,"use_local_argmax_reduction":true}' --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16]}' --default-chat-template-kwargs '{"enable_thinking":false}' --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":20}' \
    --host 0.0.0.0 \
    --port 8888 \
     \
