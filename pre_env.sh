conda create -n verl python==3.10
conda activate verl
# Make sure you have activated verl conda env
# If you need to run with megatron
#bash scripts/install_vllm_sglang_mcore.sh
# Or if you simply need to run with FSDP
USE_MEGATRON=0 bash /path/to/project/github.com/ORGANIZATION/PROJECT.git/scripts/install_vllm_sglang_mcore.sh
cd /path/to/project/github.com/ORGANIZATION/PROJECT.git
pip install --no-deps -e .
conda deactive