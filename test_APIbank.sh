
export WORLD_SIZE=8
source activate
cd /path/to/project/API-Bank
model_path=/path/to/project/checkpoints/
python /path/to/project/API-Bank/generate.py --model_paths $model_path
python /path/to/project/API-Bank/evaluate.py --model_paths $model_path 
