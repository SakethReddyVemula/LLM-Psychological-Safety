#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --output=llama3_cosafe_responses_%j.out
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL

# Activate your virtual environment
source /home2/saketh.vemula/ltg_venv/bin/activate

# Set cache directory for Hugging Face
export HF_HOME="/home2/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/home2/$USER/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="/home2/$USER/.cache/huggingface/datasets"

# Create cache directories if they don't exist
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

# Directory for output files
OUTPUT_DIR="./results/cosafe_responses"
mkdir -p $OUTPUT_DIR

echo "Starting CoSAFE response generation with Llama 3 at $(date)"

# Run the Python script
python generate_responses.py \
    --input_file ~/CoSafe-Dataset/CoSafe datasets/self_harm.json \
    --output_file $OUTPUT_DIR/llama3_responses.jsonl \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --batch_size 1 \
    --num_samples 1000  # Remove or adjust this parameter if you want to process the entire dataset

echo "Response generation completed at $(date)"