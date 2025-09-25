# ResT: Reshaping Token-Level Policy Gradients for Tool-Use Large Language Models

## 🚀 Overview

**ResT** (Reward Shaping with Token-level entropy) is a novel reinforcement learning framework that establishes a theoretical and empirical link between policy entropy and training stability for tool-use alignment. Built upon the [VERL] framework and inspired by [ToolRL], our method introduces an entropy-aware, token-level reward shaping mechanism with curriculum learning to significantly improve the training stability and performance of language models on complex tool-calling tasks.

### Key Features

- **Entropy-Based Stability Analysis**: Demonstrates that lower average entropy correlates with reduced variance in policy-gradient updates
- **Token-Level Reward Shaping**: Dynamically weights different token categories based on their semantic importance and entropy characteristics
- **Curriculum Learning Integration**: Gradually increases reasoning token weights as training progresses and entropy decreases
- **State-of-the-Art Performance**: Achieves up to 8.76% improvement over existing approaches and outperforms GPT-4o on tool-use benchmarks

## 📊 Performance Highlights

- **Multi-turn tool-use base tasks**: +1.50% over GPT-4o when fine-tuned on Qwen3-4B-2507
- **Single-turn tool-use tasks**: +4.11% over GPT-4o 
- **Curriculum vs. Static Weighting**: +4.86% improvement with our curriculum-based approach
- **Overall Benchmark Improvement**: Up to 8.76% over existing state-of-the-art methods

## 🏗️ Architecture

Our ResT framework consists of three main components:

1. **Entropy-Stability Analysis Module**: Establishes the theoretical foundation linking policy entropy to training stability
2. **Token-Level Weight Assignment**: Main Core algorithm
3. **Curriculum Learning Scheduler**: Dynamically adjusts token weights during training progression

### Token Weighting Algorithm

The core implementation assigns weights to different token categories:

- **Tool Names**: Moderate weight (γ_name)
- **Tool Arguments**: Highest weight (γ_args)  
- **Format Tags**: Moderate weight (γ_format)
- **Reasoning Content**: Adaptive weight based on training step (γ_think)


## 🛠️ Installation

### Prerequisites

- Python 3.10
- CUDA 12.4.0 (for GPU acceleration)
- PyTorch 2.6.0

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/1229095296/ResT_Tool_use_LLM.git

```

2. **Install dependencies**:
```bash
# Install core dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

3. **Additional dependencies** (optional):
```bash
# For VLLM support (recommended for inference)
pip install vllm==0.8.5

```

## 🚀 Quick Start

### Basic Training with ResT

1. **Prepare your data** in the required format (see `data/rlla_4k/` for examples)

2. **Configure training parameters**:
```bash
export EXPERIMENT_NAME="qwen3-1.7B"
export SCHEDULEREWARD=1       # Enable dynamic reward scheduling
export model_path="/path/to/huggingface.co/Qwen/Qwen3-1.7B"
```

3. **Launch ResT training**:
```bash
# Use the provided ResT training script
bash train_ResT.sh
```


### Advanced Configuration

#### Key Training Parameters

The ResT training script includes several important environment variables for controlling the algorithm behavior:

```bash
export SCHEDULEREWARD=1       # Enable dynamic reward scheduling
```


#### Multi-GPU Configuration

For large-scale training, configure distributed settings:

```bash
# Multi-GPU setup in train_ResT.sh
trainer.n_gpus_per_node=8        # GPUs per node
trainer.nnodes=1                 # Number of nodes
actor_rollout_ref.rollout.tensor_model_parallel_size=8  # Model parallelism
actor_rollout_ref.rollout.gpu_memory_utilization=0.5   # GPU memory usage
```


## 🎯 Training Examples

### Example : Qwen3-1.7B with ResT (Recommended)
```bash
# Main ResT training with ResT
bash train_ResT.sh
```


## 📈 Evaluation

### API-Bank Benchmark

Evaluate your trained model on the API-Bank benchmark:

```bash
cd API-Bank
python generate.py --model_paths /path/to/your/trained/model
python evaluate.py --model_paths /path/to/your/trained/model
```

### BFCL V3 Benchmark

For implementation, please refer to the official BFCL code.

## 🔬 Technical Details

### Theoretical Foundation

Our approach is grounded in the observation that:

1. **Entropy-Stability Correlation**: Lower average entropy in policy distributions correlates with reduced variance in policy-gradient updates
2. **Structured Token Importance**: Tool names and parameters (low-entropy, structured tokens) are primary determinants of reward outcomes
3. **Curriculum Learning Benefits**: Gradual increase in reasoning token weights leads to more stable convergence

### Implementation Highlights

- **Dynamic Weight Adjustment**: Token weights adapt based on training step and content characteristics
- **Multi-Region Processing**: Separate handling for `<think>`, `<response>`, and `<tool_call>` regions
- **Normalization Strategy**: Ensures mean weight equals 1 across valid positions for training stability


---

**ResT** - ResT: Reshaping Token-Level Policy Gradients for Tool-Use Large Language Models 🎯
