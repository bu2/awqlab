# Typical setup for a GPU instance (8x AMD EPYC 7413 + 96GB RAM + NVIDIA L40S 48GB)
# https://www.scaleway.com/en/l40s-gpu-instance/

# Install dev dependencies
export NEEDRESTART_MODE=a  # avoid "restart services" prompt from apt
sudo apt-get update && apt-get upgrade -y
sudo apt-get install -y byobu cuda-toolkit git nvtop vim

# Download and install Python (Miniforge)
mkdir dl
cd dl
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
cd ..

# Install Python with Miniforge
bash dl/Miniforge3-Linux-x86_64.sh -b -s
echo 'PATH=~/miniforge3/bin:$PATH' >miniforge-env.sh
source miniforge-env.sh

# Scaleway specific: set the Hugginface cache directory in to the /scratch ephemeral partition
sudo mkdir /scratch/huggingface
sudo chown -R ubuntu:ubuntu /scratch/huggingface
cd ~/.cache/
ln -s /scratch/huggingface ./
cd

# Clone awqlab with submodules from GitHub
# (you would need to set git+ssh authentication on your Github account)
git clone --recursive https://github.com/bu2/awqlab.git
cd awqlab
pip install -r requirements.txt

# Set HuggingFace token
export HUGGING_FACE_HUB_TOKEN="<TOKEN>"

# Install AutoAWQ
cd AutoAWQ
pip install AutoAWQ_kernels
cd ..
pip install .
## Quantize LLama 2 with AWQ GEMM and GEMV kernels
# python examples/quantize_llama2_7b_chat_awq_gemm.py
# python examples/quantize_llama2_7b_chat_awq_gemv.py
## Text generation
# python examples/generate_llama2_7b_chat_awq_gemm.py
# python examples/generate_llama2_7b_chat_awq_gemv.py
## Benchmark
# python examples/benchmark.py --model_path blehyaric/llama-2-7b-chat-hf-awq-gemm --batch_size 32
cd ..

# Install QUICK
cd QUICK
pip install -e .
## Quantize LLama 2 with AWQ QUICK kernel
# python examples/basic_quant.py --model_path meta-llama/Llama-2-7b-chat-hf --quant_path ../models/llama-2-7b-chat-hf-awq-quick
## Text generation
# python examples/generate_llama2_7b_chat_awq_quick.py
## Benchmark
# python examples/benchmark.py --model_path blehyaric/llama-2-7b-chat-hf-awq-quick --batch_size 32
cd ..

cd tinyawq
pip install tinyawq/
python generate_llama2_7b_chat_tinyawq.py
