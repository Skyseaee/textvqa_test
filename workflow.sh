#!/bin/bash

usage() {
    cat << EOF
Usage: All options are needed!
    -d: device_id (e.g., 0 or 0,1 or 2,3 or 0,1,2,3)
    -t: tp (e.g., 1, 2, 4...)
    -m: model_path
    -b: backend (e.g., lmdeploy/vllm/tensorrt-llm)
    -s: service port (default: 8080)
    -n: parallel num (default: 50)
    -k: tasks list, split by ',' (default: textvqa)
EOF
}

# Default values
service_port=8080
backend="lmdeploy"
tp=1
parallel=50
tasks="textvqa"  # 初始化 tasks 变量

# Parse command-line arguments
while getopts "d:t:m:b:s:n:k:h" opt; do  # 修改 getopts 选项列表
    case $opt in
        d) device_id="$OPTARG" ;;
        t) tp="$OPTARG" ;;
        m) model_path="$OPTARG" ;;
        b) backend="$OPTARG" ;;
        s) service_port="$OPTARG" ;;
        n) parallel="$OPTARG" ;;
        k) tasks="$OPTARG";;
        h) usage; exit 0 ;;
        *) echo "Invalid option: -$OPTARG"; usage; exit 1 ;;
    esac
done

# Check if all required variables are set
if [[ -z "$device_id" || -z "$model_path" || -z "$backend" ]]; then
    echo "Error: Missing required arguments."
    usage
    exit 1
fi

# Variables
LMDEPLOY_IMAGE_TAG="harbor.shopeemobile.com/aip/shopee-mlp-aip-llm-generater-lmdeploy:0.5.3-4a8b6d06"
CONTAINER_NAME="llvm-evaluation-test"
service_name="0.0.0.0"

declare -g lmdeploy_pid=""
# Function to start the model server
function open_model_server() {
    echo "Starting model server..."
    if [[ $backend == "lmdeploy" ]]; then
        if command -v lmdeploy >/dev/null 2>&1; then
            lmdeploy serve api_server \
                ${model_path} \
                --server-name ${service_name} \
                --server-port ${service_port} \
                --tp ${tp} \
                --max-batch-size 512 \
                --cache-max-entry-count 0.9 \
                --session-len 8192 &
            lmdeploy_pid=$!
        else
            docker run -d --gpus all --env CUDA_VISIBLE_DEVICES=$device_id \
                --privileged --shm-size=10g --ipc=host \
                -v ${model_path}:/workspace/models/${model_path} \
                -p ${service_port}:${service_port} \
                --name=${CONTAINER_NAME} ${LMDEPLOY_IMAGE_TAG} \
                lmdeploy serve api_server \
                    /workspace/models/${model_path} \
                    --server-name ${service_name} \
                    --server-port ${service_port} \
                    --tp ${tp} \
                    --max-batch-size 512 \
                    --cache-max-entry-count 0.9 \
                    --session-len 8192
        fi
    else
        echo "Invalid backend specified: $backend"
        exit 1
    fi

    sleep 3m
    echo "Waiting for server to start..."
}

# Function to stop the model server
function close_model_server() {
    echo "Stopping model server..."
    if command -v lmdeploy >/dev/null 2>&1; then
        # pkill -f "lmdeploy serve api_server"
        kill -9 "$lmdeploy_pid"
    else
        container_id=$(docker ps -a | grep ${CONTAINER_NAME} | awk '{print $1 }')
        if [[ -n "$container_id" ]]; then
            docker stop ${container_id}
            docker rm ${container_id}
        else
            echo "No container found with name ${CONTAINER_NAME}"
        fi
    fi
}

# Start model server
open_model_server

# Install the env
pip3 install -r requirements-min.txt
pip3 install -U shopee-aip-datasets -i https://pypi.shopee.io/
if ! python3 -c "import llava" &> /dev/null; then
    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA
    pip3 install -e .
    cd ..
fi

export AIP_TOKEN=FcbgizlexopCBjrwsFuEjqhnfklyoFdr

python3 __main__.py \
    --model-path $model_path \
    --task-name  $tasks \
    --api-address http://$service_name:$service_port

# Stop model server
close_model_server