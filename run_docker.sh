#!/bin/bash
#usage bash run_docker.sh /data gpu

dataset_path=$1
run_type=$2
version=v1.1.62

if [ -z "$dataset_path" ];then
  echo "Please specify the dataset path"
  exit
fi
dataset_path=$(readlink -f "$dataset_path")

echo "Docker version is ${version}"
echo "Dataset path is $(readlink -f "$dataset_path")"

open_explorer_path=$(readlink -f "$(dirname "$0")")
echo "OpenExplorer package path is $open_explorer_path"

if [ "$run_type" == "cpu" ];then
    echo "Start Docker container in CPU mode."
    docker run -it --rm \
      -v "$open_explorer_path":/open_explorer \
      -v "$dataset_path":/data/horizon_j5/data \
      openexplorer/ai_toolchain_ubuntu_20_j5_cpu:"$version"
else
    echo "Start Docker container in GPU mode."
    docker run -it --rm \
      --gpus all \
      --privileged \
      --shm-size="15g" \
      -v "$open_explorer_path":/open_explorer \
      -v "$dataset_path":/data/horizon_j5/data \
      openexplorer/ai_toolchain_ubuntu_20_j5_gpu:"$version"
fi
