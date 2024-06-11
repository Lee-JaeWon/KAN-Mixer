#!/bin/bash
# Author : Jaewon Lee (https://github.com/Lee-JaeWon)

# Set the project directory (PROJECT_DIR) as the parent directory of the current working directory
PROJECT_DIR=$(dirname "$PWD")

# Move to the parent folder of the project directory
cd "$PROJECT_DIR"

# Check if arguments are provided for the image name and tag
if [ "$#" -ne 2 ]; then
  echo "[Error] Usage: $0 <container_name> <image_name:tag>"
  exit 1
fi

# Print the current working directory to verify the change
echo "Current working directory: $PROJECT_DIR"

# Assign the arguments to variables for clarity
CONTAINER_NAME="$1"
IMAGE_NAME="$2"

# Launch the nvidia-docker container with the provided image name and tag
docker run --privileged -it \
           -e NVIDIA_DRIVER_CAPABILITIES=all \
           -e NVIDIA_VISIBLE_DEVICES=all \
           --volume="$PROJECT_DIR:/root/workspace/src" \
           --volume=/mnt/hdd3/jaewon:/root/workspace/src/dataset \
           --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
           --net=host \
           --ipc=host \
           --shm-size=2gb \
           --name="$CONTAINER_NAME" \
           --env="DISPLAY=$DISPLAY" \
           --gpus=all \
           "$IMAGE_NAME" /bin/bash


# 서버 hdd3:/mnt/hdd3