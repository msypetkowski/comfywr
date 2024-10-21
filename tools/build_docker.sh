
# Variables for container and image names
BASE_IMAGE="comfywr_intermediate:latest"
TARGET_IMAGE="comfywr:latest"

docker build -t $BASE_IMAGE -f docker/comfywr/Dockerfile .

CONTAINER_NAME=tmp_comfywr_name
docker run --gpus all --name $CONTAINER_NAME -it $BASE_IMAGE /bin/bash -c "cd /install_dir/ && python install.py"
sleep 2
docker commit $CONTAINER_NAME $TARGET_IMAGE
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo "Build completed. New image created: $TARGET_IMAGE"
