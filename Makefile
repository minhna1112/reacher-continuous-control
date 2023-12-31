IMAGE_NAME="udacity_drl/reacher_contiuous_controller"
TAG="v0.0.1"

build:
	docker build -t ${IMAGE_NAME}:${TAG} .

run:
	docker run --rm --gpus all -p 8888:8888 -v $(pwd):/workspace ${IMAGE_NAME}:${TAG}

all: build
ifeq ($(RUN),1)
	@$(MAKE) run
endif
