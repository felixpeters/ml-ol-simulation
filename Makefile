build-image:
	docker build -t felixpeters/ai-sim .

run-image:
	docker run --rm -it --name ai-sim-runner -v `pwd`:$(HOME_DIR) felixpeters/ai-sim:latest '/bin/bash'
