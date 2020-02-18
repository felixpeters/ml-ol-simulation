build-image:
	docker build -t felixpeters/ai-sim .

run-image:build-image
	docker run --rm -it --name ai-sim-runner -v `pwd`/data:/ai-sim/data felixpeters/ai-sim:latest /bin/bash -c "/bin/bash run.sh"
