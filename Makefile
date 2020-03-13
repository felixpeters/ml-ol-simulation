build-image:
	docker build -t felixpeters/ml-ol-sim .

run-image:build-image
	docker run --rm -it --name ai-sim-runner -v `pwd`/data:/ai-sim/data felixpeters/ml-ol-sim:latest /bin/bash -c "/bin/bash run.sh $(RUN_SCRIPT)"
