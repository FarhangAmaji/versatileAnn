# project utility

## How to run in Docker

windows Run docker-desktop, other system skip this step.
in project folder in cli:
docker build -t ann .

* windows cli:

docker run -it --rm -v %cd%:/app ann

* other systems:

docker run --rm -it -v ${PWD}:/app ann
docker run --rm -it -v $(pwd):/app ann


