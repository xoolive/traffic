# Docker containers

## `jupyter.Dockerfile` (stable)

This Docker image provides a Jupyter environment running with the latest release of traffic, installed with Anaconda.

```sh
# Build image
docker buildx build -f jupyter.Dockerfile -t traffic/jupyter:latest .
# Run container
docker run -it -p 8888:8888 traffic/jupyter:latest
```

Then connect to http://localhost:8888/lab

## `poetry.Dockerfile` (dev)

This Docker image provides a full development environment which you can connect from Visual Code.

- default mode: bind your local copy to the container;
- contained mode: clone a copy of the repository inside the container (uncomment lines at the end of the file);

After the first execution, run `poetry install` at the root of the project and select the freshly created Python version in Visual Code.

```sh
# Build image
docker buildx build -f docker/poetry.Dockerfile -t traffic/poetry:latest .
# Run container (default mode)
docker run --name traffic/poetry -v $(pwd):/home/user/traffic -it traffic/poetry
```

There is a `PROXY` argument you can use when building the image, with the option
`--build-arg PROXY=http://proxy.corporate:80`

## Proxy issues

Read https://docs.docker.com/network/proxy/
