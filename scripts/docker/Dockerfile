# Dockerfile for serving a PI policy.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM tck-xinan-registry.cn-chengdu.cr.aliyuncs.com/wair/openpi_server:base

WORKDIR /workspace/openpi

COPY ./docs /workspace/openpi/docs
COPY ./examples /workspace/openpi/examples
COPY ./packages /workspace/openpi/packages
COPY ./scripts /workspace/openpi/scripts
COPY ./src /workspace/openpi/src
COPY ./third_party /workspace/openpi/third_party
COPY ./pyproject.toml /workspace/openpi/pyproject.toml
COPY ./uv.lock /workspace/openpi/uv.lock

CMD ["bash", "entrypoint.sh"]