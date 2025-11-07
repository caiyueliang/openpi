# Dockerfile for serving a PI policy.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM tck-xinan-registry.cn-chengdu.cr.aliyuncs.com/wair/openpi_server:base

ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_HTTP_TIMEOUT=120
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

WORKDIR /workspace/openpi

COPY ./.cache /root/.cache
COPY ./docs /workspace/openpi/docs
COPY ./examples /workspace/openpi/examples
COPY ./images /workspace/openpi/images
COPY ./packages /workspace/openpi/packages
COPY ./scripts /workspace/openpi/scripts
COPY ./src /workspace/openpi/src
COPY ./third_party /workspace/openpi/third_party
COPY ./.pre-commit-config.yaml /workspace/openpi/.pre-commit-config.yaml
COPY ./.python-version /workspace/openpi/.python-version
COPY ./CONTRIBUTING.md /workspace/openpi/CONTRIBUTING.md
COPY ./LICENSE /workspace/openpi/LICENSE
COPY ./LICENSE_GEMMA.txt /workspace/openpi/LICENSE_GEMMA.txt
COPY ./pyproject.toml /workspace/openpi/pyproject.toml
COPY ./README.md /workspace/openpi/README.md
COPY ./uv.lock /workspace/openpi/uv.lock

# RUN uv add fastapi uvicorn --frozen -i https://pypi.tuna.tsinghua.edu.cn/simple/ 

COPY ./entrypoint.sh /workspace/openpi/entrypoint.sh

CMD ["bash", "entrypoint.sh"]