# Minimal Dockerfile to run Lean 4 & Lake and Python scripts
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y curl git build-essential python3 python3-pip ca-certificates locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
# Install elan (Lean toolchain manager) non-interactively
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
ENV PATH="/root/.elan/bin:${PATH}"
# Confirm lean installed
RUN lean --version || true
RUN pip3 install --no-cache-dir -U pip
WORKDIR /workspace
# Container entrypoint default
CMD ["/bin/bash"]
