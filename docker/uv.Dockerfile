FROM python:3.13-slim

RUN apt update && apt install -y libgdal-dev libgeos-dev libproj-dev proj-bin proj-data libarchive-dev sqlite3 git curl cmake g++

# Create a user
RUN useradd -ms /bin/bash user
USER user
WORKDIR /home/user/

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
