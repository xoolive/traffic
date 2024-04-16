FROM python:3.12-slim

RUN apt update && apt install -y libgdal-dev libgeos-dev libproj-dev proj-bin proj-data libarchive-dev sqlite3 git curl cmake g++

# Create a user
RUN useradd -ms /bin/bash user
USER user
WORKDIR /home/user/

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/home/user/.local/bin"
