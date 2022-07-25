FROM python:3.10-slim

RUN apt update && apt install -y libgdal-dev libgeos-dev libproj-dev proj-bin proj-data libarchive-dev sqlite3 git curl cmake g++

# Install latest version of libproj
WORKDIR /root
RUN git clone https://github.com/OSGeo/PROJ

WORKDIR /root/PROJ
RUN git checkout 9.0
RUN mkdir build

WORKDIR /root/PROJ/build
RUN cmake ..; cmake --build . -j4
RUN cmake --build . --target install

# Create a user
RUN useradd -ms /bin/bash user
USER user
WORKDIR /home/user/

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="${PATH}:/home/user/.poetry/bin"

RUN poetry config virtualenvs.create false
