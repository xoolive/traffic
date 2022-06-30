FROM python:3.10-slim

# If you need a proxy for your environment
ARG PROXY
ENV http_proxy=$PROXY
ENV https_proxy=$PROXY
RUN if [ -z "$PROXY" ] ; then echo "Acquire::http::Proxy \"$PROXY\";" | tee /etc/apt/apt.conf.d/01proxy ; fi

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

# Install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
ENV PATH="${PATH}:/home/user/.poetry/bin"

CMD [ "bash" ]

# If you prefer to clone and install traffic in the container (rather than bind
# the volume, uncomment the following commands)

# Clone and install traffic
# RUN git config --global http.proxy ${PROXY}
# RUN git clone https://github.com/xoolive/traffic

# WORKDIR /home/user/traffic
# RUN poetry install

# By default, open a Python terminal
# CMD [ "poetry", "run", "python" ]