FROM python:3.9.8

ENV https_proxy "http://proxy-sen.noc.sony.co.jp:10080"
ENV http_proxy "http://proxy-sen.noc.sony.co.jp:10080"

RUN apt update \
    && apt install -y curl libgl1-mesa-dev cmake

WORKDIR /app
COPY ["./app/pyproject.toml", "/app/"]

# install poetry to python environment
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 -
ENV PATH="/opt/poetry/bin:$PATH" 
RUN poetry config virtualenvs.create false \
    && poetry install

# jupyter to use extension (especially hinterland)
# RUN jupyter contrib nbextension install –user \
#     && jupyter nbextensions_configurator enable --user

# g2o eigen
# dependencies to g2o
RUN apt update \
    && apt install -y libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5

RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz \
    && tar xvf eigen-3.4.0.tar.gz \
    && cd eigen-3.4.0 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make check \
    && make install

RUN git clone https://github.com/RainerKuemmerle/g2o.git \
    && cd g2o \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make \
    && make install



