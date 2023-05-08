

FROM ubuntu:20.04

ENV https_proxy "http://proxy-sen.noc.sony.co.jp:10080"
ENV http_proxy "http://proxy-sen.noc.sony.co.jp:10080"

RUN apt update \
    && apt install -y curl \
    && apt install -y python3.8 

# コンテナにアクセスした際のデフォルトディレクトリ
WORKDIR /app

COPY ["./app/pyproject.toml", "/app/"]

# poetryのインストール先の指定
ENV POETRY_HOME=/opt/poetry
# poetryインストール
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 -
ENV PATH="/opt/poetry/bin:$PATH" 
RUN poetry config virtualenvs.create false \
    && poetry install

# install cmake
# RUN wget https://cmake.org/files/v3.4/cmake-3.4.0-rc3.tar.gz \
#     && tar xvf cmake-3.4.0-rc3.tar.gz \
#     && cd cmake-3.4.0-rc3/ \
#     && ./configure \
#     && make \
#     && make install \
#     && export PATH="/usr/local/bin:$PATH"

# install g2opy
RUN apt update \
    && apt install -y python3-pip 
    #&& pip install g2o-python==0.0.11

# RUN apt update \
#     && apt-get install -y --no-install-recommends openssh-client \
#     && mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com > ~/.ssh/known_hosts
# RUN --mount=type=ssh git clone --recursive git@github.com:miquelmassot/g2o-python.git \
#     && cd g2o-python \
#     && pip install -U --user -v .

