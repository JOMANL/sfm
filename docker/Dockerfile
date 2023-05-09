FROM python:3.9.8

ENV https_proxy "http://proxy-sen.noc.sony.co.jp:10080"
ENV http_proxy "http://proxy-sen.noc.sony.co.jp:10080"

RUN apt update \
    && apt install -y curl

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