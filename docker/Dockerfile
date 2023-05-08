FROM python:3.9.5-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends wget curl build-essential libreadline-dev \ 
libncursesw5-dev libssl-dev libsqlite3-dev libgdbm-dev libbz2-dev liblzma-dev zlib1g-dev uuid-dev libffi-dev libdb-dev \
cmake git

#サイズ削減のため不要なものは削除
RUN apt-get autoremove -y

WORKDIR /app

COPY ["./app/pyproject.toml", "/app/"]

# poetry install
ENV PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.3.2 \
    PYTHONUNBUFFERED=1

ENV POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false

ENV PATH=$POETRY_HOME/bin:$PATH

RUN curl -sSL https://install.python-poetry.org/ | python3 -
RUN poetry install

# install eigen
RUN apt install -y libeigen3-dev libgoogle-glog-dev libgflags-dev
#RUN apt install -y libboost-dev libboost1.55-dev libboost1.55-tools-dev

# # install PCL
# RUN git clone --recursive https://github.com/PointCloudLibrary/pcl \
#     && cd pcl \
#     && mkdir build \
#     && cd build \
#     && cmake -DCMAKE_BUILD_TYPE=Release .. \
#     && make \
#     && make install
RUN apt install -y unzip
ENV OPENCV_VERSION=4.5.0
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    rm ${OPENCV_VERSION}.zip && \
    mv opencv-${OPENCV_VERSION} opencv && \
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    rm ${OPENCV_VERSION}.zip && \
    mv opencv_contrib-${OPENCV_VERSION} opencv_contrib

RUN cd opencv && mkdir build && cd build && \
    cmake \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D BUILD_SHARED_LIBS=OFF \
    -D ENABLE_CXX11=ON \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_opencv_app=OFF \
    -D BUILD_opencv_java_bindings_generator=OFF \
    -D BUILD_opencv_python_bindings_generator=OFF \
    -D BUILD_opencv_python_tests=OFF \
    -D BUILD_opencv_ts=OFF \
    -D BUILD_opencv_js=OFF \ 
    -D BUILD_opencv_datasets=OFF \
    -D BUILD_opencv_dnn_objdetect=OFF \
    -D BUILD_opencv_dnn_superres=OFF \
    -D BUILD_opencv_gapi=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    .. && make -j8 && make install

RUN git clone https://github.com/boostorg/boost.git \
    && cd boost \
    && git checkout boost-1.72.0 \
    && git submodule update --init \
    && ./bootstrap.sh \
    && ./b2 toolset=gcc-9 --prefix=/usr/local -j5 \
    && ./b2 install toolset=gcc-9 --prefix=/usr/local -j5
# install cmake
# RUN wget https://cmake.org/files/v3.4/cmake-3.4.0-rc3.tar.gz \
#     && tar xvf cmake-3.4.0-rc3.tar.gz \
#     && cd cmake-3.4.0-rc3/ \
#     && ./configure \
#     && make \
#     && make install \
#     && export PATH="/usr/local/bin:$PATH"

# install g2opy
#RUN pip install g2o-python==0.0.11

# RUN apt update \
#     && apt-get install -y --no-install-recommends openssh-client \
#     && mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com > ~/.ssh/known_hosts
# RUN --mount=type=ssh git clone --recursive git@github.com:miquelmassot/g2o-python.git \
#     && cd g2o-python \
#     && pip install -U --user -v .

