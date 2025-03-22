FROM ubuntu:latest

VOLUME ["/app"]
WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    htop \
    g++ gdb vim meson pkg-config clang-format clang-tidy \
    libfmt-dev libspdlog-dev libgtest-dev pybind11-dev \
    python3 python3-loguru python3-numpy python3-pybind11

# Expose the port on which the API will listen
EXPOSE 8080
