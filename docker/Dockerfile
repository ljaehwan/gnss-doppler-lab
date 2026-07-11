FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

ARG GPS_SDR_SIM_COMMIT=28ca29a6719475195e3aabd5930c4ed02d67190f

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/osqzss/gps-sdr-sim.git /opt/gps-sdr-sim \
    && git -C /opt/gps-sdr-sim checkout --detach "${GPS_SDR_SIM_COMMIT}" \
    && make -C /opt/gps-sdr-sim \
    && install -m 0755 /opt/gps-sdr-sim/gps-sdr-sim /usr/local/bin/gps-sdr-sim

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests

RUN pip install --upgrade pip \
    && pip install -e ".[dev]"

CMD ["bash"]
