FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 iptables iputils-ping && apt-get clean \
 && pip install --no-cache-dir httpx==0.28.1
WORKDIR /app
