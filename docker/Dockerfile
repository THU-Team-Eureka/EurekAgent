ARG BASE_IMAGE=node:22-bookworm
FROM ${BASE_IMAGE}

RUN apt-get update \
    -o Acquire::ForceIPv4=true \
    -o Acquire::Retries=5 \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        fonts-liberation \
        libasound2 \
        libatk-bridge2.0-0 \
        libatk1.0-0 \
        libcairo2 \
        libcups2 \
        libdbus-1-3 \
        libdrm2 \
        libgbm1 \
        libglib2.0-0 \
        libgtk-3-0 \
        libnspr4 \
        libnss3 \
        libpango-1.0-0 \
        libu2f-udev \
        libvulkan1 \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxkbcommon0 \
        libxrandr2 \
        python3 \
        python3-numpy \
        python3-pip \
        python3-venv \
        tmux \
        xdg-utils \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_CLAUDE_CODE=0
RUN if [ "$INSTALL_CLAUDE_CODE" = "1" ]; then \
        npm install -g @anthropic-ai/claude-code; \
    fi

RUN pip3 install --break-system-packages uv
