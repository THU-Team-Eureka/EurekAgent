#!/usr/bin/env bash
# docker/setup-daemon-proxy.sh
#
# Sync Docker daemon proxy from the current shell environment.
# The daemon does not read ~/.bashrc, so its proxy must be configured
# separately via systemd. This script bridges the gap.
#
# Usage: sudo bash docker/setup-daemon-proxy.sh

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: This script must be run with sudo." >&2
    echo "Usage: sudo bash docker/setup-daemon-proxy.sh" >&2
    exit 1
fi

PROXY_HTTP="${HTTP_PROXY:-${http_proxy:-}}"
PROXY_HTTPS="${HTTPS_PROXY:-${https_proxy:-}}"
PROXY_NO="${NO_PROXY:-${no_proxy:-}}"

if [ -z "$PROXY_HTTPS" ]; then
    echo "No proxy found in environment (HTTPS_PROXY). Nothing to configure."
    exit 0
fi

mkdir -p /etc/docker
cat > /etc/docker/proxy.env << EOF
HTTP_PROXY=${PROXY_HTTP}
HTTPS_PROXY=${PROXY_HTTPS}
NO_PROXY=${PROXY_NO:-localhost,127.0.0.1}
EOF

mkdir -p /etc/systemd/system/docker.service.d
cat > /etc/systemd/system/docker.service.d/proxy.conf << EOF
[Service]
EnvironmentFile=/etc/docker/proxy.env
EOF

# Remove legacy inline Environment= proxy lines from other override files
for f in /etc/systemd/system/docker.service.d/*.conf; do
    [ -f "$f" ] || continue
    [ "$f" = "/etc/systemd/system/docker.service.d/proxy.conf" ] && continue
    if grep -q 'Environment=.*[Pp][Rr][Oo][Xx][Yy]' "$f" 2>/dev/null; then
        sed -i '/Environment=.*[Pp][Rr][Oo][Xx][Yy]/d' "$f"
    fi
done

systemctl daemon-reload
systemctl restart docker
echo "Docker daemon proxy set to $PROXY_HTTPS"
