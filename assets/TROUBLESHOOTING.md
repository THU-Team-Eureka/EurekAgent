# Docker Troubleshooting

Solutions for the most common Docker setup problems when running EurekAgent —
pulling images behind a proxy, using registry mirrors, and offline image
transfer. For general usage questions, see the
[Useful Tips](../README.md#useful-tips) section of the README.

## `docker pull` fails

Pulling images from Docker Hub requires outbound internet access.
If `docker pull` fails with a network or proxy error, try the following.

**Proxy setup:** If your network requires a proxy, set `HTTP_PROXY` and
`HTTPS_PROXY` in the current shell, then sync it to the Docker daemon:

```bash
# In your current shell:
export HTTP_PROXY=http://127.0.0.1:<YOUR_PROXY_PORT>
export HTTPS_PROXY=http://127.0.0.1:<YOUR_PROXY_PORT>
export http_proxy=$HTTP_PROXY
export https_proxy=$HTTPS_PROXY

# Ensure local services (such as grader) are not proxied
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

# In the same shell, sync to Docker daemon (requires sudo):
sudo bash docker/setup-daemon-proxy.sh

# Then retry:
docker pull node:22-bookworm
```

**Registry mirror:** If direct access to Docker Hub is unreliable, add
registry mirrors to `/etc/docker/daemon.json` (requires sudo). Merge the
`registry-mirrors` key into the existing JSON if the file already has
content:

```bash
# Edit /etc/docker/daemon.json and add the "registry-mirrors" key, e.g. (requires sudo):
{
    "registry-mirrors": [
        "https://docker.1ms.run",
        "https://dockerproxy.link",
        "https://docker.m.daocloud.io"
    ]
}

sudo systemctl restart docker
docker pull node:22-bookworm
```

**Offline transfer:** If the server has no internet access or you lack
sudo, you can transfer the image from another machine:

```bash
# On a machine with sudo, docker, and outbound internet access:
# IMPORTANT: configure your docker daemon proxy as described above if needed

# Pull and save the image
docker pull node:22-bookworm
docker image save node:22-bookworm -o node-22-bookworm.tar

# Transfer the tar file to the target server (scp, USB, etc.)
scp node-22-bookworm.tar user@server:~

# On the target server:
docker image load -i node-22-bookworm.tar

# Then build the project image:
bash docker/build.sh
```

See Docker docs for
[`docker image save`](https://docs.docker.com/reference/cli/docker/image/save/) and
[`docker image load`](https://docs.docker.com/reference/cli/docker/image/load/).
