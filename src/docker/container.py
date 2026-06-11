"""Docker container lifecycle manager for isolated agent execution."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import hashlib
from pathlib import Path
from typing import Any
import tomllib
from urllib.parse import urlsplit, urlunsplit

from ..gpu_policy import log_gpu_policy_warnings, resolve_gpu_policy
from ..workspace_setup import write_workspace_permissions, install_workspace_hooks

log = logging.getLogger(__name__)

# Env vars to resolve from host env or ~/.claude/settings.json "env" block
_API_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
)
_PROXY_ENV_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)
_FORWARDED_ENV_VARS = _API_ENV_VARS + _PROXY_ENV_VARS

_LOCAL_PROXY_HOSTS = {"localhost", "127.0.0.1", "::1"}
_DOCKER_HOST_ALIAS = "host.docker.internal"


class DockerContainer:
    """Manages a single Docker container for one pipeline run."""

    def __init__(
        self,
        *,
        image: str = "eureka-agent:node22-bookworm",
        network: str = "host",
        gpus: str = "auto",
    ) -> None:
        self._image = image
        self._network = network
        self._gpus = gpus
        self._container_id: str | None = None
        self._bootstrap_python_env = False
        self._python_env_key = ""

    @property
    def container_id(self) -> str | None:
        return self._container_id

    def start(self, workspace_dir: Path, secure_eval_env: dict[str, str]) -> str:
        """Start container with mounts. Returns container ID."""
        self._ensure_image()
        agent_home = self._ensure_agent_home(workspace_dir)
        env_vars = self._resolve_env_vars()
        self._deploy_proxy_script(workspace_dir)
        docker_args = self._build_run_args(workspace_dir, env_vars, agent_home, secure_eval_env)

        try:
            result = subprocess.run(
                ["docker", *docker_args],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(_format_docker_start_error(exc)) from exc
        self._container_id = result.stdout.strip()
        log.info("Started container %s (image=%s)", self._container_id[:12], self._image)
        if self._bootstrap_python_env:
            self._ensure_container_python_env()
        self._verify_claude_cli()

        # Write settings.local.json and install hooks.
        write_workspace_permissions(workspace_dir, hook_prefix="/workspace")
        install_workspace_hooks(workspace_dir, hook_prefix="/workspace")

        return self._container_id

    def start_grader(
        self,
        *,
        workspace_dir: Path,
        hidden_eval_dir: Path,
        host_port: int,
        token: str,
    ) -> str:
        """Start the secure grader server in a separate Docker container."""
        self._ensure_image()
        grader_home = self._ensure_grader_home(workspace_dir)
        env_vars = self._resolve_env_vars(_PROXY_ENV_VARS)
        self._deploy_proxy_script(workspace_dir)
        repo_src = Path(__file__).resolve().parents[1]
        docker_args = self._build_run_args(
            workspace_dir, env_vars, grader_home, secure_eval_env={},
            extra_mounts=[
                (hidden_eval_dir.resolve(), "/hidden_eval", "ro"),
                (repo_src, "/controller_src/src", "ro"),
            ],
            ports=[(host_port, host_port)],
            extra_env={"PYTHONPATH": "/controller_src"},
            include_agent_mounts=False,
        )
        try:
            result = subprocess.run(
                ["docker", *docker_args],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(_format_docker_start_error(exc)) from exc
        self._container_id = result.stdout.strip()
        log.info(
            "Started grader container %s (image=%s, port=%s)",
            self._container_id[:12], self._image, host_port,
        )
        if self._bootstrap_python_env:
            self._ensure_container_python_env()
        self._start_grader_process(host_port=host_port, token=token)
        return self._container_id

    @staticmethod
    def _ensure_agent_home(workspace_dir: Path) -> Path:
        """Create a UID-owned writable HOME overlay inside the workspace.

        Docker auto-creates bind-mount targets as root, which leaves $HOME
        unwritable for our non-root container user and breaks Claude's shell
        tool ("Bash tool is failing due to an initialization issue"). We
        pre-create a host-side directory owned by the invoking user and mount
        it over $HOME so the agent can persist shell state there.

        We pre-create $HOME/.claude so the later read-only sub-mount of
        .claude/skills does not cause Docker to auto-create that directory as
        root, which would block the Bash tool from writing session-env.

        We also seed a per-run copy of the host's ~/.claude.json into this
        overlay. Claude Code's Skill tool writes session state back into that
        file; mounting the host copy read-only causes every Skill invocation
        to EROFS. The copy keeps the agent's config private to the run.
        """
        agent_home = (workspace_dir / ".agent_home").resolve()
        (agent_home / ".claude").mkdir(parents=True, exist_ok=True)
        (agent_home / ".cache").mkdir(parents=True, exist_ok=True)
        # Session persistence directory — writable so --resume works after
        # SIGTERM. Without this, Claude Code cannot save conversation state
        # and "No conversation found" errors on resume.
        (agent_home / ".claude" / "projects").mkdir(parents=True, exist_ok=True)

        host_config = Path.home() / ".claude.json"
        run_config = agent_home / ".claude.json"
        if host_config.is_file():
            shutil.copy2(host_config, run_config)
            # Register /workspace as a trusted project so Claude's interactive
            # mode skips the workspace-trust dialog inside the container.
            _inject_workspace_trust(run_config)

        host_gitconfig = Path.home() / ".gitconfig"
        if host_gitconfig.is_file():
            shutil.copy2(host_gitconfig, agent_home / ".gitconfig")

        return agent_home

    @staticmethod
    def _ensure_grader_home(workspace_dir: Path) -> Path:
        """Create a minimal writable HOME for the grader container."""
        grader_home = (workspace_dir / ".grader_home").resolve()
        (grader_home / ".cache").mkdir(parents=True, exist_ok=True)
        return grader_home

    @staticmethod
    def _deploy_proxy_script(workspace_dir: Path) -> None:
        """Copy the PTY proxy script into the workspace for container access."""
        proxy_src = Path(__file__).resolve().parent / "pty_proxy.py"
        proxy_dir = workspace_dir / ".eureka_internal"
        proxy_dir.mkdir(parents=True, exist_ok=True)
        proxy_dst = proxy_dir / "pty_proxy.py"
        if not proxy_dst.exists() or proxy_src.stat().st_mtime > proxy_dst.stat().st_mtime:
            shutil.copy2(proxy_src, proxy_dst)

    def stop(self) -> None:
        """Kill and remove the container."""
        if not self._container_id:
            return
        try:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True, text=True, check=False,
            )
            log.info("Stopped container %s", self._container_id[:12])
        except Exception as e:
            log.warning("Failed to stop container: %s", e)
        self._container_id = None

    def exec_command(self, cmd: list[str]) -> list[str]:
        """Prefix a command with docker exec."""
        if not self._container_id:
            raise RuntimeError("Container not started")
        return ["docker", "exec", self._container_id, *cmd]

    def is_running(self) -> bool:
        if not self._container_id:
            return False
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self._container_id],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() == "true"

    def _verify_claude_cli(self) -> None:
        """Fail early if the container cannot execute a Linux Claude CLI."""
        if not self._container_id:
            return
        result = subprocess.run(
            self.exec_command(["sh", "-lc", "command -v claude && claude --version"]),
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            return

        details = _clean_subprocess_output(result.stderr or result.stdout)
        if "exec format error" in details:
            hint = (
                "The Docker container picked up a host Claude binary that cannot run "
                "inside Linux. On macOS, rebuild the image so it contains the Linux "
                "Claude CLI: bash docker/build.sh"
            )
        else:
            hint = (
                "The Docker image must provide a Linux Claude CLI. Rebuild it with "
                "bash docker/build.sh, or install claude inside the image."
            )
        raise RuntimeError(
            "Docker container cannot execute `claude`.\n"
            f"{hint}\n"
            f"Command output:\n{details}"
        )

    def _ensure_container_python_env(self) -> None:
        """Create/repair the mounted Docker Python environment."""
        if not self._container_id:
            return
        expected_key = self._python_env_key
        requirements = self._container_python_requirements()
        requirements_json = json.dumps(requirements)
        script = (
            "set -eu; "
            "marker=/workspace/.venv/.eureka_env_key; "
            "ok=0; "
            "if [ -x /workspace/.venv/bin/python3 ] && [ -f \"$marker\" ]; then "
            f"if [ \"$(cat \"$marker\")\" = {json.dumps(expected_key)} ]; then "
            "if /workspace/.venv/bin/python3 - <<'PY' >/dev/null 2>&1\n"
            "import sys\n"
            "raise SystemExit(0 if sys.platform.startswith('linux') else 1)\n"
            "PY\n"
            "then ok=1; fi; "
            "fi; "
            "fi; "
            "if [ \"$ok\" != 1 ]; then "
            "find /workspace/.venv -mindepth 1 -maxdepth 1 -exec rm -rf {} +; "
            "python3 -m venv --system-site-packages /workspace/.venv; "
            "/workspace/.venv/bin/python3 - <<'PY'\n"
            "import json, shutil, subprocess, sys, time\n"
            f"requirements = json.loads({requirements_json!r})\n"
            "if requirements:\n"
            "    if shutil.which('uv'):\n"
            "        command = ['uv', 'pip', 'install', '--python', sys.executable, *requirements]\n"
            "    else:\n"
            "        command = [sys.executable, '-m', 'pip', 'install', *requirements]\n"
            "    for attempt in range(1, 4):\n"
            "        try:\n"
            "            subprocess.check_call(command)\n"
            "            break\n"
            "        except subprocess.CalledProcessError as exc:\n"
            "            if attempt == 3:\n"
            "                raise\n"
            "            wait_seconds = 2 * attempt\n"
            "            print(\n"
            "                f'Container dependency install failed '\n"
            "                f'(attempt {attempt}/3, exit {exc.returncode}); '\n"
            "                f'retrying in {wait_seconds}s...',\n"
            "                file=sys.stderr,\n"
            "                flush=True,\n"
            "            )\n"
            "            time.sleep(wait_seconds)\n"
            "PY\n"
            f"printf '%s' {json.dumps(expected_key)} > \"$marker\"; "
            "fi; "
            "/workspace/.venv/bin/python3 - <<'PY'\n"
            "import sys, pathlib\n"
            "assert sys.platform.startswith('linux'), sys.platform\n"
            "print(sys.executable)\n"
            "pathlib.Path('/workspace/.venv/.eureka_python').write_text(sys.version + '\\n')\n"
            "PY"
        )
        result = subprocess.run(
            self.exec_command(["sh", "-lc", script]),
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            return
        details = _clean_subprocess_output(result.stderr or result.stdout)
        if "exec format error" in details:
            hint = (
                "The mounted Docker Python environment contains binaries for the "
                "wrong platform. Remove .eureka_docker/venvs and rerun."
            )
        else:
            hint = "The Docker image must include python3-venv, or the venv cache must be writable."
        raise RuntimeError(
            "Docker container cannot initialize `/workspace/.venv`.\n"
            f"{hint}\n"
            f"Command output:\n{details}"
        )

    def _start_grader_process(self, *, host_port: int, token: str) -> None:
        """Start the grader server inside an already-running grader container."""
        if not self._container_id:
            return
        log_path = "$HOME/grader_server.log"
        cmd = (
            "set -eu; "
            "export PYTHONPATH=/controller_src; "
            "export CUDA_VISIBLE_DEVICES=''; "
            f"exec /workspace/.venv/bin/python3 -m src.eval_grader.server "
            f"--workspace-root /workspace "
            f"--hidden-eval-dir /hidden_eval "
            f"--host 0.0.0.0 "
            f"--port {host_port} "
            f"--token {json.dumps(token)} "
            f"> {log_path} 2>&1"
        )
        result = subprocess.run(
            ["docker", "exec", "-d", self._container_id, "sh", "-lc", cmd],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            details = _clean_subprocess_output(result.stderr or result.stdout)
            raise RuntimeError(
                "Failed to start grader server inside Docker container.\n"
                f"Command output:\n{details}"
            )

    @staticmethod
    def _container_python_requirements() -> list[str]:
        """Dependencies installed into the Docker-only Python environment."""
        repo_root = Path(__file__).resolve().parents[2]
        pyproject = repo_root / "pyproject.toml"
        if not pyproject.is_file():
            return ["numpy>=2.0"]
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project", {})
        requirements: list[str] = []
        raw_deps = project.get("dependencies", [])
        if isinstance(raw_deps, list):
            requirements.extend(str(dep) for dep in raw_deps)
        optional = project.get("optional-dependencies", {})
        if isinstance(optional, dict):
            examples = optional.get("examples", [])
            if isinstance(examples, list):
                requirements.extend(str(dep) for dep in examples)
        return requirements

    # -- Internal helpers --

    def _ensure_image(self) -> None:
        """Build image if it doesn't exist locally."""
        result = subprocess.run(
            ["docker", "image", "inspect", self._image],
            capture_output=True, check=False,
        )
        if result.returncode != 0:
            build_script = Path(__file__).resolve().parents[2] / "docker" / "build.sh"
            if build_script.exists():
                log.info("Image %s not found, building...", self._image)
                try:
                    subprocess.run(
                        ["bash", str(build_script)],
                        check=True,
                        env={**os.environ, "EUREKA_DOCKER_IMAGE": self._image},
                    )
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        f"Docker build failed. The base image may not be available.\n"
                        f"Run these commands manually first:\n"
                        f"  docker pull node:22-bookworm\n"
                        f"  bash docker/build.sh\n"
                        f"If pull fails, see the Prerequisites section in README.md "
                        f"for proxy setup and offline image transfer instructions."
                    ) from e
            else:
                raise RuntimeError(
                    f"Docker image {self._image} not found and no build script at {build_script}.\n"
                    f"Please complete the Prerequisites in README.md first."
                )

    def _resolve_env_vars(self, keys: tuple[str, ...] = _FORWARDED_ENV_VARS) -> dict[str, str]:
        """Resolve env vars from host environment + ~/.claude/settings.json."""
        settings_env = self._read_settings_env()
        resolved: dict[str, str] = {}
        for key in keys:
            value = os.environ.get(key) or settings_env.get(key)
            if value:
                resolved[key] = self._normalize_proxy_env(key, value)
        if any(_is_proxy_key(key) for key in resolved):
            no_proxy = resolved.get("NO_PROXY") or resolved.get("no_proxy") or ""
            merged_no_proxy = _merge_no_proxy(no_proxy)
            resolved.setdefault("NO_PROXY", merged_no_proxy)
            resolved.setdefault("no_proxy", merged_no_proxy)
        return resolved

    def _normalize_proxy_env(self, key: str, value: str) -> str:
        """Rewrite host-local proxy URLs when that localhost is not the container host."""
        if key.lower() == "no_proxy":
            return _merge_no_proxy(value)
        if not _is_proxy_key(key):
            return value
        override_host = os.environ.get("EUREKA_DOCKER_PROXY_HOST", "").strip()
        if override_host:
            return _rewrite_proxy_host(value, override_host)
        if platform.system() == "Linux" and self._network == "host":
            return value
        return _rewrite_proxy_host(value, _DOCKER_HOST_ALIAS)

    @staticmethod
    def _read_settings_env() -> dict[str, str]:
        """Read the 'env' block from ~/.claude/settings.json."""
        settings_path = Path.home() / ".claude" / "settings.json"
        if not settings_path.exists():
            return {}
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            env_block = data.get("env", {})
            return {k: str(v) for k, v in env_block.items()} if isinstance(env_block, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _docker_python_env_dir(self, repo_root: Path) -> tuple[Path, str]:
        """Return the host-side persistent venv directory for this Docker image."""
        image_info = self._docker_image_identity()
        key_material = (
            f"{self._image}|{image_info}|python-venv-v1|"
            f"{self._dependency_fingerprint(repo_root)}"
        )
        digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:16]
        safe_image = re.sub(r"[^A-Za-z0-9_.-]+", "-", self._image).strip("-")
        safe_image = safe_image[:48] or "image"
        key = f"{safe_image}-{digest}"
        return repo_root / ".eureka_docker" / "venvs" / key, key

    @staticmethod
    def _dependency_fingerprint(repo_root: Path) -> str:
        digest = hashlib.sha256()
        for name in ("pyproject.toml", "uv.lock"):
            path = repo_root / name
            digest.update(name.encode("utf-8") + b"\0")
            if path.is_file():
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def _docker_image_identity(self) -> str:
        result = subprocess.run(
            [
                "docker", "image", "inspect", self._image,
                "--format", "{{.Id}} {{.Architecture}} {{.Os}}",
            ],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return "unknown-image"

    def _build_run_args(
        self, workspace_dir: Path, env_vars: dict[str, str], agent_home: Path,
        secure_eval_env: dict[str, str],
        extra_mounts: list[tuple[Path, str, str]] | None = None,
        ports: list[tuple[int, int]] | None = None,
        extra_env: dict[str, str] | None = None,
        command: list[str] | None = None,
        include_agent_mounts: bool = True,
    ) -> list[str]:
        home = Path.home()
        repo_root = Path(__file__).resolve().parents[2]
        docker_python_env, docker_env_key = self._docker_python_env_dir(repo_root)
        project_playwright_config = repo_root / ".claude" / "playwright-mcp.json"
        claude_skills = home / ".claude" / "skills"
        playwright_cache = home / ".cache" / "ms-playwright"
        uv_cache = home / ".cache" / "uv"
        uv_store = home / ".local" / "share" / "uv"
        uid = os.getuid()
        gid = os.getgid()
        host_is_linux = platform.system() == "Linux"
        python_env_mount = docker_python_env
        self._bootstrap_python_env = True
        self._python_env_key = docker_env_key
        python_env_mount.mkdir(parents=True, exist_ok=True)

        args = [
            "run", "--rm", "-d",
            "--init",
            "--user", f"{uid}:{gid}",
            "--network", self._network,
            "--workdir", "/workspace",
            "--hostname", "eureka-agent",
            "-e", f"HOME={home}",
            "-e", f"USER={os.environ.get('USER', 'user')}",
            "-e", "TERM=xterm-256color",
            # Workspace (read-write)
            "-v", f"{workspace_dir.resolve()}:/workspace",
            # Internal coordination dir (read-only): contains gpu_helpers.py
            # and pty_proxy.py. Agents must not be able to modify these.
            # This sub-mount stacks on top of the workspace mount above.
            "-v", f"{(workspace_dir / '.eureka_internal').resolve()}:/workspace/.eureka_internal:ro",
            # Writable HOME overlay — must come before the read-only child
            # mounts below so Docker stacks them on top of this base.
            "-v", f"{agent_home}:{home}",
        ]

        if include_agent_mounts and claude_skills.is_dir():
            # NB: ~/.claude.json is NOT bind-mounted read-only — a per-run
            # writable copy is seeded into agent_home by _ensure_agent_home.
            args.extend(["-v", f"{claude_skills}:{home}/.claude/skills:ro"])

        if include_agent_mounts:
            args.extend(["-e", "CLAUDE_CODE_DISABLE_AUTOUPDATER=1"])

        if self._needs_host_gateway_alias():
            args.extend(["--add-host", "host.docker.internal:host-gateway"])

        for host_port, container_port in ports or []:
            args.extend(["-p", f"127.0.0.1:{host_port}:{container_port}"])

        if include_agent_mounts and host_is_linux:
            # Compatibility fallback for existing Linux setups that rely on a
            # host-installed Claude CLI. macOS installs Mach-O binaries, which
            # fail inside Linux containers with "exec format error"; on macOS
            # the Docker image's own Linux CLI is used instead.
            args.extend([
                "-v", f"{home}/.local/bin:{home}/.local/bin:ro",
                "-v", f"{home}/.local/lib:{home}/.local/lib:ro",
                "-v", f"{home}/.local/share/claude:{home}/.local/share/claude:ro",
            ])

        # NB: ~/.claude/projects/ is NOT mounted read-only. The agent_home
        # overlay provides a writable version so Claude Code can persist
        # session state, which is required for --resume after SIGTERM.

        if include_agent_mounts and host_is_linux:
            host_bin = home / "bin"
            if host_bin.is_dir():
                args.extend(["-v", f"{host_bin}:{host_bin}:ro"])

            nvm_dir = home / ".nvm"
            if nvm_dir.is_dir():
                args.extend(["-v", f"{nvm_dir}:{nvm_dir}:ro"])

            npm_dir = home / ".npm"
            if npm_dir.is_dir():
                args.extend(["-v", f"{npm_dir}:{npm_dir}"])

            if playwright_cache.is_dir():
                args.extend(["-v", f"{playwright_cache}:{playwright_cache}"])

        if include_agent_mounts and project_playwright_config.is_file():
            args.extend([
                "-v",
                f"{project_playwright_config}:{home}/.claude/playwright-mcp.json:ro",
            ])

        if python_env_mount.is_dir():
            args.extend(["-v", f"{python_env_mount.resolve()}:/workspace/.venv"])
            args.extend(["-e", "VIRTUAL_ENV=/workspace/.venv"])

        for src, dst, mode in extra_mounts or []:
            suffix = f":{mode}" if mode else ""
            args.extend(["-v", f"{src}:{dst}{suffix}"])

        if uv_cache.is_dir():
            args.extend(["-v", f"{uv_cache}:{uv_cache}"])

        if uv_store.is_dir():
            args.extend(["-v", f"{uv_store}:{uv_store}"])

        # GPU support
        self._add_gpu_args(args)

        # Environment variables
        for key, value in env_vars.items():
            args.extend(["-e", f"{key}={value}"])

        # Secure eval environment variables
        for key, value in secure_eval_env.items():
            args.extend(["-e", f"{key}={value}"])

        for key, value in (extra_env or {}).items():
            args.extend(["-e", f"{key}={value}"])

        # Set PATH inside container
        path_entries = [
            "/usr/local/bin",
            "/usr/local/sbin",
        ]
        if include_agent_mounts and host_is_linux:
            path_entries.append(f"{home}/.local/bin")
        if python_env_mount.is_dir():
            path_entries.append("/workspace/.venv/bin")
        path_entries.extend([
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        ])
        args.extend(["-e", f"PATH={':'.join(path_entries)}"])

        if not (extra_env or {}).get("PYTHONPATH"):
            # Expose gpu_helpers via PYTHONPATH without placing it in the visible workspace
            args.extend(["-e", "PYTHONPATH=/workspace/.eureka_internal"])

        # Entrypoint: sleep forever for agent containers; grader containers
        # provide their own long-running server command.
        args.append(self._image)
        args.extend(command or ["sleep", "infinity"])
        return args

    def _add_gpu_args(self, args: list[str]) -> None:
        policy = resolve_gpu_policy(self._gpus)
        log_gpu_policy_warnings(policy, log)
        if policy.enable_docker_gpus:
            args.extend(["--gpus", "all"])
            args.extend([
                "-e",
                f"EUREKA_HOST_CUDA_DEVICES={','.join(str(i) for i in policy.allowed_gpu_ids)}",
            ])
        # Default-deny: agents and graders must acquire GPUs through gpu_helpers.
        args.extend(["-e", "CUDA_VISIBLE_DEVICES="])

    def _needs_host_gateway_alias(self) -> bool:
        if platform.system() != "Linux":
            return False
        if self._network != "host":
            return True
        return os.environ.get("EUREKA_DOCKER_PROXY_HOST", "").strip() == _DOCKER_HOST_ALIAS



def _inject_workspace_trust(config_path: Path) -> None:
    """Add /workspace as a trusted project in .claude.json.

    Claude Code's interactive mode shows a workspace-trust dialog when
    starting in an unrecognised directory.  Inside a Docker container the
    CWD is /workspace, so we pre-register it as a known project with
    ``hasTrustDialogAccepted: true`` to skip the dialog.
    """
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    projects = data.setdefault("projects", {})
    projects.setdefault("/workspace", {
        "allowedTools": [],
        "mcpContextUris": [],
        "mcpServers": {},
        "enabledMcpjsonServers": [],
        "hasTrustDialogAccepted": True,
    })
    config_path.write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )


def _format_docker_start_error(exc: subprocess.CalledProcessError) -> str:
    """Build a user-facing docker startup failure with captured output."""
    parts = [f"Docker failed to start container (exit {exc.returncode})."]
    stderr = _clean_subprocess_output(exc.stderr)
    stdout = _clean_subprocess_output(exc.stdout or exc.output)
    if stderr:
        parts.append(f"Docker stderr:\n{stderr}")
    if stdout:
        parts.append(f"Docker stdout:\n{stdout}")
    return "\n".join(parts)


def _is_proxy_key(key: str) -> bool:
    return key.lower() in {"http_proxy", "https_proxy", "all_proxy"}


def _rewrite_proxy_host(value: str, host: str) -> str:
    """Replace localhost proxy hosts with a host reachable from Docker."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc or parsed.hostname not in _LOCAL_PROXY_HOSTS:
        return value
    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        auth += "@"
    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"{auth}{host}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _merge_no_proxy(value: str) -> str:
    required = ("localhost", "127.0.0.1", "::1", "0.0.0.0", _DOCKER_HOST_ALIAS)
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    seen = {entry.lower() for entry in entries}
    for entry in required:
        if entry.lower() not in seen:
            entries.append(entry)
            seen.add(entry.lower())
    return ",".join(entries)


def _clean_subprocess_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode(errors="replace").strip()
    return str(output).strip()
