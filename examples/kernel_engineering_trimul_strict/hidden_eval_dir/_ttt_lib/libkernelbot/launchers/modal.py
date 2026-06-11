import asyncio
import traceback
import uuid

import modal

from libkernelbot.consts import GPU, ModalGPU
from libkernelbot.report import RunProgressReporter
from libkernelbot.run_eval import FullResult, SystemInfo
from libkernelbot.utils import setup_logging

from .launcher import Launcher

logger = setup_logging(__name__)

_REQUEUE_COUNT_DICT_NAME = "discord-bot-requeue-counts"
_REQUEUE_SENTINEL = "[MODAL_REQUEUE]"


class ModalLauncher(Launcher):
    def __init__(self, add_include_dirs: list, app_name: str):
        super().__init__("Modal", gpus=ModalGPU)
        self.additional_include_dirs = add_include_dirs
        self.app_name = app_name  # "discord-bot-runner" | "discord-bot-runner-nvfp4"
    
    async def run_submission(
        self, config: dict, gpu_type: GPU, status: RunProgressReporter
    ) -> FullResult:
        loop = asyncio.get_event_loop()
        # Add a stable per-request id so the Modal runner can count requeues across retries.
        config = dict(config)
        request_id = uuid.uuid4().hex
        config["_modal_request_id"] = request_id
        if config["lang"] == "cu":
            config["include_dirs"] = config.get("include_dirs", []) + self.additional_include_dirs
        func_type = "pytorch" if config["lang"] == "py" else "cuda"
        func_name = f"run_{func_type}_script_{gpu_type.value.lower()}"

        logger.info(f"Starting Modal run using {func_name}")

        await status.push("⏳ Waiting for Modal run to finish...")

        try:
            result = await loop.run_in_executor(
                None,
                lambda: modal.Function.from_name(self.app_name, func_name).remote(config=config),
            )
        except Exception as e:
            # Best-effort cleanup + surface a structured error instead of crashing the caller.
            requeues = 0
            try:
                d = modal.Dict.from_name(_REQUEUE_COUNT_DICT_NAME, create_if_missing=True)
                try:
                    requeues = int(d.pop(request_id))
                except KeyError:
                    requeues = 0
            except Exception:
                pass

            err = "".join(traceback.format_exception(e))
            if str(e).startswith(_REQUEUE_SENTINEL):
                # The runner increments the counter on every banned-GPU *attempt*.
                # If we got the sentinel error back here, it means retries were exhausted,
                # so the last attempt did NOT result in a requeue. Convert attempts -> retries used.
                requeues = max(0, requeues - 1)
                err = (
                    f"{err}\n\n"
                    f"Requeues used: {requeues}\n"
                    f"Modal requeue retries were exhausted while avoiding a banned GPU form factor.\n"
                    f"Consider increasing retries in `src/runners/modal_runner_archs.py` or adjusting the banned list."
                )

            await status.update("❌ Waiting for modal run to finish... Failed")
            return FullResult(success=False, error=err, runs={}, system=SystemInfo(requeues=requeues))

        await status.update("✅ Waiting for modal run to finish... Done")

        return result
