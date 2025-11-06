# serve_policy_http.py
import dataclasses
import enum
import logging
import socket
from typing import Any, Dict, Optional

import numpy as np
import tyro
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import logging
import base64
from io import BytesIO
from PIL import Image

class EnvMode(enum.Enum):
    """Supported environments."""
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""
    env: EnvMode = EnvMode.LIBERO
    default_prompt: Optional[str] = None
    port: int = 8080
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    logging.warning(f"[create_policy] args.policy: {args.policy}")
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


# === FastAPI Model Definitions ===

class InferenceRequest(BaseModel):
    image: Optional[str] = None
    wrist_image: Optional[str] = None
    state: Optional[list] = None
    prompt: Optional[str] = None  # 如果请求没给，则使用 default_prompt


class InferenceResponse(BaseModel):
    action: list
    timestamp: float
    # 可根据实际返回添加更多字段


def base64_to_pil(b64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(b64_str)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")
    
# === Main Server Logic ===

def main(args: Args) -> None:
    logging.warning(f"[main] args: {args}")
    policy = create_policy(args)
    logging.warning(f"[main] policy: {policy}")
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # 创建 FastAPI 应用
    app = FastAPI(
        title="OpenPI Policy Inference Server",
        description="Serving robot policy models via HTTP.",
        version="1.0.0"
    )

    @app.get("/")
    def root():
        return {"message": "OpenPI Policy Server is running", "env": args.env.value}

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    @app.post("/act", response_model=InferenceResponse)
    def act(request: InferenceRequest):
        try:
            logging.info(f"[act] Received request: {request.prompt}; {request.state}")

            # 构造输入数据（根据你的 policy 接口调整）
            data = {
                "observation/image": base64_to_pil(request.image),
                "observation/wrist_image": base64_to_pil(request.wrist_image),
                "observation/state": np.array(request.state),
                "prompt": request.prompt or args.default_prompt,
            }

            # 调用策略模型推理
            result = policy.infer(data)
            # logging.info(f"[act] Returning result: {result}")

            # 假设 result 包含 'action' 字段
            actions = result.get("actions") if isinstance(result, dict) else result
            timestamp = result.get("timestamp", 0.0) if isinstance(result, dict) else 0.0

            return InferenceResponse(action=actions[:5].tolist(), timestamp=timestamp)

        except Exception as e:
            logging.exception(e)
            logging.error("Policy inference failed", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    # 启动前打印信息
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Starting HTTP server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)
    logging.info("Visit http://%s:%d/docs for API documentation", local_ip, args.port)

    # 使用 Uvicorn 运行应用
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))