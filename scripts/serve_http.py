# serve_policy_http.py
import os
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

# 或者提供默认值
states_len = os.getenv('STATES_LEN', 8)


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
    status: int = 0
    message: str = "success"
    result: Dict[str, Any] = {}



def find_first_params_dir(root_dir):
    """
    在给定的根目录下，查找第一个包含名为 'params' 的子文件夹的目录，
    并返回该目录的路径。如果未找到，返回 None。
    
    :param root_dir: 要搜索的根目录路径（字符串）
    :return: 第一个包含 'params' 子目录的目录路径（字符串）或 None
    """
    for dirpath, dirnames, _ in os.walk(root_dir):
        if 'params' in dirnames:
            return dirpath
    return None

def base64_to_pil(b64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(b64_str)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")
    
# === Main Server Logic ===

def main(args: Args) -> None:
    logging.warning(f"[main] old args: {args}")

    base_dir = find_first_params_dir(root_dir=args.policy.dir)
    if base_dir:
        logging.warning(f"[main] 目录: {base_dir} 中找到 'params' 子文件夹，使用该目录作为模型路径。")
        args.policy.dir = base_dir
        logging.warning(f"[main] new args: {args}")
    else:
        logging.warning(f"[main] 目录: {args.policy.dir} 中未找到 'params' 子文件夹，请检查模型路径。")
        exit(1)

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

            if len(request.state) != states_len:
                return InferenceResponse(status=1, message=f"invalid state length, need size: (1 x {states_len})")
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

            result = {
                "action": actions[:5].tolist(),
                "timestamp": timestamp,
            }
            return InferenceResponse(status=0, result=result, message="success")

        except Exception as e:
            logging.exception(e)
            return InferenceResponse(status=1, message=f"{str(e)}")

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