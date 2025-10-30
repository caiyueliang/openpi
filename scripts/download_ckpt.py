from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# import os

# os.environ["HTTP_PROXY"] = "http://172.16.66.122:7890"
# os.environ["HTTPS_PROXY"] = "http://172.16.66.122:7890"

# config = _config.get_config("pi05_droid")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")


checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

# # [checkpoint_dir] /root/.cache/openpi/big_vision/paligemma_tokenizer.model
# checkpoint_dir = download.maybe_download("gs://big_vision/paligemma_tokenizer.model")

print(f"[checkpoint_dir] {checkpoint_dir}")

# Create a trained policy.
# policy = policy_config.create_trained_policy(config, checkpoint_dir)

# # Run inference on a dummy example.
# example = {
#     "observation/exterior_image_1_left": ...,
#     "observation/wrist_image_left": ...,
#     ...
#     "prompt": "pick up the fork"
# }
# action_chunk = policy.infer(example)["actions"]