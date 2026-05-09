import dataclasses
import enum
import logging
import pathlib
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    LIBERO_PI0 = "libero_pi0"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class LocalCheckpoint:
    """Load a policy from a local finetune (path = config_name/experiment; latest step used)."""

    # Experiment directory (e.g. checkpoints/pi05_lora_local/libero90_onetask or .../1000).
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for, or path to a local checkpoint (e.g. checkpoints/pi05_lora_local/libero90_onetask/1000).
    # When a path is given, that checkpoint is loaded instead of the default policy for an environment.
    env: EnvMode | str = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default | LocalCheckpoint = dataclasses.field(default_factory=Default)


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
    EnvMode.LIBERO_PI0: Checkpoint(
        config="pi0_libero", 
        dir="gs://openpi-assets/checkpoints/pi0_libero"
    )
}


def _resolve_local_checkpoint(experiment_dir: pathlib.Path) -> tuple[str, pathlib.Path]:
    """Resolve experiment dir to (config_name, step_dir)."""
    path = experiment_dir.resolve()
    if (path / "params").exists():
        return path.parent.parent.name, path
    config_name = path.parent.name
    step_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.isdigit()]
    step_dir = max(step_dirs, key=lambda d: int(d.name)) if step_dirs else path
    return config_name, step_dir


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case LocalCheckpoint():
            config_name, step_dir = _resolve_local_checkpoint(pathlib.Path(args.policy.dir))
            return _policy_config.create_trained_policy(
                _config.get_config(config_name), str(step_dir), default_prompt=args.default_prompt
            )
        case Default():
            # If env is a path string, load that checkpoint; otherwise use default for the environment.
            if isinstance(args.env, str):
                config_name, step_dir = _resolve_local_checkpoint(pathlib.Path(args.env))
                return _policy_config.create_trained_policy(
                    _config.get_config(config_name), str(step_dir), default_prompt=args.default_prompt
                )
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
