"""Launch file for running RL experiments in robobase."""

import logging
from pathlib import Path

import hydra
from robobase.workspace import Workspace

from dSkill.robobase_wrapper import dSkillFactory

log = logging.getLogger(__name__)


@hydra.main(config_path="cfgs", config_name="dSkill", version_base=None)
def main(cfg):
    """Main.

    Args:
        cfg: Hydra config.
    """
    workspace = Workspace(cfg, env_factory=dSkillFactory(sim=True))
    root_dir = Path.cwd()
    snapshot = root_dir / "snapshot_to_resume.pt"
    if snapshot.exists():
        log.info(f"resuming: {snapshot}")
        workspace.load_snapshot(snapshot)

    workspace.train()


if __name__ == "__main__":
    main()
