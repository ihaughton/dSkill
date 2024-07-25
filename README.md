# Dexterous Skill Discovery

Contains:
- UR5 robot + Robotiq gripper in a MuJoCo scene.
- Example code for constructing and executing a MuJoCo environment featuring a robot.
- Additional dSkill-related props.

## Install

```bash
pip install -e .
```

For dev install:
```bash
pip install -e ".[dev]"
```

## Launch Example

```bash
python scripts/environment_random_agent.py
```

## Launch Training

```bash
python dSkill/launch_train.py
```
