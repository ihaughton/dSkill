"""Example of running random actions in env."""

import time

from dSkill.sim.envs.vertical_slide import VerticalSlideDomainRand


def main():
    """Main."""
    env = VerticalSlideDomainRand(
        render_mode="human",
        friction_lower=0.9,
        object_name="vention_bracket",
        object_offset=[0.0, 0, -0.1],
        cameras=["ur5e/upper_wrist_camera", "ur5e/lower_wrist_camera"],
    )
    print("Observation Space Shape %s", env.observation_space.shape)
    print("Sample observation %s", env.observation_space.sample())

    print("Action Space Shape %s", env.action_space)
    print("Action Space Sample %s", env.action_space.sample())

    obs, _ = env.reset(seed=2)
    env.render()
    for step in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(
            f"step = {step}, action={action} obs= {obs} "
            f"reward={reward} terminated={terminated} truncated= {truncated}",
        )
        if terminated or truncated:
            print(
                f"Goal reached! reward={reward} "
                f"terminated={terminated} truncated={truncated}",
            )
            obs, _ = env.reset()
            time.sleep(2)
    env.close()


if __name__ == "__main__":
    main()
