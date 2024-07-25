import cv2
import numpy as np
import torch
from robobase.method.drqv2 import Actor, DrQV2
from robobase.method.utils import (
    extract_from_spec,
    extract_many_from_spec,
    flatten_time_dim_into_channel_dim,
    stack_tensor_dictionary,
)
from torch.distributions import Distribution

DEBUG = False
NUM_DEBUG_OBS = 3
NUM_PRIV_FEATURES = 6


class AsymActor(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def remove_priviledged_low_dim_state(self, low_dim_state):
        low_dim_state_arrays_trimmed = low_dim_state[:, :-NUM_PRIV_FEATURES]
        return low_dim_state_arrays_trimmed

    def forward(self, low_dim_obs, fused_view_feats, std) -> Distribution:
        low_dim_obs = self.remove_priviledged_low_dim_state(low_dim_obs)
        return super().forward(low_dim_obs, fused_view_feats, std)


class AsymActorCritic(DrQV2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_debug_low_dim_obs = 0
        self.num_debug_rgb_obs = 0

    def get_fully_connected_inputs_asym_actor(self) -> dict[str, tuple]:
        input_sizes = {}
        if self.rgb_latent_size > 0:
            input_sizes["fused_view_feats"] = (self.rgb_latent_size,)
        if self.low_dim_size > 0:
            input_sizes["low_dim_obs"] = (self.low_dim_size - NUM_PRIV_FEATURES,)
        if self.time_obs_size > 0:
            input_sizes["time_obs"] = (self.time_obs_size,)
        if not self.frame_stack_on_channel and self.time_dim > 0:
            for k, v in input_sizes.items():
                input_sizes[k] = (self.time_dim,) + v
        return input_sizes

    def build_actor(self):
        input_shapes = self.get_fully_connected_inputs_asym_actor()
        if "time_obs" in input_shapes:
            # We don't use time_obs for actor
            input_shapes.pop("time_obs")
        self.actor_model = self.actor_model(
            input_shapes=input_shapes,
            output_shape=self.action_space.shape[-1],
            num_envs=self.num_train_envs + self.num_eval_envs,
        )
        self.actor = AsymActor(self.actor_model).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

    def _act_extract_rgb_obs(self, observations: dict[str, torch.Tensor]):
        rgb_obs = extract_many_from_spec(observations, r"rgb.*")

        if DEBUG:
            for k, v in rgb_obs.items():
                if self.num_debug_rgb_obs >= NUM_DEBUG_OBS:
                    break
                img_array = np.transpose(v.cpu().numpy()[0, 0], (1, 2, 0))[
                    :, :, [2, 1, 0]
                ]
                img_array = img_array.astype(np.uint8)
                cv2.imwrite(f"{k}_{self.num_debug_rgb_obs}.png", img_array)
            self.num_debug_rgb_obs += 1

        rgb_obs = stack_tensor_dictionary(rgb_obs, 1)
        if self.frame_stack_on_channel:
            rgb_obs = flatten_time_dim_into_channel_dim(rgb_obs, has_view_axis=True)
        else:
            rgb_obs = rgb_obs.transpose(1, 2)
            rgb_obs = rgb_obs.view(-1, *rgb_obs.shape[2:])
        return rgb_obs

    def _act_extract_low_dim_state(self, observations: dict[str, torch.Tensor]):
        low_dim_obs = extract_from_spec(observations, "low_dim_state")

        if DEBUG and (self.num_debug_low_dim_obs < NUM_DEBUG_OBS):
            numpy_array = low_dim_obs.cpu().numpy()
            low_dim_obs_array = numpy_array[0, 0, :]
            np.savetxt(
                f"low_dim_obs_{self.num_debug_low_dim_obs}.csv",
                low_dim_obs_array,
                delimiter=",",
            )
            self.num_debug_low_dim_obs += 1

        if self.frame_stack_on_channel:
            low_dim_obs = flatten_time_dim_into_channel_dim(low_dim_obs)
        return low_dim_obs
