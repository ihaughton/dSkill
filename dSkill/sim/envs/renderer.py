"""Custom implementation of the MujocoRenderer to fix gymnasium renderers."""

import time

import glfw
import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer, WindowViewer
from mojo import Mojo


class DexRenderer(MujocoRenderer):
    """Custom implementation of the MujocoRenderer.

    PuckGymRenderer enables rendering to the on-screen window
    simultaneously with the collection of visual observations.
    """

    def __init__(self, mojo: Mojo):
        self._mojo = mojo
        super().__init__(self._mojo.model, self._mojo.data)

    def render(
        self,
        render_mode: str,
        camera_id: int | None = None,
        camera_name: str | None = None,
    ):
        if self.data != self._mojo.data or self.model != self._mojo.model:
            raise RuntimeError(
                "Renderer and Mojo data are out of sync! "
                "Renderer must be re-instantiated.",
            )
        return super().render(render_mode, camera_id, camera_name)

    def _get_viewer(self, render_mode: str):
        # Fix of the solver_iter/solver_niter typo
        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None and render_mode == "human":
            self.viewer = DexWindowViewer(self.model, self.data)
            self._set_cam_config()
            self._viewers[render_mode] = self.viewer
        self.viewer = super()._get_viewer(render_mode)
        self.viewer.make_context_current()
        return self.viewer


class DexWindowViewer(WindowViewer):
    # Add overlay cleaning when simulation is paused
    def render(self):
        """See base."""

        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.destroy_window(self.window)
                glfw.terminate()
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window,
            )
            # update scene
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.vopt,
                mujoco.MjvPerturb(),
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scn,
            )

            # marker items
            for marker in self._markers:
                self._add_marker_to_scene(marker)

            # render
            mujoco.mjr_render(self.viewport, self.scn, self.con)

            # overlay items
            if not self._hide_menu:
                for gridpos, [t1, t2] in self._overlays.items():
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.con,
                    )

            glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )
            # clear overlay
            self._overlays.clear()
            # clear markers
            self._markers.clear()

        if self._paused:
            while self._paused:
                update()
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (
                self._time_per_render * self._run_speed
            )
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

    # Fix of the AttributeError: changed self.data.solver_iter
    # to correct self.data.solver_niter
    def _create_overlay(self):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT

        if self._render_every_frame:
            self.add_overlay(topleft, "", "")
        else:
            self.add_overlay(
                topleft,
                "Run speed = %.3f x real time" % self._run_speed,
                "[S]lower, [F]aster",
            )
        self.add_overlay(
            topleft,
            "Ren[d]er every frame",
            "On" if self._render_every_frame else "Off",
        )
        self.add_overlay(
            topleft,
            "Switch camera (#cams = %d)" % (self.model.ncam + 1),
            "[Tab] (camera ID = %d)" % self.cam.fixedcamid,
        )
        self.add_overlay(topleft, "[C]ontact forces", "On" if self._contacts else "Off")
        self.add_overlay(topleft, "T[r]ansparent", "On" if self._transparent else "Off")
        if self._paused is not None:
            if not self._paused:
                self.add_overlay(topleft, "Stop", "[Space]")
            else:
                self.add_overlay(topleft, "Start", "[Space]")
                self.add_overlay(
                    topleft,
                    "Advance simulation by one step",
                    "[right arrow]",
                )
        self.add_overlay(
            topleft,
            "Referenc[e] frames",
            "On" if self.vopt.frame == 1 else "Off",
        )
        self.add_overlay(topleft, "[H]ide Menu", "")
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            self.add_overlay(topleft, "Cap[t]ure frame", "Saved as %s" % fname)
        else:
            self.add_overlay(topleft, "Cap[t]ure frame", "")
        self.add_overlay(topleft, "Toggle geomgroup visibility", "0-4")

        self.add_overlay(bottomleft, "FPS", "%d%s" % (1 / self._time_per_render, ""))
        self.add_overlay(
            bottomleft,
            "Solver iterations",
            str(self.data.solver_niter + 1),
        )
        self.add_overlay(
            bottomleft,
            "Step",
            str(round(self.data.time / self.model.opt.timestep)),
        )
        self.add_overlay(bottomleft, "timestep", "%.5f" % self.model.opt.timestep)
