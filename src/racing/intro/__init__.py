"""A standalone, deterministic, looping Formula 110 opening credit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from math import atan, atan2, cos, degrees, floor, hypot, isfinite, pi, radians, sin, tan
from pathlib import Path
from time import perf_counter
from typing import Any

from racing.intro import shaders
from racing.intro.driving import DURATION as LOOP_SECONDS
from racing.intro.driving import VISUAL_SCALE, simulate_sequence

BLUE = (75 / 255, 156 / 255, 211 / 255, 1.0)
LANE_BOUNDS = (-2.7, 2.2)


@dataclass(frozen=True)
class Pose:
    """Periodic lighting and logo pose, independent of the vehicle simulation."""

    phase: float
    logo_heading: float


def pose_at(seconds: float, duration: float = LOOP_SECONDS) -> Pose:
    if not isfinite(seconds) or not isfinite(duration) or duration <= 0:
        raise ValueError("time must be finite and duration must be positive and finite")
    t = (seconds % duration) / duration
    return Pose(t * 2 * pi, 360 * t - 100)


def smoothstep(start: float, end: float, value: float) -> float:
    amount = max(0.0, min(1.0, (value - start) / (end - start)))
    return amount * amount * (3 - 2 * amount)


class Intro:
    """Studio stage, reusable car art, and a choreographed forty-second loop."""

    def __init__(self, args: argparse.Namespace) -> None:
        from racing.graphics.panda_config import configure_panda_antialiasing, quiet_panda_image_logs

        core = import_module("panda3d.core")
        quiet_panda_image_logs()
        configure_panda_antialiasing(4)
        core.loadPrcFileData("", "audio-library-name null\nsync-video true")
        u = import_module("ursina")
        self.u, self.core, self.args = u, core, args
        self.app = u.Ursina(
            title="Formula 110 | Opening credit",
            size=args.size,
            fullscreen=False,
            borderless=False,
            development_mode=False,
            editor_ui_enabled=False,
            window_type="offscreen" if args.capture else "onscreen",
            icon=str(Path(__file__).parents[1] / "assets/textures/ursina.ico"),
            vsync=not bool(args.capture),
        )
        self.app.setBackgroundColor(0.003, 0.006, 0.012, 1)
        # Explicit clears matter for animated offscreen buffers on Cocoa.
        self.app.win.setClearColorActive(True)
        self.app.win.setClearDepthActive(True)
        u.camera.display_region.setClearColor((0.003, 0.006, 0.012, 1))
        u.camera.display_region.setClearColorActive(True)
        u.camera.display_region.setClearDepthActive(True)
        # Offscreen buffers do not send the window resize event Ursina relies on.
        u.window.update_aspect_ratio()
        u.window.exit_button.enabled = False
        u.window.fps_counter.enabled = False
        if not args.capture:
            u.mouse.visible = False
        u.scene.setAntialias(core.AntialiasAttrib.MAuto)
        self.surface = u.Shader(
            language=u.Shader.GLSL,
            vertex=shaders.VERTEX,
            fragment=shaders.SURFACE,
            default_input={"eye": u.Vec3(0, 3.5, -19), "phase": 0.0, "reflection": 0.0, "gloss": 1.0},
        )
        self.glow_shader = u.Shader(language=u.Shader.GLSL, vertex=shaders.VERTEX, fragment=shaders.GLOW)
        self.beam_shader = u.Shader(language=u.Shader.GLSL, vertex=shaders.VERTEX, fragment=shaders.BEAM)
        self.surfaces: list[Any] = []
        self._stage()
        self._logo()
        self._car()
        self._donut_haze()
        self._finish()
        self._typography()
        self.started = perf_counter()
        self.elapsed = 0.0
        self.paused = False
        self.driver = u.Entity(update=self._update, input=self._input)
        self.fullscreen = False
        self.windowed_properties: Any = None
        if args.fullscreen and not args.capture:
            self._set_fullscreen(True)
        self.seek(args.time)

    def _additive(self, entity: Any) -> Any:
        c = self.core
        entity.setAttrib(
            c.ColorBlendAttrib.make(c.ColorBlendAttrib.MAdd, c.ColorBlendAttrib.OOne, c.ColorBlendAttrib.OOne), 20
        )
        entity.setDepthWrite(False)
        entity.setTransparency(c.TransparencyAttrib.MAlpha, 10)
        return entity

    def _finish(self) -> None:
        """Bloom bright highlights while keeping the credit typography sharp."""
        manager = import_module("direct.filter.FilterManager")
        self.filter_manager = manager.FilterManager(self.app.win, self.app.cam)
        texture = self.core.Texture("intro-studio-color")
        texture.setWrapU(self.core.Texture.WMClamp)
        texture.setWrapV(self.core.Texture.WMClamp)
        # A color-only target avoids multisampled depth-texture errors on macOS.
        self.post_quad = self.filter_manager.renderSceneInto(colortex=texture)
        if self.post_quad is None:
            raise RuntimeError("Could not allocate the intro lighting buffer")
        self.post_quad.setShader(self.core.Shader.make(self.core.Shader.SL_GLSL, shaders.SCREEN_VERTEX, shaders.FINISH))
        self.post_quad.setShaderInput("tex", texture)

    def _glow(
        self,
        position: tuple[float, float, float],
        scale: Any,
        alpha: float,
        tint: tuple[float, float, float] = (0.25, 0.58, 1.0),
    ) -> Any:
        entity = self.u.Entity(
            model="quad",
            position=position,
            scale=scale,
            shader=self.glow_shader,
            color=(*tint, alpha),
            double_sided=True,
        )
        return self._additive(entity)

    def _stage(self) -> None:
        u = self.u
        self._glow((0, 3.8, 6), (23, 12), 0.34)
        self._glow((-6, 2.0, 4), (10, 8), 0.16)
        self._glow((6, 2.0, 4), (10, 8), 0.16)
        self._glow((-3.5, 6.5, 5.5), (13, 5), 0.09, (0.75, 0.85, 1.0))
        argyle_path = Path(__file__).parents[1] / "assets/graphics/argyle.png"
        argyle = self.app.loader.loadTexture(str(argyle_path))
        argyle.setWrapU(self.core.Texture.WMRepeat)
        argyle.setWrapV(self.core.Texture.WMClamp)
        argyle.setMinfilter(self.core.Texture.FTLinearMipmapLinear)
        argyle.setMagfilter(self.core.Texture.FTLinear)
        argyle.setAnisotropicDegree(16)
        floor_shader = u.Shader(
            language=u.Shader.GLSL,
            vertex=shaders.VERTEX,
            fragment=shaders.FLOOR,
            default_input={
                "car_x": -16.0,
                "car_z": -0.65,
                "car_heading": radians(110),
                "phase": 0.0,
                "eye": u.Vec3(0, 3.5, -19),
                "lane_bounds": u.Vec2(*LANE_BOUNDS),
                "argyle_aspect": argyle.getOrigFileXSize() / argyle.getOrigFileYSize(),
                "argyle_texture": argyle,
            },
        )
        self.floor = u.Entity(
            model="plane", scale=(100, 1, 100), y=-0.015, shader=floor_shader, color=(1, 1, 1, 0.58), double_sided=True
        )
        self.floor.setDepthWrite(False)
        self.floor.setTransparency(self.core.TransparencyAttrib.MAlpha, 10)
        self.floor.setBin("transparent", 20)
        for z in LANE_BOUNDS:
            u.Entity(
                model="cube", position=(0, 0.006, z), scale=(42, 0.008, 0.012), color=(0.09, 0.25, 0.40, 1), unlit=True
            )
            self._glow((0, 0.014, z), (32, 0.48), 0.16).rotation_x = 90
        self.beams: list[tuple[Any, int, int]] = []
        for side in (-1, 1):
            for index in range(3):
                x = side * (5.8 + index * 1.6)
                mount = u.Entity(position=(x, 0.08, 3.6 + index * 0.7))
                mesh = u.Mesh(
                    vertices=[(-0.045, 0, 0), (0.045, 0, 0), (1.8, 11, 0), (-1.8, 11, 0)],
                    triangles=[(0, 1, 2), (0, 2, 3)],
                    uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
                    normals=[(0, 0, -1)] * 4,
                )
                beam = u.Entity(
                    parent=mount,
                    model=mesh,
                    shader=self.beam_shader,
                    color=(0.54, 0.72, 1.0, 0.40) if index == 0 else (0.24, 0.50, 0.86, 0.27),
                    double_sided=True,
                )
                self._additive(beam)
                self.beams.append((mount, side, index))
                self._glow((x, 0.12, 3.55 + index * 0.7), (1.1, 1.1), 0.6)
                self._glow((x, 0.12, 3.54 + index * 0.7), (2.8, 0.12), 0.18, (0.6, 0.8, 1.0))
                u.Entity(
                    model="sphere",
                    position=(x, 0.09, 3.5 + index * 0.7),
                    scale=(0.15, 0.04, 0.13),
                    color=(0.55, 0.8, 1, 1),
                    unlit=True,
                )
        # A matched pair of architectural light blades frames the central mark.
        for side in (-1, 1):
            x, y, z = side * 7.15, 3.5, 4.2
            blade = u.Entity(
                model="cube",
                position=(x, y, z),
                scale=(0.022, 4.3, 0.022),
                rotation_z=-side * 19,
                color=(0.55, 0.77, 1.0, 1),
                unlit=True,
            )
            halo = self._glow((x, y, z - 0.03), (0.75, 5.8), 0.22, (0.46, 0.7, 1.0))
            halo.rotation_z = blade.rotation_z
            self._glow((x, 0.012, 2.4), (2.4, 6.0), 0.12).rotation_x = 90
        self.mirror = u.Entity(name="floor-reflections", scale=(1, -1, 1), y=-0.035)

    def _logo(self) -> None:
        from racing.intro.geometry import logo_mesh

        u = self.u
        self.logo = u.Entity(
            name="beveled-f110", model=logo_mesh(u), y=3.35, z=0.55, color=BLUE, shader=self.surface, double_sided=True
        )
        self.logo.setTransparency(self.core.TransparencyAttrib.MNone, 10)
        self.surfaces.append(self.logo)
        self.logo_reflection = self.logo.copyTo(self.mirror)
        self.logo_reflection.setShaderInput(self.core.ShaderInput("reflection", 1.0, 10))

    def _car(self) -> None:
        from racing.graphics.render_assets import create_scene_assets
        from racing.graphics.vehicle_visuals import add_robot_visuals, create_showcase_robot

        u = self.u
        self.rig = create_showcase_robot(u)
        assets = create_scene_assets()
        add_robot_visuals(ursina=u, robot=self.rig, assets=assets, team_color=BLUE)
        self.driving = simulate_sequence()
        self.chassis = self.rig.chassis_np
        self.car = u.Entity(name="physical-car-assembly", scale=VISUAL_SCALE)
        self.chassis.parent = self.car
        # Apply a consistent studio paint shader to every existing car part.
        stack = [self.chassis]
        while stack:
            entity = stack.pop()
            stack.extend(entity.children)
            if entity.model is not None:
                entity.shader = self.surface
                entity.setTransparency(self.core.TransparencyAttrib.MNone, 10)
                entity.set_shader_input("gloss", 1.0 if entity in self.rig.team_paint_entities else 0.28)
                self.surfaces.append(entity)
        # Bullet supplies world-space wheel transforms, including suspension,
        # steering, and rotation. Keep wheels alongside the chassis, as in-game.
        for wheel in self.rig.wheel_nodes:
            wheel.parent = self.car
        self.car_reflection = self.car.copyTo(self.mirror)
        self.reflected_chassis = self.car_reflection.find(f"**/{self.chassis.name}")
        self.car_reflection.setShaderInput(self.core.ShaderInput("reflection", 1.0, 10))
        self.reflected_wheels = [self.car_reflection.find(f"**/{wheel.name}") for wheel in self.rig.wheel_nodes]
        self.underglow = self._glow((0, 0.014, -0.65), (5.0, 1.9), 0.15)
        self.underglow.rotation_x = 90

    def _donut_haze(self) -> None:
        """A small, short-lived trail from the rear tires during the donut."""
        u = self.u
        shader = u.Shader(language=u.Shader.GLSL, vertex=shaders.VERTEX, fragment=shaders.SMOKE)
        self.tire_haze: list[tuple[Any, float, int]] = []
        for index in range(6):
            for side in (-1, 1):
                wisp = u.Entity(model="quad", shader=shader, double_sided=True, enabled=False)
                wisp.setTransparency(self.core.TransparencyAttrib.MAlpha, 10)
                wisp.setDepthWrite(False)
                self.tire_haze.append((wisp, 0.05 + index * 0.17, side))

    def _typography(self) -> None:
        u = self.u
        self.presenter = u.Text(
            parent=u.camera.ui,
            text="  ".join("COMP110 PRESENTS"),
            origin=(0, 0),
            y=0.427,
            scale=0.43,
            color=(0.42, 0.55, 0.66, 1),
            use_tags=False,
        )
        self.title = u.Text(
            parent=u.camera.ui,
            text="  ".join(self.args.title.upper()),
            origin=(0, 0),
            y=0.385,
            scale=0.65,
            color=(0.64, 0.77, 0.88, 1),
            use_tags=False,
        )
        self.subtitle = u.Text(
            parent=u.camera.ui,
            text="  ".join(self.args.subtitle.upper()),
            origin=(0, 0),
            y=-0.382,
            scale=0.48,
            color=(0.38, 0.52, 0.65, 1),
            use_tags=False,
        )
        for x in (-0.21, 0.21):
            u.Entity(
                parent=u.camera.ui,
                model="quad",
                position=(x, -0.382, 0),
                scale=(0.06, 0.0006),
                color=(0.2, 0.4, 0.57, 1),
            )

    def seek(self, seconds: float) -> None:
        u = self.u
        pose = pose_at(seconds, self.args.duration)
        self.logo.rotation_y = pose.logo_heading
        self.logo.y = 3.35 + 0.045 * sin(pose.phase)
        self.logo_reflection.setTransform(self.logo.getTransform())
        driving_seconds = seconds * LOOP_SECONDS / self.args.duration
        frame = self.driving.sample(driving_seconds)
        car_x, _, car_z = (value * VISUAL_SCALE for value in frame.chassis.position)
        visible = frame.visible
        self.car.enabled = visible
        self.car_reflection.show() if visible else self.car_reflection.hide()
        self.underglow.x, self.underglow.z = car_x, car_z
        self.underglow.rotation_y = frame.heading - 90
        self.underglow.enabled = visible
        for node, transform, reflected in zip(
            (self.chassis, *self.rig.wheel_nodes),
            (frame.chassis, *frame.wheels),
            (self.reflected_chassis, *self.reflected_wheels),
            strict=True,
        ):
            # Replace the entire transform to avoid decomposition/rounding
            # accumulating as the loop repeats or the user seeks backwards.
            state = self.core.TransformState.makePosQuatScale(
                u.Vec3(*transform.position), self.core.Quat(*transform.rotation), u.Vec3(1, 1, 1)
            )
            node.setTransform(state)
            reflected.setTransform(state)
        for mount, side, index in self.beams:
            mount.rotation_z = -side * (20 + index * 8) + 13 * sin(pose.phase + index * 1.2)
        loop_time = driving_seconds % LOOP_SECONDS
        # Anticipate the alternate donut's approach to the lens, then glide
        # back to the original composition after its exit.
        pullback = (floor(driving_seconds / LOOP_SECONDS) % 2) * smoothstep(9.0, 12.3, loop_time)
        pullback *= 1 - smoothstep(14.1, 16.5, loop_time)
        u.camera.position = (
            0.22 * sin(pose.phase),
            3.75 + 0.12 * cos(pose.phase) - 0.35 * pullback,
            -18.8 + 0.3 * cos(pose.phase) - 6.5 * pullback,
        )
        aspect = u.window.aspect_ratio
        u.camera.fov = degrees(2 * atan(tan(radians(21 + 2 * pullback)) * max(1, aspect / (16 / 9))))
        u.camera.clip_plane_far = 150
        dx, dy, dz = -u.camera.x, 2.0 - 0.5 * pullback - u.camera.y, 0.3 - u.camera.z
        u.camera.rotation = (degrees(atan2(-dy, hypot(dx, dz))), degrees(atan2(dx, dz)), 0)
        eye = u.camera.world_position
        for entity in self.surfaces:
            entity.set_shader_input("phase", pose.phase)
            entity.set_shader_input("eye", eye)
        for reflected in (self.car_reflection, self.logo_reflection):
            reflected.setShaderInput(self.core.ShaderInput("phase", pose.phase, 10))
            reflected.setShaderInput(self.core.ShaderInput("eye", eye, 10))
        self.floor.set_shader_input("car_x", car_x)
        self.floor.set_shader_input("car_z", car_z)
        self.floor.set_shader_input("car_heading", radians(frame.heading))
        self.floor.set_shader_input("phase", pose.phase)
        self.floor.set_shader_input("eye", eye)
        self.post_quad.setShaderInput("pixel_size", u.Vec2(1 / self.app.win.getXSize(), 1 / self.app.win.getYSize()))
        for wisp, age, side in self.tire_haze:
            past = self.driving.sample(driving_seconds - age)
            wisp.enabled = visible and past.stage == "donut" and past.skid > 0.05
            tire = past.wheels[2 if side == -1 else 3].position
            wisp.position = (tire[0] * VISUAL_SCALE, 0.26 + age * 0.38, tire[2] * VISUAL_SCALE)
            wisp.scale = (0.65 + age * 0.9, 0.42 + age * 0.7)
            wisp.color = (0.30, 0.39, 0.48, 0.16 * past.skid * (1 - age / 1.2) ** 2)

    def _update(self) -> None:
        if self.args.capture:
            return
        now = perf_counter()
        if not self.paused:
            self.elapsed += now - self.started
        self.started = now
        self.seek(self.args.time + self.elapsed)

    def _input(self, key: str) -> None:
        if key == "escape":
            self.u.application.quit()
        elif key == "space":
            self.paused = not self.paused
        elif key == "r":
            self.elapsed = -self.args.time
        elif key == "f" and not self.args.capture:
            self._set_fullscreen(not self.fullscreen)

    def _set_fullscreen(self, enabled: bool) -> None:
        """Enter fullscreen, restoring the exact prior window bounds on exit.

        Ursina's fullscreen property is not initialized from its constructor in
        the project's pinned version, so keep this program's state explicitly.
        """
        if enabled:
            self.windowed_properties = self.core.WindowProperties(self.app.win.getProperties())
            props = self.core.WindowProperties()
            monitor = self.u.window.main_monitor
            width = monitor.width if monitor else self.app.pipe.getDisplayWidth()
            height = monitor.height if monitor else self.app.pipe.getDisplayHeight()
            props.setUndecorated(True)
            props.setFullscreen(True)
            props.setSize(width, height)
            props.setOrigin(monitor.x if monitor else 0, monitor.y if monitor else 0)
        else:
            props = self.windowed_properties
        self.app.win.requestProperties(props)
        self.fullscreen = enabled


def positive_number(value: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def finite_number(value: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise argparse.ArgumentTypeError("must be finite")
    return number


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x"))
        if width < 320 or height < 240:
            raise ValueError
        return width, height
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use WIDTHxHEIGHT, at least 320x240") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Loop the standalone Formula 110 3D event opening credit.")
    parser.add_argument("--fullscreen", action="store_true", help="open fullscreen for event projection")
    parser.add_argument("--size", type=parse_size, default=(1600, 900), help="window/capture size (default: 1600x900)")
    parser.add_argument("--duration", type=positive_number, default=LOOP_SECONDS, help="seconds per loop (default: 40)")
    parser.add_argument("--title", default="FORMULA 110", help="small heading above the logo")
    parser.add_argument("--subtitle", default="AUTONOMOUS RACING", help="small caption below the stage")
    parser.add_argument("--time", type=finite_number, default=0.0, help="start at this many seconds into the loop")
    parser.add_argument("--capture", type=Path, metavar="PNG", help="save one frame offscreen, then exit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    intro = Intro(args)
    try:
        if args.capture:
            args.capture.parent.mkdir(parents=True, exist_ok=True)
            for _ in range(6):
                intro.app.step()
            result = intro.app.screenshot(namePrefix=str(args.capture.resolve()), defaultFilename=False)
            if result is None:
                raise RuntimeError("Panda3D did not write the intro frame")
            print(f"Captured {result}")
        else:
            print("Formula 110 intro — Space: pause / R: restart / F: fullscreen / Esc: exit")
            intro.app.run()
    finally:
        intro.filter_manager.cleanup()
        intro.app.destroy()
