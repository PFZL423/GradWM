"""Handwritten capsule arm MJCF generator."""

N_ARM_DOFS = 7
N_FINGER_DOFS = 2
TOTAL_DOFS = N_ARM_DOFS + N_FINGER_DOFS
BODY_NAMES_ARM = [f"L{i}" for i in range(1, N_ARM_DOFS + 1)]
BODY_NAME_PALM = "ee_palm"
FINGER_BODIES = ["finger_left", "finger_right"]
TCP_MARKER_BODY = "tcp_marker"


def _fmt(x: float) -> str:
    return f"{float(x):.8g}"


def _fmt_vec(xs: tuple[float, ...]) -> str:
    return " ".join(_fmt(x) for x in xs)


def _axis_vec(axis: str) -> str:
    axes = {
        "x": "1 0 0",
        "y": "0 1 0",
        "z": "0 0 1",
    }
    try:
        return axes[axis.lower()]
    except KeyError as e:
        raise ValueError(f"joint axis must be one of x/y/z, got {axis!r}") from e


def _check_len(label: str, value: tuple, expected: int) -> None:
    if len(value) != expected:
        raise ValueError(f"{label} must have length {expected}, got {len(value)}")


def _check_positive(label: str, value: tuple[float, ...]) -> None:
    if any(x <= 0 for x in value):
        raise ValueError(f"{label} entries must be positive")


def make_arm_gripper_mjcf(
    *,
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seg_lengths: tuple[float, ...] = (0.10, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08),
    seg_radii: tuple[float, ...] = (0.04, 0.035, 0.03, 0.028, 0.025, 0.022, 0.020),
    seg_masses: tuple[float, ...] = (0.6, 0.5, 0.5, 0.4, 0.3, 0.2, 0.15),
    joint_axes: tuple[str, ...] = ("z", "y", "z", "y", "z", "y", "z"),
    joint_damping: float = 0.20,
    joint_armature: float = 0.005,
    joint_range_deg: tuple[float, float] = (-170.0, 170.0),
    finger_length: float = 0.06,
    finger_thickness: float = 0.012,
    finger_mass: float = 0.05,
    finger_open: float = 0.04,
    finger_range: tuple[float, float] = (0.0, 0.04),
    finger_damping: float = 0.5,
    name: str = "arm7_gripper",
) -> str:
    _check_len("base_pos", base_pos, 3)
    _check_len("seg_lengths", seg_lengths, N_ARM_DOFS)
    _check_len("seg_radii", seg_radii, N_ARM_DOFS)
    _check_len("seg_masses", seg_masses, N_ARM_DOFS)
    _check_len("joint_axes", joint_axes, N_ARM_DOFS)
    _check_len("joint_range_deg", joint_range_deg, 2)
    _check_len("finger_range", finger_range, 2)
    _check_positive("seg_lengths", seg_lengths)
    _check_positive("seg_radii", seg_radii)
    _check_positive("seg_masses", seg_masses)
    _check_positive("finger geometry", (finger_length, finger_thickness, finger_mass, finger_open))
    if joint_range_deg[0] >= joint_range_deg[1]:
        raise ValueError("joint_range_deg must be increasing")
    if finger_range[0] > finger_range[1]:
        raise ValueError("finger_range must be increasing")

    colors = [
        "0.85 0.24 0.20 1",
        "0.20 0.44 0.82 1",
        "0.24 0.62 0.40 1",
        "0.86 0.58 0.18 1",
        "0.45 0.34 0.72 1",
        "0.18 0.60 0.66 1",
        "0.76 0.32 0.46 1",
    ]

    bodies_open = []
    bodies_close = []
    for i, body_name in enumerate(BODY_NAMES_ARM):
        body_pos_z = 0.0 if i == 0 else seg_lengths[i - 1]
        halflen = seg_lengths[i] * 0.5
        axis = _axis_vec(joint_axes[i])
        bodies_open.append(
            f'            <body name="{body_name}" pos="0 0 {_fmt(body_pos_z)}">\n'
            f'                <joint name="J{i + 1}" type="hinge" axis="{axis}" limited="true" '
            f'range="{_fmt(joint_range_deg[0])} {_fmt(joint_range_deg[1])}" '
            f'damping="{_fmt(joint_damping)}" armature="{_fmt(joint_armature)}"/>\n'
            f'                <geom name="{body_name}_capsule" type="capsule" pos="0 0 {_fmt(halflen)}" '
            f'size="{_fmt(seg_radii[i])} {_fmt(halflen)}" mass="{_fmt(seg_masses[i])}" '
            f'contype="0" conaffinity="0" rgba="{colors[i]}"/>'
        )
        bodies_close.append("            </body>")

    palm_x = max(0.018, finger_thickness)
    palm_y = finger_open + finger_thickness
    palm_z = max(0.008, finger_thickness * 0.5)
    finger_half = finger_thickness * 0.5
    finger_center_y = finger_open + finger_half
    marker_radius = min(0.004, max(0.002, finger_thickness * 0.25))

    palm_block = (
        f'            <body name="{BODY_NAME_PALM}" pos="0 0 {_fmt(seg_lengths[-1])}">\n'
        f'                <geom name="palm_box" type="box" pos="0 0 0" '
        f'size="{_fmt(palm_x)} {_fmt(palm_y)} {_fmt(palm_z)}" mass="0.08" '
        f'contype="0" conaffinity="0" rgba="0.25 0.25 0.25 1"/>\n'
        f'                <body name="{TCP_MARKER_BODY}" pos="0 0 {_fmt(finger_length)}">\n'
        f'                    <geom name="{TCP_MARKER_BODY}" type="sphere" size="{_fmt(marker_radius)}" '
        f'mass="0.000001" contype="0" conaffinity="0" rgba="1 0 0 1"/>\n'
        f'                </body>\n'
        f'                <body name="{FINGER_BODIES[0]}" pos="0 {_fmt(finger_center_y)} 0">\n'
        f'                    <joint name="finger_left_slide" type="slide" axis="0 -1 0" limited="true" '
        f'range="{_fmt(finger_range[0])} {_fmt(finger_range[1])}" '
        f'damping="{_fmt(finger_damping)}" armature="0.001"/>\n'
        f'                    <geom name="finger_left_box" type="box" pos="0 0 {_fmt(finger_length * 0.5)}" '
        f'size="{_fmt(finger_half)} {_fmt(finger_half)} {_fmt(finger_length * 0.5)}" '
        f'mass="{_fmt(finger_mass)}" contype="0" conaffinity="0" rgba="0.65 0.65 0.65 1"/>\n'
        f'                </body>\n'
        f'                <body name="{FINGER_BODIES[1]}" pos="0 -{_fmt(finger_center_y)} 0">\n'
        f'                    <joint name="finger_right_slide" type="slide" axis="0 1 0" limited="true" '
        f'range="{_fmt(finger_range[0])} {_fmt(finger_range[1])}" '
        f'damping="{_fmt(finger_damping)}" armature="0.001"/>\n'
        f'                    <geom name="finger_right_box" type="box" pos="0 0 {_fmt(finger_length * 0.5)}" '
        f'size="{_fmt(finger_half)} {_fmt(finger_half)} {_fmt(finger_length * 0.5)}" '
        f'mass="{_fmt(finger_mass)}" contype="0" conaffinity="0" rgba="0.65 0.65 0.65 1"/>\n'
        f'                </body>\n'
        f'            </body>'
    )

    nested_body = "\n".join(bodies_open) + "\n" + palm_block + "\n" + "\n".join(bodies_close)

    return (
        f'<mujoco model="{name}">\n'
        f'    <compiler angle="degree" autolimits="true"/>\n'
        f'    <worldbody>\n'
        f'        <body name="arm_base" pos="{_fmt_vec(base_pos)}">\n'
        f'            <geom name="arm_base_box" type="box" pos="0 0 0" '
        f'size="0.05 0.05 0.012" mass="0.05" contype="0" conaffinity="0" rgba="0.12 0.12 0.12 1"/>\n'
        f'{nested_body}\n'
        f'        </body>\n'
        f'    </worldbody>\n'
        f'</mujoco>\n'
    )


if __name__ == "__main__":
    print(make_arm_gripper_mjcf())
