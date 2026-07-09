"""Minimal Genesis Shadow Hand URDF loading smoke test.

Run:
    conda run -n genesis --no-capture-output python scripts/archive/shadow_hand_min_load.py
"""

import genesis as gs


def main():
    gs.init(backend=gs.cpu, precision="32", logging_level="warning")

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())

    hand = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
            fixed=True,
            pos=(0.0, 0.0, 0.4),
        )
    )

    scene.build()

    print("[shadow_hand] build ok")
    print(f"[shadow_hand] n_dofs={hand.n_dofs}")
    print(f"[shadow_hand] n_links={hand.n_links}")
    print(f"[shadow_hand] n_geoms={hand.n_geoms}")
    print("[shadow_hand] first joints:")
    for joint in hand.joints[:10]:
        print(f"  - {joint.name}")

    for _ in range(5):
        scene.step()

    print("[shadow_hand] step ok")


if __name__ == "__main__":
    main()
