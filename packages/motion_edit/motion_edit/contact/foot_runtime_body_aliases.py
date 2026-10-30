from __future__ import annotations

import importlib


_INSTALLED = False


def install_split_foot_runtime_body_aliases() -> None:
    """Allow canonical foot anchors to see all fixed sole-sphere runtime bodies.

    Split anchors retain ``body=left_foot/right_foot`` and carry heel/toe identity
    in metadata. Runtime body matching therefore needs the whole sole body
    family; strict raw shape IDs perform the actual heel/toe separation.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    legacy = importlib.import_module("motion_edit.contact.newton_bindings")
    groups = []
    for group in legacy._BODY_ALIAS_GROUPS:
        values = tuple(str(item) for item in group)
        if "left_foot" in values:
            groups.append(
                (
                    *values,
                    "left_ankle_intermediate_1_link",
                    "left_ankle_roll_sphere_1_link",
                    "left_ankle_roll_sphere_2_link",
                    "left_ankle_roll_sphere_3_link",
                    "left_ankle_roll_sphere_4_link",
                    "left_ankle_roll_sphere_5_link",
                )
            )
        elif "right_foot" in values:
            groups.append(
                (
                    *values,
                    "right_ankle_intermediate_1_link",
                    "right_ankle_roll_sphere_1_link",
                    "right_ankle_roll_sphere_2_link",
                    "right_ankle_roll_sphere_3_link",
                    "right_ankle_roll_sphere_4_link",
                    "right_ankle_roll_sphere_5_link",
                )
            )
        else:
            groups.append(values)
    legacy._BODY_ALIAS_GROUPS = tuple(
        tuple(dict.fromkeys(group))
        for group in groups
    )
    _INSTALLED = True
