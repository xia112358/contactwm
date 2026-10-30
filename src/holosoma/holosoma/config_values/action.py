"""Default action manager configurations."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

none = None

g1_29dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=1.0,
            clip=None,
        ),
    }
)

DEFAULTS = {
    "none": none,
    "g1_29dof_joint_pos": g1_29dof_joint_pos,
}
