"""State adapter for IsaacLab Newton state access.

This module provides a lightweight adapter that handles object state access
using Newton-native xyzw quaternions, matching holosoma's task-side convention.
"""

from __future__ import annotations

import torch


class IsaacSimStateAdapter:
    """Lightweight adapter for IsaacLab Newton state access.

    This adapter provides a clean interface for object state management while
    keeping quaternions in Newton-native xyzw order.

    Parameters
    ----------
    device : torch.device
        Device for tensor operations
    object_registry : ObjectRegistry
        Registry for object index resolution
    scene : InteractiveScene
        IsaacLab scene containing rigid objects
    robot_states : RootStatesProxy
        Robot states proxy exposing xyzw root states
    """

    def __init__(self, device: torch.device, object_registry, scene, robot, robot_states):
        self.device = device
        self._object_registry = object_registry
        self._scene = scene
        self._robot = robot
        self._robot_states = robot_states
        self._objects_dirty = False

    def resolve_indices(self, indices: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Resolve flat indices to (object_name, env_ids) pairs.

        Parameters
        ----------
        indices : torch.Tensor
            Flat tensor indices to resolve

        Returns
        -------
        list[tuple[str, torch.Tensor]]
            List of (object_name, env_ids) pairs
        """
        return self._object_registry.resolve_indices(indices)

    def get_object_states(self, obj_name: str, env_ids: torch.Tensor) -> torch.Tensor:
        """Get object states in Newton-native xyzw format.

        Parameters
        ----------
        obj_name : str
            Name of the object to get states for
        env_ids : torch.Tensor
            Environment IDs to query

        Returns
        -------
        torch.Tensor
            Object states in xyzw format, shape [len(env_ids), 13]

        Raises
        ------
        ValueError
            If object type is unknown
        """
        obj_type = self._object_registry.get_object_type(obj_name)

        if obj_type == "robot":
            return self._robot_states[env_ids]

        elif obj_type == "individual":
            rigid_object = self._scene.rigid_objects[obj_name]
            return rigid_object.data.root_state_w.torch[env_ids].clone()

        elif obj_type == "scene":
            scene_collection = self._scene.rigid_objects["usd_scene_objects"]
            object_index = self._object_registry.get_scene_position(obj_name)
            return scene_collection.data.body_state_w.torch[env_ids, object_index].clone()

        else:
            raise ValueError(f"Unknown object type '{obj_type}' for object '{obj_name}'")

    def write_object_states(self, obj_name: str, states: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Write object states in Newton-native xyzw format.

        Parameters
        ----------
        obj_name : str
            Name of the object to update
        states : torch.Tensor
            New states in xyzw format, shape [len(env_ids), 13]
        env_ids : torch.Tensor
            Environment IDs to update
        """
        obj_type = self._object_registry.get_object_type(obj_name)

        if obj_type == "robot":
            # NOTE: Intentionally does NOT apply env origins offsets for backwards compatibilty
            #       with existing robot root state setters
            self._robot.write_root_pose_to_sim_index(root_pose=states[:, :7], env_ids=env_ids)
            self._robot.write_root_velocity_to_sim_index(root_velocity=states[:, 7:], env_ids=env_ids)

        elif obj_type == "individual":
            rigid_object = self._scene.rigid_objects[obj_name]
            # For now, do NOT apply env origins as WBT does this itself. We need to update WBT first
            # before uncommenting this.
            # states[:, 0:3] += self._scene.env_origins[env_ids]  # Apply environment origins
            rigid_object.write_root_pose_to_sim_index(root_pose=states[:, :7], env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim_index(root_velocity=states[:, 7:], env_ids=env_ids)

        elif obj_type == "scene":
            scene_collection = self._scene.rigid_objects["usd_scene_objects"]
            object_index = self._object_registry.get_scene_position(obj_name)

            converted_states = states.clone()
            converted_states[:, 0:3] += self._scene.env_origins[env_ids]  # Apply environment origins

            # Update the collection's state tensor
            current_states = scene_collection.data.body_state_w.torch[env_ids].clone()
            current_states[:, object_index] = converted_states
            scene_collection.write_body_pose_to_sim_index(body_poses=current_states[..., :7], env_ids=env_ids)
            scene_collection.write_body_velocity_to_sim_index(body_velocities=current_states[..., 7:], env_ids=env_ids)

        else:
            raise ValueError(f"Unknown object type '{obj_type}' for object '{obj_name}'")

        # Mark states as dirty for batch synchronization
        self.mark_dirty()

    def is_dirty(self) -> bool:
        """Check if states have been modified and need synchronization.

        Returns
        -------
        bool
            True if states are dirty and need sync
        """
        return self._objects_dirty

    def clear_dirty(self) -> None:
        """Clear the dirty flag after synchronization."""
        self._objects_dirty = False

    def mark_dirty(self) -> None:
        """Explicitly mark states as dirty."""
        self._objects_dirty = True
