from __future__ import annotations

import torch
from loguru import logger

class AllRootStatesProxy:
    """Direct routing proxy using StateAdapter for unified state access.

    This proxy provides a unified interface for accessing object states across
    different object types (robots, scene objects, individual objects) using
    a lightweight state adapter for index resolution and routing.

    Parameters
    ----------
    state_adapter : IsaacSimStateAdapter
        The state adapter providing object state access in simulator-native format.

    Attributes
    ----------
    _adapter : IsaacSimStateAdapter
        Reference to the state adapter instance.
    """

    def __init__(self, state_adapter):
        self._adapter = state_adapter

    def __getitem__(self, indices):
        """Get states using StateAdapter resolution.

        Parameters
        ----------
        indices : torch.Tensor or tuple
            Tensor indices or tuple of (tensor_indices, column_slice).

        Returns
        -------
        torch.Tensor
            Object states for the specified indices.

        Raises
        ------
        KeyError
            If no objects are found for the given indices.
        """
        # Handle different index types
        if isinstance(indices, tuple):
            tensor_indices, column_slice = indices
        else:
            tensor_indices, column_slice = indices, slice(None)

        if not isinstance(tensor_indices, torch.Tensor):
            tensor_indices = torch.tensor(tensor_indices, device=self._adapter.device)

        resolved_objects = self._adapter.resolve_indices(tensor_indices)
        if not resolved_objects:
            raise KeyError(f"No objects found for indices: {tensor_indices}")

        # iterate over all objects and get views into their state tensors
        results = []
        for obj_name, env_ids in resolved_objects:
            obj_states = self._adapter.get_object_states(obj_name, env_ids)
            results.append(obj_states[:, column_slice])

        return torch.cat(results, dim=0) if len(results) > 1 else results[0]

    def __setitem__(self, indices, values):
        """Setters to write directly into tensor. Routes to adapter functions."""

        if isinstance(indices, tuple):
            tensor_indices, column_slice = indices
        else:
            tensor_indices, column_slice = indices, slice(None)

        if not isinstance(tensor_indices, torch.Tensor):
            tensor_indices = torch.tensor(tensor_indices, device=self._adapter.device)

        resolved_objects = self._adapter.resolve_indices(tensor_indices)
        if not resolved_objects:
            raise KeyError(f"No objects found for indices: {tensor_indices}")

        # Route to write method via adapter
        values_offset = 0
        for obj_name, env_ids in resolved_objects:
            num_envs_for_obj = len(env_ids)
            obj_values = values[values_offset : values_offset + num_envs_for_obj]
            values_offset += num_envs_for_obj
            self._adapter.write_object_states(obj_name, obj_values, env_ids)

        # Dirty flag is automatically set in adapter.write_object_states()

    @property
    def shape(self):
        """Return shape of the unified tensor (for compatibility)"""
        total_actors = len(self._adapter._object_registry.objects) * self._adapter._object_registry.num_envs
        return torch.Size([total_actors, 13])

    @property
    def device(self):
        """Return device of the unified tensor"""
        return self._adapter.device

    @property
    def dtype(self):
        """Return dtype of the unified tensor"""
        return torch.float32

    def clone(self):
        """Return a clone of the unified tensor (for compatibility)"""
        resolved_objects = self._adapter._object_registry.resolved_objects()

        if not resolved_objects:
            return torch.empty(0, 13, device=self.device, dtype=self.dtype)

        results = []
        for obj_name, env_ids in resolved_objects:
            obj_states = self._adapter.get_object_states(obj_name, env_ids)
            results.append(obj_states)

        return torch.cat(results, dim=0) if len(results) > 1 else results[0]


class RootStatesProxy:
    """Wrapper for Newton root states tensor.

    IsaacLab Newton exposes root state quaternions in xyzw order, matching
    holosoma's task-side convention.  Keep this proxy as a compatibility layer
    for older call sites, but do not reorder quaternion columns.

    Parameters
    ----------
    tensor_xyzw : torch.Tensor
        Root states tensor with quaternions in xyzw format.

    Attributes
    ----------
    tensor_xyzw : torch.Tensor
        Root states tensor with quaternions in xyzw format.
    """

    def __init__(self, tensor_xyzw: torch.Tensor):
        self.reset(tensor_xyzw)

    def reset(self, tensor_xyzw: torch.Tensor):
        self.tensor_xyzw = tensor_xyzw
        # Backwards-compatible alias for code that still references the old
        # PhysX-era name. In the Newton-only copy this is also xyzw.
        self.tensor_wxyz = self.tensor_xyzw

    def __getitem__(self, index):
        """Get tensor values in xyzw quaternion format.

        Parameters
        ----------
        index : int, slice, or tensor
            Index for tensor access.

        Returns
        -------
        torch.Tensor
            Tensor values with quaternions in xyzw format.
        """
        return self.tensor_xyzw[index]

    def __setitem__(self, index, value_xyzw):
        """Set tensor values from xyzw quaternion format.

        Parameters
        ----------
        index : int, slice, or tensor
            Index for tensor access.
        value_xyzw : torch.Tensor
            Values to set with quaternions in xyzw format.
        """
        self.tensor_xyzw[index] = value_xyzw
        self.tensor_wxyz = self.tensor_xyzw

    def _get_xyzw(self, env_ids=None):
        """Get tensor in Newton-native xyzw quaternion format.

        Parameters
        ----------
        env_ids : torch.Tensor, optional
            Environment IDs to select, by default None (returns all).

        Returns
        -------
        torch.Tensor
            Tensor with quaternions in xyzw format.
        """
        if env_ids is None:
            return self.tensor_xyzw
        return self.tensor_xyzw[env_ids]

    def _get_wxyz(self, env_ids=None):
        """Compatibility alias; returns Newton-native xyzw in this copy."""
        return self._get_xyzw(env_ids)
