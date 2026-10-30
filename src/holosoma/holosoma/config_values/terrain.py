from holosoma.config_types.terrain import MeshType, TerrainManagerCfg, TerrainTermCfg

terrain_motion_matched = TerrainManagerCfg(
    terrain_term=TerrainTermCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        mesh_type=MeshType.LOAD_OBJ,
        func="holosoma.managers.terrain.terms.locomotion:TerrainLocomotion",
        terrain_length=8.0,
        terrain_width=8.0,
        num_rows=4,
        num_cols=4,
        motion_matched_manifest="",
    )
)

DEFAULTS = {
    "terrain_motion_matched": terrain_motion_matched,
}
