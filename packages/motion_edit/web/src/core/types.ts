export type Vec3 = [number, number, number];

export type MotionItem = {
  motion_id: string;
  label: string;
  provenance: string;
  ready: boolean;
};

export type RecentMotion = {
  label: string;
  motion_id: string;
  provenance: string;
  active: boolean;
  last_opened_at: string | null;
};

export type Anchor = {
  anchor_id: string;
  body: string;
  start_frame: number;
  end_frame: number;
  world_position: Vec3 | null;
  surface_id: string | null;
  object_id: string | null;
  surface_coordinates: { u: number; v: number } | null;
  metadata: Record<string, unknown>;
  surface_origin: Vec3 | null;
  surface_normal: Vec3 | null;
  surface_tangent_u: Vec3 | null;
  surface_tangent_v: Vec3 | null;
};

export type Surface = {
  surface_id: string;
  origin: Vec3;
  normal: Vec3;
  tangent_u: Vec3;
  tangent_v: Vec3;
  bounds: { u: [number, number]; v: [number, number] };
  metadata: Record<string, unknown>;
};

export type Session = {
  review?: Review;
  motion_id: string;
  read_only: boolean;
  capabilities: {
    playback: boolean;
    edit_contacts: boolean;
    generate: boolean;
  };
  provenance: string;
  source_motion_id: string;
  fps: number;
  qpos: number[][];
  joint_names: string[];
  robot_urdf_url: string;
  terrain_obj_url: string | null;
  graph: {
    anchors: Anchor[];
    transitions: Array<{ start_frame: number; end_frame: number; metadata: Record<string, unknown> }>;
  };
  surfaces: Surface[];
  contact_force: {
    mask_status?: string;
    part_order: string[];
    forces: Vec3[][];
    masks: boolean[][];
    positions: Vec3[][];
    position_valid: boolean[][];
    sample_points: Vec3[][];
    sample_forces: Vec3[][];
  };
};

export type ReviewRecord = {
  label: string; method: string; iteration: string; q: number[];
  normal_starts: Vec3[]; normals: Vec3[];
  diagnostics: Record<string, unknown>;
};
export type Review = {
  collection?: { path: string; terrain_override: string | null; case_index: number; reference_index: number; cases: {index:number;label:string}[] };
  comparison?: { label: string; qpos: number[][]; fps: number; joint_names: string[]; terrain_obj_url: string | null };
  segments?: { start: number; end: number; label: string }[];
  depth_mm?: number[];
  support_frames?: {part:number;load_known:boolean;load_state:string;normal_force_n:number|null;loaded_speed_cm_s:number|null}[][];
  support_assessment_schema?: string;
  actual_support_status?: string;
  kind: 'motion' | 'report' | 'support_motion' | 'support_slip'; source: string; notice: string;
  slip_frames?: { part: number; surface: number; body: string; position: Vec3; delta: Vec3; motion_cm: number }[][];
  reference?: string; records?: ReviewRecord[];
  phases?: { start: number; end: number; part: string; aggregate: Vec3[]; rollouts: Vec3[][]; aggregate_net_cm: number; rollout_median_net_cm: number }[];
};
