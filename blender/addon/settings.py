"""Blender properties shared by the interactive add-on and the scene evaluator."""
import bpy

from . import grid_view


COMMON_CONFIG = (
    ("cfl", "f"), ("max_time_step", "f"), ("buoyancy", "f"), ("smoke_weight", "f"),
    ("up", "3f"), ("wind", "3f"), ("cooling", "f"), ("dissipation", "f"),
    ("pressure_relative_tolerance", "f"), ("divergence_tolerance", "f"),
    ("pressure_max_iterations", "i"),
)
CONFIG = {
    "DENSE": (("cell_size", "f"), ("dense_origin", "3f"), ("dense_resolution", "3i")) + COMMON_CONFIG,
    "SPARSE": (("cell_size", "f"), ("initial_blocks", "i"), ("amr_levels", "i"), ("amr_padding", "i"),
            ("amr_adapt_interval", "i"), ("amr_coarsen_delay", "i"), ("keep_alive_adaptations", "i"),
            ("amr_smoke_error", "f"), ("amr_temperature_error", "f"), ("amr_velocity_error", "f"), ("amr_relative_error", "f"),
            ("smoke_threshold", "f"), ("temperature_threshold", "f"), ("velocity_threshold", "f")) + COMMON_CONFIG,
}
OBJECT_INPUTS = ("role", "active", "smoke_rate", "temperature_rate", "velocity_rate", "velocity", "velocity_space", "inherit_velocity", "surface_offset")


def enable_timeline(self, context):
    from . import runtime
    scene = self.id_data
    if self.enabled:
        scene.sync_mode = "NONE"
        scene.render.use_lock_interface = True
    else:
        runtime.close_session(scene.as_pointer())


def input_changed(self, context):
    from . import runtime
    owner = self.id_data
    if isinstance(owner, bpy.types.Scene):
        runtime.invalidate(owner)
        grid_view.refresh(self, context)
    else:
        for scene in bpy.data.scenes:
            if owner.name in scene.objects:
                runtime.invalidate(scene)


def volume_object(self, obj):
    return obj.type == "VOLUME" and obj.name in self.id_data.objects


def output_changed(self, context):
    from . import runtime
    record = runtime.records.get(self.id_data.as_pointer())
    if not runtime.busy and record is not None:
        runtime.display(self.id_data, record, self.id_data.frame_current)


class PYRO_PG_scene(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Timeline Simulation", description="Compute missing frames when the timeline advances; disable to play only cached frames and release the solver process", default=False, options=set(), update=enable_timeline)
    algorithm: bpy.props.EnumProperty(name="Algorithm", items=[("SPARSE", "Sparse (AMR)", "Dynamic sparse domain with adaptive-resolution leaf blocks"), ("DENSE", "Dense", "Fixed axis-aligned grid with open boundaries")], default="SPARSE", options=set(), update=input_changed)
    cell_size: bpy.props.FloatProperty(name="Voxel Size", description="Cell width in Blender world units; the finest level for AMR. Smaller cells increase detail and simulation cost", default=0.025, min=0.0001, subtype="DISTANCE", options=set(), update=input_changed)
    dense_origin: bpy.props.FloatVectorProperty(name="Domain Minimum", description="World-space minimum corner of the fixed dense domain", default=(-1.6, -1.6, -0.4), subtype="TRANSLATION", options=set(), update=input_changed)
    dense_resolution: bpy.props.IntVectorProperty(name="Resolution", description="Voxel counts along world X, Y and Z; domain size is resolution times voxel size", default=(128, 128, 192), min=1, options=set(), update=input_changed)
    initial_blocks: bpy.props.IntProperty(name="Initial Leaf Capacity", description="Initial reservation across all AMR levels; each leaf holds 8 x 8 x 8 cells. Storage grows automatically as needed", default=4096, min=128, options=set(), update=input_changed)
    amr_levels: bpy.props.IntProperty(name="Resolution Levels", description="L0 is the finest; each next level doubles voxel size", default=3, min=1, max=8, options=set(), update=input_changed)
    amr_padding: bpy.props.IntProperty(name="Coarse Padding", description="Pressure-domain padding measured in coarsest-level blocks", default=1, min=1, options=set(), update=input_changed)
    amr_adapt_interval: bpy.props.IntProperty(name="Adapt Interval", description="Number of solver substeps between AMR topology adaptations", default=2, min=1, options=set(), update=input_changed)
    amr_coarsen_delay: bpy.props.IntProperty(name="Coarsen Delay", description="Consecutive quiet adaptations required before merging fine leaves", default=4, min=1, options=set(), update=input_changed)
    amr_smoke_error: bpy.props.FloatProperty(name="Smoke Error", description="Absolute smoke reconstruction error used by AMR refinement and coarsening", default=0.0003125, min=1e-8, precision=7, options=set(), update=input_changed)
    amr_temperature_error: bpy.props.FloatProperty(name="Temperature Error", description="Absolute temperature reconstruction error used by AMR refinement and coarsening", default=0.00125, min=1e-8, precision=7, options=set(), update=input_changed)
    amr_velocity_error: bpy.props.FloatProperty(name="Velocity Error", description="Absolute velocity reconstruction error used by AMR refinement and coarsening", default=0.001875, min=1e-8, precision=7, options=set(), update=input_changed)
    amr_relative_error: bpy.props.FloatProperty(name="Relative Error", description="Relative field reconstruction error used together with absolute AMR thresholds", default=0.003125, min=0, precision=7, options=set(), update=input_changed)
    keep_alive_adaptations: bpy.props.IntProperty(name="Keep Alive (adaptations)", default=4, min=0, options=set(), update=input_changed)
    smoke_threshold: bpy.props.FloatProperty(name="Smoke Activation", default=1e-5, min=0, precision=6, options=set(), update=input_changed)
    temperature_threshold: bpy.props.FloatProperty(name="Temperature Activation", default=1e-4, min=0, precision=6, options=set(), update=input_changed)
    velocity_threshold: bpy.props.FloatProperty(name="Velocity Activation", default=0.025, min=0, options=set(), update=input_changed)
    cfl: bpy.props.FloatProperty(name="CFL", description="Maximum advective distance per substep relative to cell width; smaller values require more substeps", default=0.75, min=0.01, max=1, options=set(), update=input_changed)
    max_time_step: bpy.props.FloatProperty(name="Maximum Step", description="Maximum solver substep duration in seconds; the solver can choose shorter steps", default=1/48, min=0.00001, subtype="TIME", options=set(), update=input_changed)
    buoyancy: bpy.props.FloatProperty(name="Buoyancy", description="Acceleration along Up per unit of temperature", default=1, options=set(), update=input_changed)
    smoke_weight: bpy.props.FloatProperty(name="Smoke Weight", description="Acceleration opposite Up per unit of smoke density", default=0.05, options=set(), update=input_changed)
    up: bpy.props.FloatVectorProperty(name="Up Direction", description="World-space vector used by buoyancy and smoke weight", default=(0, 0, 1), options=set(), update=input_changed)
    wind: bpy.props.FloatVectorProperty(name="Background Wind", description="Ambient world-space velocity in Blender units per second", default=(0, 0, 0), options=set(), update=input_changed)
    cooling: bpy.props.FloatProperty(name="Cooling", description="Exponential temperature decay rate per second", default=0.1, min=0, options=set(), update=input_changed)
    dissipation: bpy.props.FloatProperty(name="Dissipation", description="Exponential smoke density decay rate per second", default=0.01, min=0, options=set(), update=input_changed)
    pressure_relative_tolerance: bpy.props.FloatProperty(name="Relative Tolerance", description="Relative pressure-solve residual target; smaller values request a more accurate solve", default=1e-5, min=1e-8, precision=7, options=set(), update=input_changed)
    divergence_tolerance: bpy.props.FloatProperty(name="Divergence Tolerance", description="Absolute divergence target for pressure projection", default=2e-4, min=1e-7, precision=7, options=set(), update=input_changed)
    pressure_max_iterations: bpy.props.IntProperty(name="Maximum Iterations", description="Pressure iteration limit per substep; frame diagnostics sum all substeps", default=300, min=8, options=set(), update=input_changed)
    animation_tolerance: bpy.props.FloatProperty(name="Motion Error (voxels)", description="Maximum transform interpolation error in finest-voxel units; reduces the sampling interval for curved motion", default=0.05, min=0.001, max=0.5, options=set(), update=input_changed)
    checkpoint_interval: bpy.props.IntProperty(name="Checkpoint Interval", description="Save a resumable solver checkpoint every N simulated frames; also saved with the .blend and when releasing the solver", default=24, min=1, options=set())
    cache_root: bpy.props.StringProperty(name="Cache Folder", description="Parent folder for new simulation caches; // is relative to a saved .blend", subtype="DIR_PATH", default="//pyro_cache/", options=set())
    volume: bpy.props.PointerProperty(name="Output Volume", description="Volume object receiving the world-space density cache; keep its transform at identity", type=bpy.types.Object, poll=volume_object, update=output_changed)
    object_list_index: bpy.props.IntProperty(default=0, min=0, options={"SKIP_SAVE"})
    grid_show: bpy.props.BoolProperty(name="Show Solver Grid", default=False, options=set(), update=grid_view.refresh)
    grid_color: bpy.props.EnumProperty(name="Color", items=[("LEVEL", "Resolution Level", "L0 is finest; each level doubles voxel size"), ("CHANGES", "Frame Changes", "Compare with the previous output frame")], default="LEVEL", options=set(), update=grid_view.refresh)
    grid_slice: bpy.props.EnumProperty(name="Slice Axis", items=[("NONE", "Full Grid", ""), ("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", "")], options=set(), update=grid_view.refresh)
    grid_position: bpy.props.FloatProperty(name="Slice Position", description="World coordinate of the slicing plane; intersects all visible resolution levels", default=0, subtype="DISTANCE", options=set(), update=grid_view.refresh)
    grid_level: bpy.props.IntProperty(name="AMR Level (-1 = All)", default=-1, min=-1, max=7, options=set(), update=grid_view.refresh)
    grid_xray: bpy.props.BoolProperty(name="X-Ray", default=False, options=set(), update=grid_view.redraw)
    grid_opacity: bpy.props.FloatProperty(name="Opacity", default=0.7, min=0, max=1, subtype="FACTOR", options=set(), update=grid_view.refresh)
    grid_removed: bpy.props.BoolProperty(name="Show Removed Blocks", default=True, options=set(), update=grid_view.refresh)
    grid_cells: bpy.props.BoolProperty(name="Selected Block: 8 x 8 x 8 Cells", default=False, options=set(), update=grid_view.refresh)
    cache_directory: bpy.props.StringProperty(subtype="DIR_PATH", options=set())
    stale: bpy.props.BoolProperty(default=False, options=set())
    cache_start: bpy.props.IntProperty(default=1, options=set())
    cache_frames: bpy.props.IntProperty(default=0, options=set())
    error: bpy.props.StringProperty(options=set())


class PYRO_PG_object(bpy.types.PropertyGroup):
    role: bpy.props.EnumProperty(name="Pyro Role", items=[("NONE", "None", "No simulation role; keep the mesh"), ("EMITTER", "Source", "Emit inside this closed mesh"), ("COLLIDER", "Collider", "One-way solid boundary")], options=set(), update=input_changed)
    active: bpy.props.BoolProperty(name="Active", description="Participate in simulation at this time; can be keyframed independently of render visibility", default=True, update=input_changed)
    smoke_rate: bpy.props.FloatProperty(name="Smoke Rate", description="Smoke density added per second inside the source", default=1, min=0, update=input_changed)
    temperature_rate: bpy.props.FloatProperty(name="Temperature Rate", description="Temperature added per second inside the source; drives buoyancy", default=3, update=input_changed)
    velocity_rate: bpy.props.FloatProperty(name="Velocity Strength", description="Exponential relaxation rate per second toward the source velocity; zero disables velocity injection", default=4, min=0, update=input_changed)
    velocity: bpy.props.FloatVectorProperty(name="Velocity", description="Target source velocity in Blender units per second", default=(0, 0, 1.5), update=input_changed)
    velocity_space: bpy.props.EnumProperty(name="Velocity Space", items=[("LOCAL", "Local", "Rotate velocity with the emitter"), ("WORLD", "World", "World-space velocity")], options=set(), update=input_changed)
    inherit_velocity: bpy.props.FloatProperty(name="Inherit Motion", description="Amount of source transform velocity added to the injection velocity", default=1, min=0, update=input_changed)
    surface_offset: bpy.props.FloatProperty(name="Surface Offset", description="Signed collider surface offset in Blender world units; positive expands the solid", default=0, subtype="DISTANCE", update=input_changed)


def register():
    bpy.utils.register_class(PYRO_PG_object)
    bpy.utils.register_class(PYRO_PG_scene)
    bpy.types.Object.pyro = bpy.props.PointerProperty(type=PYRO_PG_object)
    bpy.types.Scene.pyro = bpy.props.PointerProperty(type=PYRO_PG_scene)


def unregister():
    del bpy.types.Scene.pyro
    del bpy.types.Object.pyro
    bpy.utils.unregister_class(PYRO_PG_scene)
    bpy.utils.unregister_class(PYRO_PG_object)
