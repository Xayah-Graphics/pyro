"""Shared scene and object configuration for Properties and the Pyro sidebar."""
import textwrap

import bpy
import numpy as np

from . import grid_view, runtime


def draw_hint(layout, context, text):
    width = max(18, int(context.region.width / (7 * context.preferences.system.ui_scale)) - 7)
    for line in textwrap.wrap(text, width=width):
        layout.label(text=line)


class PYRO_UL_objects(bpy.types.UIList):
    def filter_items(self, context, data, propname):
        objects = getattr(data, propname)
        flags = bpy.types.UI_UL_list.filter_items_by_name(self.filter_name, self.bitflag_filter_item, objects, reverse=self.use_filter_invert) if self.filter_name else [self.bitflag_filter_item] * len(objects)
        for index, obj in enumerate(objects):
            if obj.type != "MESH" or obj.pyro.role == "NONE":
                flags[index] = self.bitflag_item_never_show
        order = bpy.types.UI_UL_list.sort_items_by_name(objects) if self.use_filter_sort_alpha else []
        return flags, order

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        select = row.row(align=True)
        select.enabled = item.name in context.view_layer.objects and not item.hide_viewport
        select.operator("pyro.select_object", text=item.name, icon="MOD_FLUID" if item.pyro.role == "EMITTER" else "MOD_PHYSICS", emboss=False, depress=item == context.object).object_name = item.name
        row.prop(item.pyro, "active", text="")
        remove = row.operator("pyro.set_role", text="", icon="X", emboss=False)
        remove.object_name, remove.role = item.name, "NONE"


def draw_simulation(layout, context):
    scene, props = context.scene, context.scene.pyro
    lfm = props.algorithm == "LFM"
    key = scene.as_pointer()
    record = runtime.records.get(key)
    view = grid_view.views.get(key)
    stats = view["stats"] if view and not props.stale else None
    objects = runtime.inputs.participants(scene)
    sources = sum(obj.pyro.role == "EMITTER" for obj in objects)
    colliders = len(objects) - sources
    active_sources = sum(obj.pyro.role == "EMITTER" and obj.pyro.active for obj in objects)
    frontier = props.cache_start + props.cache_frames - 1
    needs_save = not bpy.data.filepath and props.cache_root.startswith("//")
    layout.use_property_split = True
    layout.use_property_decorate = False

    row = layout.row()
    row.enabled = props.enabled or not needs_save
    row.prop(props, "enabled")
    if props.stale:
        row = layout.row()
        row.alert = True
        row.label(text="Inputs changed — reset required", icon="ERROR")
    elif props.error:
        layout.label(text="Simulation error — see Diagnostics", icon="ERROR")
    elif not props.cache_frames:
        layout.label(text="Ready to simulate" if objects or lfm else "Add a source to begin", icon="INFO")
    else:
        layout.label(text="Timeline simulation enabled" if props.enabled else "Cache playback only", icon="PLAY" if props.enabled else "PAUSE")
    draw_hint(layout, context, f"Timeline: {scene.frame_current} | Computed through: {frontier if props.cache_frames else '—'}")
    if stats:
        draw_hint(layout, context, f"Viewing cache frame {stats['frame']} | {stats['time']:.3f} s")
    elif props.cache_frames and not props.stale:
        layout.label(text="No cached result at the current frame")
    if not active_sources and not lfm:
        layout.label(text="No source active at this frame", icon="INFO")
    if needs_save:
        draw_hint(layout, context, "Save the .blend or choose an absolute cache folder")
        row = layout.row()
        row.operator_context = "INVOKE_DEFAULT"
        row.operator("wm.save_as_mainfile", text="Save Project", icon="FILE_TICK")
    row = layout.row(align=True)
    row.operator("pyro.evaluate", text="Compute Current", icon="PLAY")
    row.operator("pyro.evaluate", text="Next Frame", icon="TRIA_RIGHT").next_frame = True
    layout.operator("pyro.reset", icon="FILE_REFRESH")

    header, body = layout.panel("pyro_objects", default_closed=False)
    header.label(text="Scene Objects")
    if body:
        body.label(text=f"Static colliders: {colliders}" if lfm else f"Sources: {sources} | Colliders: {colliders}")
        row = body.row(align=True)
        row.operator_context = "INVOKE_DEFAULT"
        if not lfm:
            row.operator("pyro.add_object", text="New Source", icon="ADD").role = "EMITTER"
        row.operator("pyro.add_object", text="New Collider", icon="ADD").role = "COLLIDER"
        body.label(text="New meshes are placed at the 3D cursor")
        row = body.row(align=True)
        row.enabled = context.mode == "OBJECT" and any(obj.type == "MESH" for obj in context.selected_objects)
        if not lfm:
            row.operator("pyro.set_role", text="Selected → Source").role = "EMITTER"
        row.operator("pyro.set_role", text="Selected → Collider").role = "COLLIDER"
        row = body.row()
        row.enabled = context.mode == "OBJECT" and any(obj.type == "MESH" and obj.pyro.role != "NONE" for obj in context.selected_objects)
        row.operator("pyro.set_role", text="Remove Selected Roles").role = "NONE"
        if objects:
            body.template_list("PYRO_UL_objects", "participants", scene, "objects", props, "object_list_index", rows=4)
            body.label(text="Click a name to select; X removes its role")
        else:
            body.label(text="Create a mesh, or assign an existing mesh")
        draw_hint(body, context, "LFM uses constant boundary inflow and static colliders" if lfm else "All sources and colliders share this scene's solver")

    header, body = layout.panel("pyro_solver", default_closed=False)
    header.label(text="Solver & Domain")
    if body:
        body.prop(props, "algorithm")
        body.prop(props, "cell_size", text="Finest Voxel Size" if props.algorithm == "SPARSE" else "Voxel Size")
        if props.algorithm != "SPARSE":
            body.prop(props, "dense_origin")
            body.prop(props, "dense_resolution")
            nx, ny, nz = props.dense_resolution
            draw_hint(body, context, f"Size: {nx*props.cell_size:.4g} × {ny*props.cell_size:.4g} × {nz*props.cell_size:.4g} BU")
            body.label(text=f"Domain cells: {nx*ny*nz:,}")
            body.label(text="Fixed domain; prescribed inflow/outflow" if lfm else "Fixed world-aligned domain; open boundaries")
        elif props.algorithm == "SPARSE":
            body.prop(props, "initial_blocks")
            draw_hint(body, context, "Leaf storage grows automatically as needed")
            body.prop(props, "amr_levels")
            draw_hint(body, context, f"Coarsest voxel: {props.cell_size*2**(props.amr_levels-1):.4g} BU")
            draw_hint(body, context, "Domain grows with smoke; local detail controls refinement")
        draw_hint(body, context, "Geometry and voxel sizes use Blender world units")
        body.prop(props, "grid_show", text="Preview Solver Grid")

    header, body = layout.panel("pyro_timing", default_closed=True)
    header.label(text="Time & Stepping")
    if body:
        body.prop(scene, "frame_start", text="Simulation Start")
        body.prop(scene, "frame_end", text="Playback End")
        body.prop(scene.render, "fps")
        body.prop(scene.render, "fps_base")
        draw_hint(body, context, f"Frame duration: {scene.render.fps_base/scene.render.fps:.6g} s")
        if lfm:
            body.prop(props, "lfm_cycle_steps")
            draw_hint(body, context, f"One complete cycle per frame; step = {scene.render.fps_base/scene.render.fps/props.lfm_cycle_steps:.6g} s")
        else:
            body.prop(props, "cfl")
            body.prop(props, "max_time_step")
            body.prop(props, "animation_tolerance")
        body.label(text="Start frame is the initialized flow" if lfm else "Start frame is the initial, empty state")
        draw_hint(body, context, "Future frames compute synchronously, including skipped frames")
        draw_hint(body, context, "Earlier frames display their cache; playback end sets the loop range")

    header, body = layout.panel("pyro_forces", default_closed=True)
    header.label(text="Wind Tunnel Inflow" if lfm else "Forces & Dissipation")
    if body:
        for name in (("wind",) if lfm else ("buoyancy", "smoke_weight", "up", "wind", "cooling", "dissipation")):
            body.prop(props, name)
        if lfm:
            draw_hint(body, context, "World-space velocity prescribed on opposite domain faces; no buoyancy or smoke emission")

    header, body = layout.panel("pyro_pressure", default_closed=True)
    header.label(text="Pressure Solve")
    if body:
        for name in ("pressure_relative_tolerance", "divergence_tolerance", "pressure_max_iterations"):
            body.prop(props, name)
        if props.algorithm == "SPARSE":
            body.prop(props, "amr_padding")

    if props.algorithm == "SPARSE":
        header, body = layout.panel("pyro_adaptation", default_closed=True)
        header.label(text="AMR Refinement & Activation")
        if body:
            for name in ("amr_adapt_interval", "amr_coarsen_delay", "amr_smoke_error", "amr_temperature_error", "amr_velocity_error", "amr_relative_error"):
                body.prop(props, name)
            draw_hint(body, context, "Lower reconstruction errors retain finer detail")
            for name in ("smoke_threshold", "temperature_threshold", "velocity_threshold"):
                body.prop(props, name)
            body.prop(props, "keep_alive_adaptations")

    header, body = layout.panel("pyro_output", default_closed=True)
    header.label(text="Vorticity Output & Material" if lfm else "Smoke Output & Material")
    if body:
        body.prop(props, "volume", text="Output Volume")
        volume = props.volume
        if volume is None:
            body.operator("pyro.create_volume", icon="OUTLINER_DATA_VOLUME")
            draw_hint(body, context, "Also created automatically on the first computation")
        else:
            row = body.row()
            row.enabled = volume.name in context.view_layer.objects and not volume.hide_viewport
            row.operator("pyro.select_object", text="Select Output Volume", icon="OUTLINER_DATA_VOLUME").object_name = volume.name
            body.prop(volume, "active_material", text="Material")
            material = volume.active_material
            output = next((node for node in material.node_tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None) if material else None
            shader = output.inputs["Volume"].links[0].from_node if output and output.inputs["Volume"].is_linked else None
            if shader and shader.type == "PRINCIPLED_VOLUME":
                for name, label in (("Density", "Display Scale"), ("Color", "Volume Color"), ("Anisotropy", "Anisotropy")):
                    row = body.row()
                    row.enabled = not shader.inputs[name].is_linked
                    row.prop(shader.inputs[name], "default_value", text=label)
            elif material:
                draw_hint(body, context, "Edit the custom volume material in the Shader Editor")
            else:
                body.operator("pyro.create_volume", text="Create Volume Material", icon="MATERIAL")
            draw_hint(body, context, "VDB field is in world space; keep the output transform at identity")
        if lfm:
            draw_hint(body, context, "Vorticity measures local rotation of the flow. Display scale affects appearance only")
        draw_hint(body, context, "Material changes do not require a simulation reset")

    header, body = layout.panel("pyro_cache", default_closed=True)
    header.label(text="Cache & Resume")
    if body:
        body.prop(props, "cache_root", text="New Cache Folder")
        body.prop(props, "checkpoint_interval")
        draw_hint(body, context, "Changing the folder applies after the next reset")
        if props.cache_frames:
            draw_hint(body, context, f"Cached range: {props.cache_start}–{frontier} ({props.cache_frames} frames)")
            body.progress(factor=min(1, props.cache_frames/max(1, scene.frame_end-props.cache_start+1)), text="Playback range cached")
        else:
            body.label(text="No cached frames")
        if record:
            checkpoint = record["manifest"]["checkpoint"]
            draw_hint(body, context, f"Resume checkpoint: {checkpoint}" if checkpoint >= record["manifest"]["start"] else "Resume checkpoint: not saved yet")
        if key in runtime.sessions:
            draw_hint(body, context, f"Resident solver frame: {runtime.sessions[key]['current']}")
        else:
            body.label(text="Solver process: stopped")
        if props.cache_directory:
            path = body.row()
            path.enabled = False
            path.prop(props, "cache_directory", text="Current Cache")
            body.operator("wm.path_open", text="Open Cache Folder", icon="FILE_FOLDER").filepath = props.cache_directory
        draw_hint(body, context, "Saving the .blend also saves a solver checkpoint")
        draw_hint(body, context, "Reset recycles this cache and returns to the start frame")

    header, body = layout.panel("pyro_diagnostics", default_closed=False)
    header.label(text="Frame Diagnostics")
    if body:
        if props.error:
            body.operator("pyro.copy_error", text="Copy Full Error", icon="COPYDOWN")
            for line in props.error.splitlines():
                row = body.column()
                row.alert = True
                draw_hint(row, context, line)
        if stats:
            draw_hint(body, context, f"Cached {stats['algorithm']} frame {stats['frame']} | Time {stats['time']:.6g} s")
            body.label(text=f"Solver: {stats['step_ms']:.2f} ms")
            draw_hint(body, context, f"GPU readback: {stats['export_ms']:.2f} ms | VDB: {stats['convert_ms']:.2f} ms")
            draw_hint(body, context, f"Substeps: {stats['substeps']} | Pressure iterations: {stats['pressure_iterations']}")
            draw_hint(body, context, f"Max divergence: {stats['divergence']:.4g} | Speed: {stats['max_speed']:.4g} BU/s")
            body.label(text=f"Fluid cells: {stats['cells']:,}")
            if stats["algorithm"] == "SPARSE":
                draw_hint(body, context, f"Leaves: {stats['blocks']:,} | Allocated capacity: {stats['capacity']:,}")
            draw_hint(body, context, f"Solver GPU allocation: {stats['bytes']/1048576:.1f} MiB")
            if stats["algorithm"] == "LFM":
                draw_hint(body, context, f"Max vorticity: {stats['vorticity_max']:.6g} /s")
            else:
                draw_hint(body, context, f"Smoke integral: {stats['smoke_integral']:.6g}")
            draw_hint(body, context, "Timings belong to the displayed cached frame")
        elif props.stale:
            draw_hint(body, context, "Cached diagnostics are outdated; reset to recompute")
        else:
            body.label(text="Compute a frame to see solver diagnostics")


def draw_object(layout, context):
    obj = context.object
    layout.use_property_split = True
    layout.use_property_decorate = True
    if obj is None or obj.type != "MESH":
        draw_hint(layout, context, "Select a mesh to configure a source or collider")
        return
    props = obj.pyro
    lfm = context.scene.pyro.algorithm == "LFM"
    layout.prop(obj, "name", text="Object")
    layout.prop(props, "role")
    if props.role == "NONE":
        row = layout.row(align=True)
        for role, label in (("EMITTER", "Make Source"), ("COLLIDER", "Make Collider")):
            if lfm and role == "EMITTER":
                continue
            action = row.operator("pyro.set_role", text=label)
            action.object_name, action.role = obj.name, role
        return
    layout.prop(props, "active")
    if context.scene.pyro.stale:
        layout.label(text="Inputs changed — reset required", icon="ERROR")
    if props.role == "EMITTER":
        if lfm:
            layout.label(text="LFM uses boundary inflow; remove the Source role", icon="ERROR")
            return
        header, body = layout.panel("pyro_emission", default_closed=False)
        header.label(text="Emission")
        if body:
            body.prop(props, "smoke_rate")
            body.prop(props, "temperature_rate")
            body.label(text="Emit inside the closed mesh volume")
        header, body = layout.panel("pyro_injection", default_closed=False)
        header.label(text="Source Velocity")
        if body:
            for name in ("velocity_space", "velocity", "velocity_rate", "inherit_velocity"):
                body.prop(props, name)
    else:
        layout.prop(props, "surface_offset")
        draw_hint(layout, context, "One-way collision; the solver does not move this object")
    header, body = layout.panel("pyro_object_transform", default_closed=True)
    header.label(text="Transform")
    if body:
        body.prop(obj, "location")
        rotation = {"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}.get(obj.rotation_mode, "rotation_euler")
        body.prop(obj, rotation, text="Rotation")
        body.prop(obj, "scale")
        body.prop(obj, "dimensions")
        draw_hint(body, context, "LFM requires static colliders; reset after edits" if lfm else "Transforms, source rates and Active can be keyframed")
    header, body = layout.panel("pyro_object_geometry", default_closed=True)
    header.label(text="Geometry & Display")
    if body:
        body.prop(obj, "data", text="Mesh")
        draw_hint(body, context, f"Base mesh: {len(obj.data.vertices):,} vertices / {len(obj.data.polygons):,} faces")
        draw_hint(body, context, "Use a closed mesh with outward-facing normals")
        draw_hint(body, context, "Static modifiers are supported" if lfm else "Static modifiers and transform animation are supported")
        body.label(text="Deforming meshes are not supported")
        body.prop(obj, "display_type")
        body.prop(obj, "hide_render", text="Hide Mesh in Renders")
        draw_hint(body, context, "Simulation participation is controlled by Active")


class PYRO_PT_simulation(bpy.types.Panel):
    bl_label = "Pyro Simulation"
    bl_idname = "PYRO_PT_simulation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pyro"

    def draw(self, context):
        draw_simulation(self.layout, context)


class PYRO_PT_scene(bpy.types.Panel):
    bl_label = "Pyro Simulation"
    bl_idname = "PYRO_PT_scene"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"

    def draw(self, context):
        draw_simulation(self.layout, context)


class PYRO_PT_active_object(bpy.types.Panel):
    bl_label = "Source / Collider"
    bl_idname = "PYRO_PT_active_object"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pyro"

    def draw(self, context):
        draw_object(self.layout, context)


class PYRO_PT_physics(bpy.types.Panel):
    bl_label = "Pyro"
    bl_idname = "PYRO_PT_physics"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "physics"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def draw(self, context):
        draw_object(self.layout, context)


class PYRO_PT_grid(bpy.types.Panel):
    bl_label = "Solver Grid"
    bl_idname = "PYRO_PT_grid"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Pyro"
    bl_options = {"DEFAULT_CLOSED"}

    def draw_header(self, context):
        self.layout.prop(context.scene.pyro, "grid_show", text="")

    def draw(self, context):
        layout, props = self.layout, context.scene.pyro
        layout.use_property_split = True
        layout.use_property_decorate = False
        if props.grid_show and not context.space_data.overlay.show_overlays:
            layout.prop(context.space_data.overlay, "show_overlays", text="Viewport Overlays")
        if props.algorithm == "SPARSE":
            layout.prop(props, "grid_level")
        if props.algorithm == "SPARSE":
            layout.prop(props, "grid_color")
            if props.grid_color == "CHANGES":
                layout.prop(props, "grid_removed")
                layout.label(text="Green: new | Red: removed/replaced")
                layout.label(text="Cyan: refined | Orange: coarsened")
                layout.label(text="Compared with the previous output frame")
        layout.prop(props, "grid_slice")
        if props.grid_slice != "NONE":
            layout.prop(props, "grid_position")
        layout.prop(props, "grid_xray")
        layout.prop(props, "grid_opacity")
        view = grid_view.views.get(context.scene.as_pointer())
        if props.algorithm != "SPARSE":
            nx, ny, nz = props.dense_resolution
            layout.label(text=f"Configured domain: {nx} × {ny} × {nz}")
            draw_hint(layout, context, f"Cells: {nx*ny*nz:,} | Voxel: {props.cell_size:.4g} BU")
            return
        layout.operator("pyro.inspect_grid", icon="EYEDROPPER")
        layout.prop(props, "grid_cells")
        header, body = layout.panel("pyro_grid_statistics", default_closed=False)
        header.label(text="Grid Snapshot")
        if not body:
            return
        if props.stale:
            body.label(text="Outdated — reset simulation", icon="ERROR")
            return
        if view is None:
            body.label(text="No grid snapshot at the current frame")
            return
        stats = view["stats"]
        draw_hint(body, context, f"Frame {stats['frame']} | Finest voxel: {stats['spacing']:.4g} BU")
        body.progress(factor=stats["blocks"]/stats["capacity"], text=f"Blocks: {stats['blocks']:,} / {stats['capacity']:,}")
        draw_hint(body, context, "Allocated capacity expands automatically")
        for level, count in enumerate(stats["level_blocks"]):
            draw_hint(body, context, f"L{level} · {grid_view.LEVEL_NAMES[level]}: {count:,} | h={stats['spacing']*(1 << level):.4g}")
        equivalent = sum(count*512*8**level for level, count in enumerate(stats["level_blocks"]))
        body.label(text=f"Leaf cells: {stats['blocks']*512:,}")
        body.label(text=f"Same coverage at L0: {equivalent:,} cells")
        if view["blocks"] is None:
            body.label(text="Enable the grid to load block details")
            return
        grid_view.compare(view)
        changes = view["changes"]
        draw_hint(body, context, f"Added: {np.count_nonzero(changes):,} | Removed/replaced: {len(view['removed']):,}")
        draw_hint(body, context, f"Refined: {np.count_nonzero(changes == 2):,} | Coarsened: {np.count_nonzero(changes == 3):,}")
        if not view["dirty"]:
            draw_hint(body, context, f"Blocks after display filters: {view['visible']:,}")
        selection = view["selection"]
        if selection is not None:
            records = view["removed"] if selection[2] else view["blocks"]
            matches = np.flatnonzero(np.all(records["coordinate"] == selection[0], axis=1) & (records["level"] == selection[1]))
            if len(matches):
                block = records[matches[0]]
                spacing = stats["spacing"] * (1 << int(block["level"]))
                box = body.box()
                box.label(text=f"L{block['level']} block: {selection[0]}")
                draw_hint(box, context, f"Voxel: {spacing:.4g} | Width: {8*spacing:.4g} BU")
                draw_hint(box, context, "Removed/replaced (previous frame)" if selection[2] else "Current frame")
                box.label(text=f"Quiet adaptations: {block['quiet']}")
                draw_hint(box, context, f"Keep-alive: {block['remaining']} adaptations")


classes = (PYRO_UL_objects, PYRO_PT_simulation, PYRO_PT_scene, PYRO_PT_active_object,
           PYRO_PT_physics, PYRO_PT_grid)
