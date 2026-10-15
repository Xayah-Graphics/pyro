"""World-space solver leaf overlays, level-aware inspection and dense domain preview."""
import json
from pathlib import Path

import bpy
from bpy_extras import view3d_utils
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np


views = {}
domains = {}
shader = None
handler = None
CORNERS = np.array([(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)], dtype=np.float32)
EDGES = np.array([(i, i | (1 << axis)) for axis in range(3) for i in range(8) if not i & (1 << axis)], dtype=np.uint32)
LEVEL_NAMES = ("Cyan", "Orange", "Violet", "Lime", "Blue", "Pink", "Yellow", "White")
LEVEL_COLORS = np.array(((0.1, 0.8, 1), (1, 0.5, 0.1), (0.7, 0.3, 1), (0.45, 1, 0.2),
                         (0.25, 0.4, 1), (1, 0.3, 0.65), (1, 0.9, 0.15), (0.85, 0.9, 0.95)), dtype=np.float32)


def redraw(self, context):
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def refresh(self, context):
    key = self.id_data.as_pointer()
    domains.pop(key, None)
    view = views.get(key)
    if view:
        view["dirty"] = True
        if not self.grid_show:
            view["blocks"] = None
            view["removed"] = view["changes"] = None
            view["batches"] = []
        elif self.algorithm == "SPARSE" and not self.stale and view["blocks"] is None:
            load(self.id_data, Path(view["stamp"][0]), view["stamp"][1], view["start"])
    redraw(self, context)


def load(scene, directory, frame, start):
    key = scene.as_pointer()
    stamp = (str(directory), frame)
    old = views.get(key)
    want_blocks = scene.pyro.grid_show and scene.pyro.algorithm == "SPARSE"
    if old and old["stamp"] == stamp and (not want_blocks or old["blocks"] is not None):
        return
    with np.load(directory / f"grid_{frame-start+1:06d}.npz") as data:
        stats = json.loads(str(data["stats"]))
        blocks = data["blocks"] if want_blocks else None
    view = {"stamp": stamp, "start": start, "stats": stats, "blocks": blocks, "dirty": True,
            "batches": [], "selection": old["selection"] if old else None, "removed": None, "changes": None}
    views[key] = view


def compare(view):
    if view["changes"] is not None:
        return
    directory, frame = view["stamp"]
    start, blocks, stats = view["start"], view["blocks"], view["stats"]
    previous = blocks[:0]
    if frame > start:
        with np.load(Path(directory) / f"grid_{frame-start:06d}.npz") as data:
            previous = data["blocks"]
    keys = [(*map(int, c), int(l)) for c, l in zip(blocks["coordinate"], blocks["level"])]
    before = [(*map(int, c), int(l)) for c, l in zip(previous["coordinate"], previous["level"])]
    current_set, previous_set = set(keys), set(before)
    changes = np.array([k not in previous_set for k in keys], dtype=np.uint8)
    removed = previous[np.array([k not in current_set for k in before], dtype=bool)]
    # The key includes the level: a parent and child may share integer coordinates.
    for i, k in enumerate(keys):
        if changes[i] and any((k[0] >> d, k[1] >> d, k[2] >> d, k[3]+d) in previous_set for d in range(1, stats["levels"]-k[3])):
            changes[i] = 2
    coarsened = set()
    for c, level in zip(removed["coordinate"], removed["level"]):
        for parent in range(int(level)+1, stats["levels"]):
            d = parent-int(level)
            k = (int(c[0]) >> d, int(c[1]) >> d, int(c[2]) >> d, parent)
            if k in current_set:
                coarsened.add(k)
                break
    for i, k in enumerate(keys):
        if k in coarsened:
            changes[i] = 3
    view.update(removed=removed, changes=changes)


def visible(scene):
    props, view = scene.pyro, views[scene.as_pointer()]
    blocks = view["blocks"]
    records, changes = blocks, None
    if props.grid_color == "CHANGES":
        compare(view)
        changes = view["changes"]
        if props.grid_removed and len(view["removed"]):
            records = np.concatenate((blocks, view["removed"]))
            changes = np.concatenate((changes, np.zeros(len(view["removed"]), dtype=np.uint8)))
    gone = np.arange(len(records)) >= len(blocks)
    mask = np.ones(len(records), dtype=bool)
    if props.grid_level >= 0:
        mask &= records["level"] == props.grid_level
    if props.grid_slice != "NONE":
        axis = "XYZ".index(props.grid_slice)
        width = np.ldexp(8 * view["stats"]["spacing"], records["level"])
        lower = records["coordinate"][:, axis] * width
        mask &= (lower <= props.grid_position) & (props.grid_position < lower + width)
    return records[mask], gone[mask], changes[mask] if changes is not None else None


def build(scene):
    global shader
    props, view = scene.pyro, views[scene.as_pointer()]
    records, gone, changes = visible(scene)
    view["visible"] = len(records)
    view["batches"] = []
    if shader is None:
        shader = gpu.shader.from_builtin("POLYLINE_FLAT_COLOR")
    if len(records):
        colors = np.ones((len(records), 4), dtype=np.float32)
        if props.grid_color == "LEVEL":
            colors[:, :3] = LEVEL_COLORS[records["level"]]
        elif props.grid_color == "CHANGES":
            colors[:] = (0.40, 0.47, 0.55, 0.3)
            colors[changes == 1] = (0.1, 1.0, 0.5, 1)
            colors[changes == 2] = (0.1, 0.8, 1.0, 1)
            colors[changes == 3] = (1.0, 0.6, 0.1, 1)
            colors[gone] = (1.0, 0.25, 0.3, 0.7)
        colors[:, 3] *= props.grid_opacity
        spacing = np.ldexp(view["stats"]["spacing"], records["level"])
        vertices = ((records["coordinate"][:, None, :] + CORNERS[None, :, :]) * (8 * spacing[:, None, None])).astype(np.float32)
        indices = (np.arange(len(records), dtype=np.uint32)[:, None, None] * 8 + EDGES[None, :, :]).reshape(-1, 2)
        batch = batch_for_shader(shader, "LINES", {"pos": vertices.reshape(-1, 3), "color": np.repeat(colors, 8, axis=0)}, indices=indices)
        view["batches"].append((batch, 1.2))
        selection = view["selection"]
        if selection is not None:
            selected = np.flatnonzero(np.all(records["coordinate"] == selection[0], axis=1) & (records["level"] == selection[1]) & (gone == selection[2]))
            if len(selected):
                index = selected[0]
                positions = vertices[index][EDGES].reshape(-1, 3)
                if props.grid_cells:
                    lines = []
                    for axis in range(3):
                        for a in range(9):
                            for b in range(9):
                                first = np.zeros(3)
                                first[(axis+1) % 3], first[(axis+2) % 3] = a, b
                                last = first.copy()
                                last[axis] = 8
                                lines.extend((first, last))
                    positions = ((np.array(lines) + records["coordinate"][index] * 8) * spacing[index]).astype(np.float32)
                colors = np.tile(np.array((1, 0.95, 0.3, 1), dtype=np.float32), (len(positions), 1))
                view["batches"].append((batch_for_shader(shader, "LINES", {"pos": positions, "color": colors}), 1.8))
            else:
                view["selection"] = None
    else:
        view["selection"] = None
    view["dirty"] = False


def draw():
    global shader
    context = bpy.context
    props = context.scene.pyro
    key = context.scene.as_pointer()
    if not props.grid_show or not context.space_data.overlay.show_overlays:
        return
    if props.algorithm != "SPARSE":
        stamp = (tuple(props.dense_origin), tuple(props.dense_resolution), props.cell_size, props.grid_slice, props.grid_position, props.grid_opacity)
        view = domains.get(key)
        if view is None or view["stamp"] != stamp:
            if shader is None:
                shader = gpu.shader.from_builtin("POLYLINE_FLAT_COLOR")
            origin = np.array(props.dense_origin, dtype=np.float32)
            resolution = np.array(props.dense_resolution)
            extent = resolution * props.cell_size
            positions = ((origin + CORNERS * extent)[EDGES].reshape(-1, 3)).astype(np.float32)
            if props.grid_slice != "NONE":
                axis = "XYZ".index(props.grid_slice)
                if origin[axis] <= props.grid_position < origin[axis]+extent[axis]:
                    for varying in ((axis+1) % 3, (axis+2) % 3):
                        across = 3-axis-varying
                        lines = np.empty((resolution[across]+1, 2, 3), dtype=np.float32)
                        lines[:] = origin
                        lines[:, :, axis] = props.grid_position
                        lines[:, 1, varying] += extent[varying]
                        lines[:, :, across] += np.arange(resolution[across]+1, dtype=np.float32)[:, None] * props.cell_size
                        positions = np.concatenate((positions, lines.reshape(-1, 3)))
            colors = np.tile(np.array((0.25, 0.65, 1.0, props.grid_opacity), dtype=np.float32), (len(positions), 1))
            view = {"stamp": stamp, "batches": [(batch_for_shader(shader, "LINES", {"pos": positions, "color": colors}), 1.2)]}
            domains[key] = view
    else:
        view = views.get(key)
        if props.stale or view is None or view["blocks"] is None:
            return
        if view["dirty"]:
            build(context.scene)
    depth, mask, blend = gpu.state.depth_test_get(), gpu.state.depth_mask_get(), gpu.state.blend_get()
    try:
        gpu.state.depth_test_set("NONE" if props.grid_xray else "LESS_EQUAL")
        gpu.state.depth_mask_set(False)
        gpu.state.blend_set("ALPHA")
        shader.bind()
        shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
        for batch, width in view["batches"]:
            shader.uniform_float("lineWidth", width)
            batch.draw(shader)
    finally:
        gpu.state.depth_test_set(depth)
        gpu.state.depth_mask_set(mask)
        gpu.state.blend_set(blend)


class PYRO_OT_grid_pick(bpy.types.Operator):
    bl_idname = "pyro.inspect_grid"
    bl_label = "Inspect Block"
    bl_description = "Click a visible leaf block; right-click or Escape cancels"

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D" and context.scene.pyro.algorithm == "SPARSE" and not context.scene.pyro.stale and context.scene.pyro.grid_show and context.scene.as_pointer() in views and views[context.scene.as_pointer()]["blocks"] is not None

    def invoke(self, context, event):
        context.window.cursor_modal_set("CROSSHAIR")
        context.workspace.status_text_set("Pyro Grid: click a leaf block; Esc / right-click to cancel")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"} or not self.poll(context):
            context.window.cursor_modal_restore()
            context.workspace.status_text_set(None)
            return {"CANCELLED"}
        if event.type != "LEFTMOUSE" or event.value != "PRESS":
            return {"PASS_THROUGH"}
        region = next(region for region in context.area.regions if region.type == "WINDOW")
        mouse = (event.mouse_x-region.x, event.mouse_y-region.y)
        if not (0 <= mouse[0] < region.width and 0 <= mouse[1] < region.height):
            return {"RUNNING_MODAL"}
        # Start on the near clip plane; the Blender orthographic-camera origin
        # helper returns a point halfway through the clip range, behind the grid.
        inverse = np.linalg.inv(np.array(context.space_data.region_3d.perspective_matrix, dtype=np.float64))
        near_clip = inverse @ np.array((2*mouse[0]/region.width-1, 2*mouse[1]/region.height-1, -1, 1))
        origin = near_clip[:3] / near_clip[3]
        direction = np.array(view3d_utils.region_2d_to_vector_3d(region, context.space_data.region_3d, mouse), dtype=np.float64)
        view = views[context.scene.as_pointer()]
        records, gone, _ = visible(context.scene)
        width = np.ldexp(8 * view["stats"]["spacing"], records["level"])
        lower = records["coordinate"] * width[:, None]
        upper = lower + width[:, None]
        near, far = np.full(len(records), -np.inf), np.full(len(records), np.inf)
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                far[(origin[axis] < lower[:, axis]) | (origin[axis] > upper[:, axis])] = -np.inf
            else:
                a, b = (lower[:, axis]-origin[axis])/direction[axis], (upper[:, axis]-origin[axis])/direction[axis]
                near, far = np.maximum(near, np.minimum(a, b)), np.minimum(far, np.maximum(a, b))
        distance = np.where(far >= np.maximum(near, 0), np.maximum(near, 0), np.inf)
        view["selection"] = None
        if len(distance) and np.isfinite(distance.min()):
            index = int(distance.argmin())
            view["selection"] = (tuple(int(v) for v in records["coordinate"][index]), int(records["level"][index]), bool(gone[index]))
        refresh(context.scene.pyro, context)
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)
        return {"FINISHED"}


def register():
    global handler
    bpy.utils.register_class(PYRO_OT_grid_pick)
    if not bpy.app.background:
        handler = bpy.types.SpaceView3D.draw_handler_add(draw, (), "WINDOW", "POST_VIEW")


def unregister():
    global handler, shader
    if handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(handler, "WINDOW")
        handler = None
    views.clear()
    domains.clear()
    shader = None
    bpy.utils.unregister_class(PYRO_OT_grid_pick)
