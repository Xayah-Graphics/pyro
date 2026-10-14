"""Explicit simulation inputs, sampled through Blender's evaluated dependency graph."""
import hashlib
import json
import math
from pathlib import Path
import re

import bpy
from mathutils import Vector
import numpy as np

from .settings import CONFIG, OBJECT_INPUTS


solver_digest = None
TRANSFORMS = ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale",
              "delta_location", "delta_rotation_euler", "delta_rotation_quaternion", "delta_scale")


def participants(scene):
    return sorted((obj for obj in scene.objects if obj.type == "MESH" and obj.pyro.role != "NONE"), key=lambda obj: obj.name_full)


def dependencies(objects):
    paths = {}
    pending = [(obj, set(TRANSFORMS) | {"pyro." + name for name in OBJECT_INPUTS}) for obj in objects]
    while pending:
        obj, required = pending.pop()
        watched = paths.setdefault(obj, set())
        added = required - watched
        if not added:
            continue
        watched.update(added)
        if added.intersection(TRANSFORMS):
            if obj.parent:
                pending.append((obj.parent, set(TRANSFORMS)))
            for constraint in obj.constraints:
                watched.update(constraint.path_from_id(prop.identifier) for prop in constraint.bl_rna.properties
                               if prop.identifier != "rna_type" and not prop.is_readonly)
                for name in ("target", "pole_target", "space_object"):
                    target = getattr(constraint, name, None)
                    if isinstance(target, bpy.types.Object):
                        pending.append((target, set(TRANSFORMS)))
        if obj.animation_data:
            for curve in obj.animation_data.drivers:
                if curve.data_path not in watched:
                    continue
                for variable in curve.driver.variables:
                    for target in variable.targets:
                        if isinstance(target.id, bpy.types.Object):
                            pending.append((target.id, {re.sub(r"\[\d+\]$", "", target.data_path)} if variable.type == "SINGLE_PROP" else set(TRANSFORMS)))
    return paths


def geometry(objects, graph, *, mesh_data=True):
    triangles, corners, hashes = [], [], {}
    for obj in objects:
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            indices = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
            mesh.vertices.foreach_get("co", vertices)
            mesh.loop_triangles.foreach_get("vertices", indices)
            vertices = vertices.reshape(-1, 3)
            if mesh_data:
                triangles.append(vertices[indices].reshape(-1, 9))
                lower, upper = vertices.min(axis=0), vertices.max(axis=0)
                corners.append(np.array([[upper[k] if i & (1 << k) else lower[k] for k in range(3)] + [1] for i in range(8)]).T)
            hashes[obj.name_full] = hashlib.sha256(vertices.tobytes() + indices.tobytes()).hexdigest()
        finally:
            evaluated.to_mesh_clear()
    return triangles, corners, hashes


def settings_key(scene):
    return (scene.frame_start, scene.render.fps, scene.render.fps_base, scene.pyro.animation_tolerance, scene.pyro.algorithm,
            CONFIG[scene.pyro.algorithm],
            *(tuple(getattr(scene.pyro, name)) if kind in {"3f", "3i"} else getattr(scene.pyro, name) for name, kind in CONFIG[scene.pyro.algorithm]))


def poses(dependencies):
    return {obj.as_pointer(): tuple(v for row in obj.matrix_world for v in row) for obj, paths in dependencies.items() if paths.intersection(TRANSFORMS)}


def signature(scene, dependencies, hashes):
    global solver_digest
    if solver_digest is None:
        root = Path(__file__).parent
        digest = hashlib.sha256((root / "bin" / "pyro.exe").read_bytes())
        for name in ("bridge.py", "inputs.py"):
            digest.update((root / name).read_bytes())
        solver_digest = digest.hexdigest()

    def properties(item, animated=()):
        values = {}
        for prop in item.bl_rna.properties:
            if prop.identifier == "rna_type" or prop.is_readonly:
                continue
            value = getattr(item, prop.identifier)
            path = item.path_from_id(prop.identifier) if animated else ""
            if (path, 0) in animated and isinstance(value, (str, int, float, bool)):
                values[prop.identifier] = None
            elif isinstance(value, bpy.types.ID):
                values[prop.identifier] = value.name_full
            elif isinstance(value, (str, int, float, bool)) or value is None:
                values[prop.identifier] = value
            elif getattr(prop, "is_array", False):
                values[prop.identifier] = [None if (path, i) in animated else v for i, v in enumerate(value)]
            elif isinstance(value, set):
                values[prop.identifier] = sorted(value)
        return values

    values, dynamic = [], False
    for obj, paths in sorted(dependencies.items(), key=lambda item: item[0].name_full):
        curves, animated, nla = [], set(), []
        animation = obj.animation_data
        relevant_actions = set()
        if animation:
            actions = [(animation.action, animation.action_slot_handle)] if animation.action else []
            actions.extend((strip.action, strip.action_slot_handle) for track in animation.nla_tracks for strip in track.strips if strip.action)
            for action, slot in dict.fromkeys(actions):
                for layer in action.layers:
                    for strip in layer.strips:
                        for bag in strip.channelbags:
                            if bag.slot_handle != slot:
                                continue
                            for curve in bag.fcurves:
                                if curve.data_path not in paths:
                                    continue
                                relevant_actions.add((action, slot))
                                animated.add((curve.data_path, curve.array_index))
                                curves.append((action.name_full, slot, curve.data_path, curve.array_index, curve.mute, curve.extrapolation,
                                    [(list(k.co), list(k.handle_left), list(k.handle_right), k.interpolation, k.easing, k.amplitude, k.back, k.period) for k in curve.keyframe_points],
                                    [properties(modifier) for modifier in curve.modifiers]))
            for track in animation.nla_tracks:
                if any((strip.action, strip.action_slot_handle) in relevant_actions for strip in track.strips):
                    nla.append((track.mute, track.is_solo, [properties(strip) for strip in track.strips
                        if not strip.action or (strip.action, strip.action_slot_handle) in relevant_actions]))
            for curve in animation.drivers:
                if curve.data_path not in paths:
                    continue
                animated.add((curve.data_path, curve.array_index))
                curves.append((curve.data_path, curve.array_index, curve.driver.type, curve.driver.expression, curve.driver.use_self,
                    [(v.name, v.type, [(t.id.name_full if t.id else None, t.data_path, t.bone_target, t.transform_type, t.transform_space) for t in v.targets]) for v in curve.driver.variables]))
        value = {"name": obj.name_full, "curves": curves, "nla": nla}
        transforms = bool(paths.intersection(TRANSFORMS))
        if transforms:
            value.update(parent=obj.parent.name_full if obj.parent else None, parent_type=obj.parent_type, parent_bone=obj.parent_bone,
                parent_inverse=[list(row) for row in obj.matrix_parent_inverse], rotation_mode=obj.rotation_mode,
                constraints=[properties(c, animated) for c in obj.constraints])
        for path in sorted(paths):
            if transforms and path.startswith("constraints["):
                continue
            data = obj.path_resolve(path)
            if isinstance(data, (str, int, float, bool)):
                value[path] = None if (path, 0) in animated else data
            else:
                value[path] = [None if (path, i) in animated else v for i, v in enumerate(data)]
        if obj.name_full in hashes:
            value["mesh"] = hashes[obj.name_full]
        if relevant_actions:
            action = animation.action if (animation.action, animation.action_slot_handle) in relevant_actions else None
            value["animation"] = (action.name_full if action else None, animation.action_slot_handle if action else None,
                                  animation.use_nla, animation.action_blend_type, animation.action_influence)
        dynamic |= bool(curves) or transforms and bool(obj.constraints)
        values.append(value)
    digest = hashlib.sha256(json.dumps((solver_digest, settings_key(scene), values), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, dynamic


def sample(scene, objects, frame):
    integer = math.floor(frame)
    scene.frame_set(integer, subframe=frame-integer)
    graph = bpy.context.evaluated_depsgraph_get()
    result = []
    for obj in objects:
        evaluated = obj.evaluated_get(graph)
        props = evaluated.pyro
        velocity = props.velocity[:]
        if props.velocity_space == "LOCAL":
            velocity = evaluated.matrix_world.to_quaternion() @ Vector(velocity)
        result.append((np.array(evaluated.matrix_world, dtype=np.float64)[:3], props.active,
            (props.smoke_rate, props.temperature_rate, props.velocity_rate, *velocity, props.inherit_velocity), props.surface_offset))
    return result
