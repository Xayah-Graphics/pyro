"""Synchronous timeline evaluation, persistent checkpoints and cache playback."""
import json
from pathlib import Path
import uuid

import bpy
from bpy.app.handlers import persistent

from . import bridge, grid_view, inputs, process


records = {}
sessions = {}
busy = False
frame_evaluating = set()


def publish(record):
    path = record["directory"] / "cache.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record["manifest"], separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def checkpoint(key):
    session = sessions.get(key)
    if session is None:
        return
    record = records[key]
    if session["current"] > record["manifest"]["checkpoint"]:
        bridge.checkpoint(session)
        record["manifest"]["checkpoint"] = session["current"]
        publish(record)


def close_session(key, save=True):
    try:
        if save:
            checkpoint(key)
    finally:
        session = sessions.pop(key, None)
        if session is not None:
            bridge.close(session)


def invalidate(scene):
    if busy or scene.as_pointer() in frame_evaluating or not scene.pyro.cache_directory:
        return
    scene.pyro.stale = True


def create_volume(scene):
    volume = scene.pyro.volume
    if volume is None:
        volume = bpy.data.objects.new("Pyro Volume", bpy.data.volumes.new("Pyro Field"))
        scene.collection.objects.link(volume)
        volume.hide_viewport = volume.hide_render = True
        scene.pyro.volume = volume
    if volume.active_material is None:
        material = bpy.data.materials.new("Pyro Volume")
        material.node_tree.nodes.clear()
        output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
        shader = material.node_tree.nodes.new("ShaderNodeVolumePrincipled")
        shader.inputs["Density"].default_value = 0.15 if scene.pyro.algorithm == "LFM" else 8
        material.node_tree.links.new(shader.outputs["Volume"], output.inputs["Volume"])
        if volume.data.materials:
            volume.active_material = material
        else:
            volume.data.materials.append(material)
    material = volume.active_material
    output = next((node for node in material.node_tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
    shader = output.inputs["Volume"].links[0].from_node if output and output.inputs["Volume"].is_linked else None
    if shader and shader.type == "PRINCIPLED_VOLUME":
        shader.inputs["Density Attribute"].default_value = "vorticity" if scene.pyro.algorithm == "LFM" else "density"
    return volume


def connect(scene):
    props = scene.pyro
    objects = inputs.participants(scene)
    geometry, corners, hashes = inputs.geometry(objects, bpy.context.evaluated_depsgraph_get())
    dependencies = inputs.dependencies(objects)
    signature, dynamic = inputs.signature(scene, dependencies, hashes)
    if props.algorithm == "LFM":
        if any(obj.pyro.role == "EMITTER" for obj in objects):
            raise RuntimeError("LFM uses boundary inflow; remove Source roles or choose a smoke solver")
        if dynamic:
            raise RuntimeError("LFM requires static collider geometry and transforms")
    if props.cache_directory:
        directory = Path(props.cache_directory)
        manifest = json.loads((directory / "cache.json").read_text(encoding="utf-8"))
        props.stale = props.stale or manifest.get("format") != 3 or signature != manifest["signature"]
    else:
        if props.cache_root.startswith("//") and not bpy.data.filepath:
            raise RuntimeError("Save the Blender project before using a relative cache folder")
        directory = Path(bpy.path.abspath(props.cache_root)) / uuid.uuid4().hex
        directory.mkdir(parents=True)
        manifest = {"format": 3, "generation": directory.name, "algorithm": props.algorithm, "signature": signature, "start": scene.frame_start, "completed": scene.frame_start-1, "checkpoint": scene.frame_start-1}
        props.cache_directory = str(directory)
    watched = {obj.as_pointer() for obj in dependencies}
    for obj in dependencies:
        if obj.type == "MESH":
            watched.add(obj.data.as_pointer())
        animation = obj.animation_data
        if animation:
            if animation.action:
                watched.add(animation.action.as_pointer())
            watched.update(strip.action.as_pointer() for track in animation.nla_tracks for strip in track.strips if strip.action)
    record = {"directory": directory, "manifest": manifest, "objects": objects, "geometry": geometry, "corners": corners, "hashes": hashes,
        "watched": watched, "members": [(obj.name_full, obj.pyro.role) for obj in objects], "settings": inputs.settings_key(scene),
        "dependencies": dependencies, "poses": inputs.poses(dependencies),
        "dynamic": dynamic, "displayed": scene.frame_current}
    records[scene.as_pointer()] = record
    props.cache_start = manifest["start"]
    props.cache_frames = manifest["completed"] - manifest["start"] + 1
    publish(record)
    return record


def display(scene, record, frame):
    props = scene.pyro
    volume = props.volume
    manifest = record["manifest"]
    available = manifest["start"] <= frame <= manifest["completed"]
    if volume is not None:
        volume.hide_viewport = volume.hide_render = not available
    if available and volume is not None:
        path = str(record["directory"] / f"volume_{frame-manifest['start']+1:06d}.vdb")
        if volume.data.filepath != path or volume.data.is_sequence:
            volume.data.grids.unload()
            volume.data.is_sequence = False
            volume.data.filepath = path
            volume.data.update_tag()
    if available and not props.stale:
        grid_view.load(scene, record["directory"], frame, manifest["start"])
    else:
        grid_view.views.pop(scene.as_pointer(), None)
    record["displayed"] = frame


def evaluate(scene):
    global busy
    if busy:
        return
    key = scene.as_pointer()
    props = scene.pyro
    requested, subframe = scene.frame_current, scene.frame_subframe
    if not props.enabled and not props.cache_directory:
        return
    busy = True
    final_frame = requested
    try:
        record = records.get(key) or connect(scene)
        manifest = record["manifest"]
        if not props.enabled:
            display(scene, record, requested)
            return
        if props.stale:
            raise RuntimeError("Simulation inputs changed. Reset Simulation to recompute.")
        if inputs.settings_key(scene) != record["settings"]:
            props.stale = True
            raise RuntimeError("Simulation settings changed. Reset Simulation to recompute.")
        if requested >= manifest["start"] and requested > manifest["completed"]:
            scene.sync_mode = "NONE"
            scene.render.use_lock_interface = True
            create_volume(scene)
            session = sessions.get(key)
            if session is None:
                session = bridge.start(scene, record["objects"], record["geometry"], record["corners"], record["dynamic"], record["directory"])
                sessions[key] = session
                if (record["directory"] / "state.bin").exists():
                    bridge.checkpoint(session, restore=True)
                    manifest["checkpoint"] = session["current"]
                if manifest["completed"] < manifest["start"]:
                    bridge.export(session, True)
                    manifest["completed"] = manifest["start"]
                    publish(record)
            while session["current"] < requested:
                bridge.advance(scene, session)
                new_frame = session["current"] > manifest["completed"]
                bridge.export(session, new_frame)
                if new_frame:
                    manifest["completed"] = session["current"]
                    publish(record)
                if (session["current"] - session["start"]) % props.checkpoint_interval == 0:
                    checkpoint(key)
            props.cache_frames = manifest["completed"] - manifest["start"] + 1
        display(scene, record, requested)
        props.error = ""
    except Exception as error:
        props.error = str(error)
        if key in records:
            final_frame = records[key]["displayed"]
            display(scene, records[key], final_frame)
        close_session(key, save=False)
        for window in bpy.context.window_manager.windows:
            if window.scene == scene and window.screen.is_animation_playing:
                with bpy.context.temp_override(window=window):
                    bpy.ops.screen.animation_cancel(restore_frame=False)
        raise
    finally:
        try:
            # Sampling must never leave animated transforms or the Volume at an intermediate time.
            scene.frame_set(final_frame, subframe=subframe if final_frame == requested else 0)
            if key in records:
                records[key]["poses"] = inputs.poses(records[key]["dependencies"])
        finally:
            busy = False


def reset(scene):
    global busy
    key = scene.as_pointer()
    busy = True
    try:
        close_session(key, save=False)
        records.pop(key, None)
        grid_view.views.pop(key, None)
        props = scene.pyro
        if props.volume:
            props.volume.data.grids.unload()
            props.volume.data.filepath = ""
            props.volume.hide_viewport = props.volume.hide_render = True
        if props.cache_directory:
            directory = Path(props.cache_directory).resolve()
            if directory.exists():
                manifest = json.loads((directory / "cache.json").read_text(encoding="utf-8"))
                if directory.name != manifest["generation"]:
                    raise RuntimeError("Cache directory does not match its generation")
                process.recycle(str(directory))
        props.cache_directory = props.error = ""
        props.cache_frames = 0
        props.stale = False
        scene.frame_set(scene.frame_start)
    finally:
        busy = False


@persistent
def frame_pre(scene, depsgraph=None):
    if not busy:
        frame_evaluating.add(scene.as_pointer())


@persistent
def frame_post(scene, depsgraph=None):
    if busy:
        return
    try:
        evaluate(scene)
    finally:
        frame_evaluating.discard(scene.as_pointer())


@persistent
def edited(scene, depsgraph):
    if busy or scene.as_pointer() in frame_evaluating or scene.pyro.stale:
        return
    record = records.get(scene.as_pointer())
    if record is None:
        return
    if inputs.settings_key(scene) != record["settings"] or [(obj.name_full, obj.pyro.role) for obj in inputs.participants(scene)] != record["members"]:
        invalidate(scene)
        return
    relevant = [update for update in depsgraph.updates if update.id.original.as_pointer() in record["watched"]]
    if not relevant:
        return
    if any(update.is_updated_transform for update in relevant) and inputs.poses(record["dependencies"]) != record["poses"]:
        invalidate(scene)
        return
    # Material edits may tag the owning Object too; compare actual supported inputs.
    geometry_ids = {update.id.original.as_pointer() for update in relevant if update.is_updated_geometry}
    changed_meshes = [obj for obj in record["objects"] if obj.as_pointer() in geometry_ids or obj.data.as_pointer() in geometry_ids]
    if changed_meshes:
        _, _, hashes = inputs.geometry(changed_meshes, depsgraph, mesh_data=False)
        record["hashes"].update(hashes)
    signature, _ = inputs.signature(scene, inputs.dependencies(record["objects"]), record["hashes"])
    if signature != record["manifest"]["signature"]:
        invalidate(scene)


@persistent
def saving(_):
    for key in list(sessions):
        checkpoint(key)


@persistent
def closing(_):
    for key in list(sessions):
        close_session(key)
    records.clear()
    grid_view.views.clear()
    grid_view.domains.clear()
    frame_evaluating.clear()


@persistent
def loaded(_):
    global busy
    busy = True
    try:
        for scene in bpy.data.scenes:
            if not scene.pyro.cache_directory:
                continue
            window = bpy.context.window_manager.windows[0]
            previous = window.scene
            try:
                window.scene = scene
                with bpy.context.temp_override(window=window):
                    record = connect(scene)
                    display(scene, record, scene.frame_current)
            except Exception as error:
                scene.pyro.error = str(error)
            finally:
                window.scene = previous
    finally:
        busy = False


handlers = (
    ("frame_change_pre", frame_pre), ("frame_change_post", frame_post),
    ("depsgraph_update_post", edited), ("save_pre", saving),
    ("load_pre", closing), ("load_post", loaded),
    ("undo_pre", closing), ("undo_post", loaded),
    ("redo_pre", closing), ("redo_post", loaded), ("exit_pre", closing),
)
