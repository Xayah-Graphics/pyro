"""Author the independent LFM delta-wing wind tunnel example."""
import importlib.util
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import bpy
from mathutils import Vector


root = Path(__file__).resolve().parent
destination = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
spec = importlib.util.spec_from_file_location("pyro_authoring", root / "addon" / "__init__.py", submodule_search_locations=[str(root / "addon")])
addon = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = addon
spec.loader.exec_module(addon)
addon.register()
scene = bpy.context.scene
scene.name = "LFM Delta Wing"
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.volumetric_tile_size = "2"
scene.eevee.volumetric_samples = 128
scene.eevee.use_volumetric_shadows = True
scene.eevee.volumetric_shadow_samples = 32
scene.render.resolution_x, scene.render.resolution_y = 1400, 900
scene.render.resolution_percentage = 100
scene.render.fps = 20
scene.frame_start, scene.frame_end = 1, 160
scene.unit_settings.system = "METRIC"
scene.pyro.algorithm = "LFM"
scene.pyro.cell_size = 1/128
scene.pyro.dense_origin = (0, 0, 0)
scene.pyro.dense_resolution = (256, 128, 128)
scene.pyro.wind = (0.6*math.cos(math.radians(20)), 0, 0.6*math.sin(math.radians(20)))
scene.pyro.lfm_cycle_steps = 5
scene.pyro.grid_show = True
scene.pyro.grid_opacity = 0.25

# Convert the reference Y-up scene to Blender Z-up with (x, y, z) -> (x, 1-z, y).
# This is a proper rotation plus translation: winding stays outward and the
# domain stays [0, 2] x [0, 1] x [0, 1]. Wind uses the rotation without translation.
# Match the reference solid_sdf.npy (tip X=0.1); its display OBJ uses 0.11.
vertices = [(0.1, y, 0.5) for y in (0.3, 0.33125)]
vertices += [(0.1+math.cos(math.radians(15)), y, 0.5+sign*math.sin(math.radians(15)))
             for sign in (1, -1) for y in (0.3, 0.33125)]
vertices = [(x, 1-z, y) for x, y, z in vertices]
faces = [(2, 1, 0), (4, 3, 2), (1, 3, 5), (0, 5, 4), (2, 0, 4), (3, 1, 2), (5, 3, 4), (1, 5, 0)]
mesh = bpy.data.meshes.new("Delta Wing")
mesh.from_pydata(vertices, [], faces)
mesh.update()
wing = bpy.data.objects.new("Delta Wing", mesh)
scene.collection.objects.link(wing)
wing.pyro.role = "COLLIDER"
material = bpy.data.materials.new("Wing Metal")
shader = material.node_tree.nodes.get("Principled BSDF")
shader.inputs["Base Color"].default_value = (0.12, 0.2, 0.28, 1)
shader.inputs["Metallic"].default_value = 0.55
shader.inputs["Roughness"].default_value = 0.28
mesh.materials.append(material)

volume = addon.runtime.create_volume(scene)
volume.name = volume.data.name = "Flow Vorticity"
volume.data.filepath = ""
volume.hide_select = True
shader = volume.active_material.node_tree.nodes.get("Principled Volume")
shader.inputs["Density"].default_value = 0.25
shader.inputs["Color"].default_value = (0.65, 0.82, 0.95, 1)
shader.inputs["Anisotropy"].default_value = 0.15

world = bpy.data.worlds.new("Wind Tunnel Background")
world.node_tree.nodes["Background"].inputs[0].default_value = (0.025, 0.035, 0.055, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.25
scene.world = world
for name, position, energy, size, color in (
    ("Key", (0.6, -0.8, 2.5), 350, 3.0, (0.75, 0.85, 1.0)),
    ("Rim", (1.6, 2.0, 1.3), 250, 2.0, (1.0, 0.65, 0.35)),
):
    light = bpy.data.lights.new(name, "AREA")
    light.energy, light.shape, light.size, light.color = energy, "DISK", size, color
    obj = bpy.data.objects.new(name, light)
    scene.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (Vector((0.9, 0.5, 0.45))-obj.location).to_track_quat("-Z", "Y").to_euler()
camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
scene.collection.objects.link(camera)
# Reference camera position/view after the same coordinate conversion.
camera.location = (-0.073, -0.810, 0.846)
camera.rotation_euler = Vector((0.545, 0.785, -0.294)).to_track_quat("-Z", "Y").to_euler()
camera.data.sensor_fit = "VERTICAL"
camera.data.lens = camera.data.sensor_height / (2*math.tan(math.radians(60)/2))
camera.data.clip_start, camera.data.clip_end = 0.01, 100
scene.camera = camera
for obj in bpy.context.selected_objects:
    obj.select_set(False)
wing.select_set(True)
bpy.context.view_layer.objects.active = wing
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_perspective = "CAMERA"
            area.spaces.active.shading.type = "MATERIAL"
            area.spaces.active.shading.use_scene_world = True
            area.spaces.active.shading.use_scene_lights = True
        elif area.type == "PROPERTIES":
            area.spaces.active.context = "SCENE"
scene.pyro.enabled = True
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(destination), relative_remap=False)
addon.unregister()
