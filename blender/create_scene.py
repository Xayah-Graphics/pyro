"""Author an independent example .blend in a separate background Blender instance."""

import importlib.util
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
scene.name = "Smoke Around a Sphere"
scene.render.engine = "BLENDER_EEVEE"
scene.eevee.volumetric_tile_size = "2"
scene.eevee.volumetric_samples = 128
scene.eevee.use_volumetric_shadows = True
scene.eevee.volumetric_shadow_samples = 32
scene.pyro.algorithm = "SPARSE"
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = 192
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 1
scene.render.resolution_x = 768
scene.render.resolution_y = 960
scene.render.resolution_percentage = 100

bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.5, location=(0, 0, 1.5))
collider = bpy.context.object
collider.name = "Collision Sphere"
bpy.ops.object.shade_smooth()

bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.2, location=(0, 0, 0.25))
emitter = bpy.context.object
emitter.name = "Smoke Emitter"
emitter.color = (1, 0.35, 0.05, 1)
emitter.display_type = "WIRE"
emitter.hide_render = True
emitter.pyro.role = "EMITTER"
collider.pyro.role = "COLLIDER"
material = bpy.data.materials.new("Emitter Guide")
material.node_tree.nodes.clear()
output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
transparent = material.node_tree.nodes.new("ShaderNodeBsdfTransparent")
material.node_tree.links.new(transparent.outputs[0], output.inputs["Surface"])
emitter.data.materials.append(material)


bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.18, depth=0.1)
nozzle = bpy.context.object
nozzle.name = "Nozzle"
nozzle.parent = emitter
nozzle.location = (0, 0, -0.20)
bevel = nozzle.modifiers.new("Soft Edges", "BEVEL")
bevel.width = 0.012
bevel.segments = 3
bpy.ops.object.shade_smooth()

bpy.ops.mesh.primitive_plane_add(size=200)
floor = bpy.context.object
floor.name = "Ground"
for obj, color, metallic, roughness in (
    (collider, (0.025, 0.16, 0.33, 1), 0.3, 0.24),
    (nozzle, (0.09, 0.11, 0.13, 1), 0.7, 0.3),
    (floor, (0.025, 0.03, 0.04, 1), 0.0, 0.8),
):
    material = bpy.data.materials.new(obj.name)
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    obj.data.materials.append(material)

volume = addon.runtime.create_volume(scene)
volume.name = volume.data.name = "Smoke"
volume.hide_select = True
shader = volume.data.materials[0].node_tree.nodes.get("Principled Volume")
shader.inputs["Color"].default_value = (0.72, 0.77, 0.82, 1)
shader.inputs["Anisotropy"].default_value = 0.2

for name, position, energy, size, color in (
    ("Key Light", (3, -4, 6), 1800, 5, (1.0, 0.88, 0.74)),
    ("Rim Light", (-3, 2, 5), 2400, 4, (0.7, 0.82, 1.0)),
    ("Fill Light", (-4, -2, 3), 500, 4, (0.75, 0.85, 1.0)),
):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (Vector((0, 0, 2)) - obj.location).to_track_quat("-Z", "Y").to_euler()

camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
scene.collection.objects.link(camera)
camera.location = (5, -9, 4.2)
camera.rotation_euler = (Vector((0, 0, 2.2)) - camera.location).to_track_quat("-Z", "Y").to_euler()
camera.data.type = "ORTHO"
camera.data.ortho_scale = 6
scene.camera = camera
world = bpy.data.worlds.new("Studio World")
world.node_tree.nodes.get("Background").inputs["Color"].default_value = (0.035, 0.045, 0.06, 1)
world.node_tree.nodes.get("Background").inputs["Strength"].default_value = 0.3
scene.world = world
scene.pyro.enabled = True
bpy.ops.object.select_all(action="DESELECT")
collider.select_set(True)
bpy.context.view_layer.objects.active = collider
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == "VIEW_3D":
            space = area.spaces.active
            space.region_3d.view_perspective = "CAMERA"
            space.shading.type = "RENDERED"
            space.overlay.show_overlays = False
            space.show_region_ui = True
destination.parent.mkdir(parents=True, exist_ok=True)
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(destination))
addon.unregister()
