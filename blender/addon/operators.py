"""Scene authoring and synchronous simulation actions."""
import bpy

from . import runtime


class PYRO_OT_add_object(bpy.types.Operator):
    bl_idname = "pyro.add_object"
    bl_label = "Add Pyro Object"
    bl_description = "Create a source or collider mesh at the 3D cursor"
    bl_options = {"REGISTER", "UNDO"}

    role: bpy.props.EnumProperty(name="Role", items=[("EMITTER", "Source", "Emit smoke and heat"), ("COLLIDER", "Collider", "One-way solid boundary")])
    shape: bpy.props.EnumProperty(name="Shape", items=[("SPHERE", "Sphere", ""), ("CUBE", "Cube", ""), ("CYLINDER", "Cylinder", "")])
    size: bpy.props.FloatProperty(name="Size", description="Diameter or side length in Blender units", default=1, min=0.0001, subtype="DISTANCE")

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and not runtime.busy

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        location = context.scene.cursor.location
        if self.shape == "SPHERE":
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=self.size/2, location=location)
        elif self.shape == "CUBE":
            bpy.ops.mesh.primitive_cube_add(size=self.size, location=location)
        else:
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=self.size/2, depth=self.size, location=location)
        obj = context.view_layer.objects.active
        obj.name = "Pyro Source" if self.role == "EMITTER" else "Pyro Collider"
        obj.pyro.role = self.role
        if self.role == "EMITTER":
            obj.display_type = "WIRE"
            obj.hide_render = True
            material = bpy.data.materials.new("Pyro Source Guide")
            material.node_tree.nodes.clear()
            output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
            transparent = material.node_tree.nodes.new("ShaderNodeBsdfTransparent")
            material.node_tree.links.new(transparent.outputs[0], output.inputs["Surface"])
            obj.data.materials.append(material)
        runtime.invalidate(context.scene)
        return {"FINISHED"}


class PYRO_OT_set_role(bpy.types.Operator):
    bl_idname = "pyro.set_role"
    bl_label = "Set Pyro Role"
    bl_description = "Assign the named object or selected meshes; None removes the role without deleting geometry"
    bl_options = {"REGISTER", "UNDO"}

    role: bpy.props.EnumProperty(name="Role", items=[("EMITTER", "Source", ""), ("COLLIDER", "Collider", ""), ("NONE", "None", "")])
    object_name: bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and not runtime.busy

    def execute(self, context):
        objects = [context.scene.objects[self.object_name]] if self.object_name else [obj for obj in context.selected_objects if obj.type == "MESH"]
        if not objects:
            self.report({"INFO"}, "Select a mesh to assign a Pyro role")
            return {"CANCELLED"}
        for obj in objects:
            obj.pyro.role = self.role
        runtime.invalidate(context.scene)
        return {"FINISHED"}


class PYRO_OT_select_object(bpy.types.Operator):
    bl_idname = "pyro.select_object"
    bl_label = "Select Pyro Object"
    bl_description = "Select this object to edit its settings; unhide it and unlock selection in the current view layer"
    bl_options = {"INTERNAL", "UNDO"}

    object_name: bpy.props.StringProperty(options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        obj = context.view_layer.objects[self.object_name]
        for selected in context.selected_objects:
            selected.select_set(False)
        obj.hide_set(False)
        obj.hide_select = False
        obj.select_set(True)
        context.view_layer.objects.active = obj
        context.scene.pyro.object_list_index = list(context.scene.objects).index(obj)
        if context.area.type == "PROPERTIES":
            context.space_data.pin_id = None
            context.space_data.context = "PHYSICS" if obj.type == "MESH" else "DATA"
        return {"FINISHED"}


class PYRO_OT_volume(bpy.types.Operator):
    bl_idname = "pyro.create_volume"
    bl_label = "Create Output Volume"
    bl_description = "Create the output volume and its scalar-field material in this scene"
    bl_options = {"UNDO"}

    def execute(self, context):
        runtime.create_volume(context.scene)
        return {"FINISHED"}


class PYRO_OT_evaluate(bpy.types.Operator):
    bl_idname = "pyro.evaluate"
    bl_label = "Evaluate Simulation"
    bl_description = "Enable simulation and synchronously compute the requested frame, including all missing frames"

    next_frame: bpy.props.BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        props = context.scene.pyro
        return not runtime.busy and not props.stale and context.scene.frame_current >= context.scene.frame_start and (bool(bpy.data.filepath) or not props.cache_root.startswith("//"))

    def execute(self, context):
        scene = context.scene
        scene.pyro.enabled = True
        try:
            if self.next_frame:
                runtime.busy = True
                try:
                    scene.frame_set(scene.frame_current + 1)
                finally:
                    runtime.busy = False
            runtime.evaluate(scene)
        except Exception as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class PYRO_OT_reset(bpy.types.Operator):
    bl_idname = "pyro.reset"
    bl_label = "Reset Simulation"
    bl_description = "Move this simulation's cache to the Recycle Bin and return to its start frame"

    @classmethod
    def poll(cls, context):
        props = context.scene.pyro
        return not runtime.busy and bool(props.cache_directory or props.stale or props.error)

    def execute(self, context):
        runtime.reset(context.scene)
        return {"FINISHED"}


class PYRO_OT_copy_error(bpy.types.Operator):
    bl_idname = "pyro.copy_error"
    bl_label = "Copy Full Error"
    bl_description = "Copy the complete last solver error to the clipboard"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        context.window_manager.clipboard = context.scene.pyro.error
        return {"FINISHED"}


class PYRO_MT_add(bpy.types.Menu):
    bl_idname = "PYRO_MT_add"
    bl_label = "Pyro"

    def draw(self, context):
        if context.scene.pyro.algorithm != "LFM":
            self.layout.operator("pyro.add_object", text="Smoke Source", icon="MOD_FLUID").role = "EMITTER"
        self.layout.operator("pyro.add_object", text="Collider", icon="MOD_PHYSICS").role = "COLLIDER"
        self.layout.separator()
        self.layout.operator("pyro.create_volume", icon="OUTLINER_DATA_VOLUME")


def draw_add_menu(self, context):
    self.layout.menu("PYRO_MT_add", icon="MOD_FLUID")


classes = (PYRO_OT_add_object, PYRO_OT_set_role, PYRO_OT_select_object, PYRO_OT_volume,
           PYRO_OT_evaluate, PYRO_OT_reset, PYRO_OT_copy_error, PYRO_MT_add)
