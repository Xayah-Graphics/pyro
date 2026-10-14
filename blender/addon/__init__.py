import bpy

from . import grid_view, operators, runtime, settings, ui


def register():
    settings.register()
    for item in operators.classes:
        bpy.utils.register_class(item)
    grid_view.register()
    for item in ui.classes:
        bpy.utils.register_class(item)
    bpy.types.VIEW3D_MT_add.append(operators.draw_add_menu)
    for name, callback in runtime.handlers:
        getattr(bpy.app.handlers, name).append(callback)


def unregister():
    runtime.closing(None)
    for name, callback in runtime.handlers:
        getattr(bpy.app.handlers, name).remove(callback)
    bpy.types.VIEW3D_MT_add.remove(operators.draw_add_menu)
    for item in reversed(ui.classes):
        bpy.utils.unregister_class(item)
    grid_view.unregister()
    for item in reversed(operators.classes):
        bpy.utils.unregister_class(item)
    settings.unregister()
