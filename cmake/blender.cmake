find_file(PYRO_CUDART_DLL cudart64_13.dll REQUIRED PATHS "${CUDAToolkit_BIN_DIR}/x64" "${CUDAToolkit_BIN_DIR}" NO_DEFAULT_PATH)

set(PYRO_ADDON_STAGE "${CMAKE_BINARY_DIR}/blender-addon")
file(MAKE_DIRECTORY "${PYRO_ADDON_STAGE}/bin")
add_custom_target(pyro_blender
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_SOURCE_DIR}/blender/addon/__init__.py"
        "${CMAKE_SOURCE_DIR}/blender/addon/process.py"
        "${CMAKE_SOURCE_DIR}/blender/addon/settings.py"
        "${CMAKE_SOURCE_DIR}/blender/addon/inputs.py"
        "${CMAKE_SOURCE_DIR}/blender/addon/runtime.py"
        "${CMAKE_SOURCE_DIR}/blender/addon/bridge.py"
        "${CMAKE_SOURCE_DIR}/blender/addon/grid_view.py"
        "${CMAKE_SOURCE_DIR}/blender/addon/blender_manifest.toml"
        "${CMAKE_SOURCE_DIR}/LICENSE"
        "${PYRO_ADDON_STAGE}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:pyro>" "${PYRO_CUDART_DLL}" "${PYRO_ADDON_STAGE}/bin"
    COMMAND "${CMAKE_COMMAND}" -E tar cf "${CMAKE_BINARY_DIR}/pyro_blender.zip" --format=zip
        __init__.py process.py settings.py inputs.py runtime.py bridge.py grid_view.py blender_manifest.toml LICENSE bin
    DEPENDS pyro
    WORKING_DIRECTORY "${PYRO_ADDON_STAGE}"
    VERBATIM
)
