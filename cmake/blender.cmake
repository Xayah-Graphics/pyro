find_file(PYRO_CUDART_DLL cudart64_13.dll REQUIRED PATHS "${CUDAToolkit_BIN_DIR}/x64" "${CUDAToolkit_BIN_DIR}" NO_DEFAULT_PATH)

set(PYRO_ADDON_STAGE "${CMAKE_BINARY_DIR}/blender-addon")
file(MAKE_DIRECTORY "${PYRO_ADDON_STAGE}/bin")
set(PYRO_ADDON_FILES
    __init__.py operators.py ui.py process.py settings.py inputs.py runtime.py bridge.py grid_view.py blender_manifest.toml
)
list(TRANSFORM PYRO_ADDON_FILES PREPEND "${CMAKE_SOURCE_DIR}/blender/addon/" OUTPUT_VARIABLE PYRO_ADDON_SOURCES)
add_custom_target(pyro_blender
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${PYRO_ADDON_SOURCES} "${CMAKE_SOURCE_DIR}/LICENSE" "${PYRO_ADDON_STAGE}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:pyro_stream>" "${PYRO_CUDART_DLL}" "${PYRO_ADDON_STAGE}/bin"
    COMMAND "${CMAKE_COMMAND}" -E tar cf "${CMAKE_BINARY_DIR}/pyro_blender.zip" --format=zip
        ${PYRO_ADDON_FILES} LICENSE "bin/$<TARGET_FILE_NAME:pyro_stream>" bin/cudart64_13.dll
    DEPENDS pyro_stream
    WORKING_DIRECTORY "${PYRO_ADDON_STAGE}"
    VERBATIM
)
