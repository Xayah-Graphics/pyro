find_package(CUDAToolkit 13.4 REQUIRED)

if(WIN32)
    add_compile_options(-Xcompiler=/Zc:preprocessor)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
