function(hl_aot_compile f)
    set(bin_dir BIN RENAME)
    set(inc_dir INC RENAME)
    set(obj_dir OBJ RENAME)
    set(extra_libs LIBRARIES CONFIGURATIONS)
    set(halide_args HALIDE_ARGS RENAME)
    set(no_compilation NO_COMPILATION RENAME)
    cmake_parse_arguments(AOT "" "${bin_dir};${inc_dir};${obj_dir};${halide_args};${no_compilation}" "${extra_libs}" ${ARGN})

    add_executable(${f} "${f}.cpp")
    set_target_properties(${f} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${AOT_BIN}
    )
    target_link_libraries(${f}
        ${HALIDE_LIBRARY}
        ${AOT_LIBRARIES}
    )

    # Run the halide file to generate .h and .o
    add_custom_command(
        OUTPUT ${AOT_OBJ}/${f}.o ${AOT_INC}/${f}.h
        DEPENDS ${f}
        WORKING_DIRECTORY ${AOT_BIN}
        COMMAND "./${f}"
        ARGS  ${AOT_HALIDE_ARGS}
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${AOT_BIN}/${f}.o ${AOT_OBJ}/${f}.o
        COMMAND ${CMAKE_COMMAND}
        ARGS -E remove ${AOT_BIN}/${f}.o
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${AOT_BIN}/${f}.h ${AOT_INC}/${f}.h
        COMMAND ${CMAKE_COMMAND}
        ARGS -E remove ${AOT_BIN}/${f}.h
    )
endfunction()
