set(hl_src
      hl_bvlc_alexnet
)

set(halide_objs)
foreach(f ${hl_src})
    hl_aot_compile(${f}
        BIN ${bin}/hl
        OBJ ${obj}
        INC ${inc}/hl
    )
    list(APPEND halide_objs ${obj}/${f}.o)
endforeach(f)

add_library(conv
    ${halide_objs}
)

SET_TARGET_PROPERTIES (
    conv
    PROPERTIES LINKER_LANGUAGE CXX
)
