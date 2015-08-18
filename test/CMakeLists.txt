find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
set(test_bindir ${bin}/test)

find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIRS})

add_executable(test_conv test_conv.cpp)
include_directories(${HALIDE_INCLUDES})

GTEST_ADD_TESTS(test_conv "" test_conv.cpp)

set_target_properties(test_conv PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${test_bindir})

target_link_libraries(test_conv
    ${GTEST_BOTH_LIBRARIES}
    ${PNG_LIBRARIES}
    conv
)
