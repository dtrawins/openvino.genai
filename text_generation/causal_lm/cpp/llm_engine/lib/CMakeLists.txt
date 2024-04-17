cmake_minimum_required(VERSION 3.15)
project(casual_lm)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

add_subdirectory(../../../../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")

include_directories(include)
file(GLOB sources src/*.cpp) # upper case variable names reserved for exported variables by convention
add_library(casual_lm ${sources}) # STATIC by default
target_include_directories(casual_lm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(casual_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(casual_lm PRIVATE openvino::runtime)
set_target_properties(casual_lm PROPERTIES CXX_STANDARD 20)
set_target_properties(casual_lm PROPERTIES CXX_STANDARD_REQUIRED ON)

install(TARGETS casual_lm         
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin )