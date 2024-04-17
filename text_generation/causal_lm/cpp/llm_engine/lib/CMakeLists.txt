cmake_minimum_required(VERSION 3.15)
project(causal_lm)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

add_subdirectory(../../../../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")

include_directories(include)
file(GLOB sources src/*.cpp) # upper case variable names reserved for exported variables by convention
add_library(causal_lm ${sources}) # STATIC by default
target_include_directories(causal_lm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(causal_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(causal_lm PRIVATE openvino::runtime)
set_target_properties(causal_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(causal_lm PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(causal_lm PROPERTIES PUBLIC_HEADER "include/block_manager.hpp;include/llm_engine.hpp;include/paged_attention.hpp;include/scheduler.hpp;include/cache_manager.hpp;include/model_config.hpp;include/sampler.hpp;include/sequence_group.hpp;include/debug_utils.hpp;include/model_runner.hpp;include/sampling_parameters.hpp;include/timer.hpp")

install(TARGETS causal_lm         
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin 
        PUBLIC_HEADER DESTINATION include)