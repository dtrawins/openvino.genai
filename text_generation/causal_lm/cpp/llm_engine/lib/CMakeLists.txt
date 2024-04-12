cmake_minimum_required(VERSION 3.15)
project(causal_lm)
include_directories(include)
file(GLOB sources src/*.cpp) # upper case variable names reserved for exported variables by convention
add_library(causal_lm ${sources}) # STATIC by default
target_include_directories(causal_lm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)