// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>

#include <openvino/runtime/tensor.hpp>

template <typename T>
static void print_array(T * array, size_t size) {
    std::cout << " => [ ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << " ] " << std::endl;
}

static void print_tensor(std::string name, ov::Tensor tensor) {
    std::cout << name;
    if (tensor.get_element_type() == ov::element::i32) {
        print_array(tensor.data<int>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::i64) {
        print_array(tensor.data<int64_t>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::f32) {
        print_array(tensor.data<float>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::boolean) {
        print_array(tensor.data<bool>(), tensor.get_size());
    }
}

static std::vector<std::string> split(const std::string &input, char delim) {
    std::vector<std::string> result;
    std::stringstream ss (input);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}

static bool is_absolute_path(const std::string& path) {
    return !path.empty() && (path[0] == '/');
}

static std::string join_path(std::initializer_list<std::string> segments) {
        std::string joined;

        for (const auto& seg : segments) {
            if (joined.empty()) {
                joined = seg;
            } else if (is_absolute_path(seg)) {
                if (joined[joined.size() - 1] == '/') {
                    joined.append(seg.substr(1));
                } else {
                    joined.append(seg);
                }
            } else {
                if (joined[joined.size() - 1] != '/') {
                    joined.append("/");
                }
                joined.append(seg);
            }
        }

        return joined;
    }

static bool is_path_escaped(const std::string& path) {
    std::size_t lhs = path.find("../");
    std::size_t rhs = path.find("/..");
    return (std::string::npos != lhs && lhs == 0) || (std::string::npos != rhs && rhs == path.length() - 3) || std::string::npos != path.find("/../");
}

static std::string get_openvino_tokenizer_path(std::string input_path) {
    const std::string LIB_NAME = "libopenvino_tokenizers.so";
    const char DELIM = ':';

    std::vector<std::string> search_order = {"LD_PRELOAD", "LD_LIBRARY_PATH"};
    
    if (is_path_escaped(input_path)) {
        std::cout << "ERROR: Path: " << input_path << " get_openvino_tokenizer_path path is escaped: " << std::endl;
        return "";
    }

    if (input_path == "") {
        std::filesystem::path cwd = std::filesystem::current_path();
        input_path = join_path({cwd, LIB_NAME});
    }

    if (input_path.find(LIB_NAME) == std::string::npos) {
        input_path = join_path({input_path, LIB_NAME});
    }

    try {
        if (std::filesystem::exists(input_path))
            return input_path;
    } catch (const std::exception& e) {
        std::cout << "WARNING: Path: " << input_path << std::endl << " get_openvino_tokenizer_path exception: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "WARNING: Path: " << input_path << std::endl << " get_openvino_tokenizer_path exception. " << std::endl;
    }
    
    try {
        for (auto& env_var: search_order) {
            std::string env_val = std::string(std::getenv(env_var.c_str()));
            std::cout << "DEBUG: env_val " << env_val << std::endl;
            if (env_val == "")
                continue;
            
            std::vector<std::string> paths = split(env_val, DELIM);
            for (auto& path: paths) {
                std::cout << "DEBUG: path " << path << std::endl;
                if (is_path_escaped(path))
                    return std::string();

                input_path = join_path({path, LIB_NAME});
                std::cout << "DEBUG: input_path " << input_path << std::endl;
                if (std::filesystem::exists(input_path))
                    return input_path;
            }
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR: Path: " << input_path << std::endl << " get_openvino_tokenizer_path exception: " << e.what() << std::endl;
    } catch (...) {}

    return LIB_NAME;
}
