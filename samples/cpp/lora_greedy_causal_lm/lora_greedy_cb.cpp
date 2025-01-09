// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/continuous_batching_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <ADAPTER_SAFETENSORS_FILE> \"<PROMPT>\"");

    std::string models_path = argv[1];
    std::string adapter_path = argv[2];
    std::string prompt = argv[3];
    std::string device = "CPU";  // GPU can be used as well

    using namespace ov::genai;
    SchedulerConfig schedulerConfig{};
    ov::AnyMap tokenizerPluginConfig{};
    ov::AnyMap pluginConfig{};

    Adapter adapter(adapter_path);
    pluginConfig.insert(ov::genai::adapters(adapter, 0.75f));

    ContinuousBatchingPipeline pipe(models_path, schedulerConfig, device, pluginConfig, tokenizerPluginConfig);    // register all required adapters here

    GenerationConfig generationConfig{};
    generationConfig.max_new_tokens = 100;


    std::cout << "Generate with LoRA adapter and alpha set to 0.75:" << std::endl;
    auto result = pipe.generate(std::vector<std::string>{prompt}, std::vector<ov::genai::GenerationConfig>{generationConfig});
    for (auto & res : result.at(0).m_generation_ids) {
        std::cout << res << std::endl;
    }


} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
