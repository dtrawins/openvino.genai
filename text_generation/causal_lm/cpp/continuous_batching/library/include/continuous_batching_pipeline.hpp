// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <openvino/openvino.hpp>

#include "scheduler_config.hpp"
#include "tokenizer.hpp"
#include "generation_config.hpp"
#include "generation_handle.hpp"

struct GenerationResult {
    // request ID - obsolete when handle API is approved as handle will connect results with prompts.
    uint64_t m_request_id;

    // in a generic case we have multiple generation results per initial prompt
    // depending on sampling parameters (e.g. beam search or parallel sampling)
    std::vector<std::string> m_generation_ids;
    // scores
    std::vector<float> m_scores;
};

class ContinuousBatchingPipeline {
    class Impl;
    std::shared_ptr<Impl> m_impl;

public:
    ContinuousBatchingPipeline(const std::string& models_path,
                               const SchedulerConfig& scheduler_config,
                               const std::string& tokenizer_lib_path = {});

    std::shared_ptr<Tokenizer> get_tokenizer();

    GenerationConfig get_config() const;

    GenerationHandle add_request(uint64_t request_id, std::string prompt, GenerationConfig sampling_params);

    void step();

    bool has_running_requests() const;

    bool has_awaiting_requests() const;

    // more high level interface, which can process multiple prompts in continuous batching manner
    std::vector<GenerationResult> generate(const std::vector<std::string>& prompts, std::vector<GenerationConfig> sampling_params);
};
