
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <limits>
#include <functional>

enum class StopCriteria {
    EARLY,
    HEURISTIC,
    NEVER
};

// TODO: implement better interface, because currently sequence is not available to public API
class Sequence;

struct GenerationConfig {
    // Generic
    size_t max_new_tokens = 30;
    size_t max_length = std::numeric_limits<std::size_t>::max(); // m_max_new_tokens should have priority over m_max_length
    bool ignore_eos = false;

    // Beam search specific
    size_t num_groups = 1;
    size_t group_size = 1; // beam_width
    float diversity_penalty = 1.0f; // 0.0 means no diversity
    StopCriteria stop_criteria = StopCriteria::HEURISTIC;
    size_t num_return_sequences = 3;  // is used by beam search, in other case is equal to batch size

    float repetition_penalty = 1.0f;
    float length_penalty = 1.0f;
    size_t no_repeat_ngram_size = std::numeric_limits<size_t>::max();
    std::function<bool(const Sequence&)> early_finish = [] (const Sequence&) { return false; };

    // Multinomial
    float temperature = 0.0f; // by default we use greedy sampling
    int top_k = -1; // maybe to assign vocab_size ?
    float top_p = 1.0f; // by default convsider all tokens
    bool do_sample;

    // special tokens IDs
    int64_t bos_token_id = -1;
    int64_t pad_token_id = -1;
    int64_t eos_token_id = -1;

    // reads generation config from HF generation_config.json
    static GenerationConfig from_file(const std::string& generation_config_json);

    static GenerationConfig greedy();

    static GenerationConfig beam_search();

    static GenerationConfig multinomial();

    bool is_gready_sampling() const {
        return temperature == 0.0f && !is_beam_search();
    }

    bool is_beam_search() const {
        return num_groups * group_size > 1;
    }
};
