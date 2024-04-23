// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "generation_handle.hpp"
#include "generation_stream.hpp"

bool GenerationHandle::generation_finished() {
    return m_generation_stream->generation_finished();
}

bool GenerationHandle::can_read() {
    return m_generation_stream->can_read();
}

std::unordered_map<uint64_t, GenerationOutput> GenerationHandle::read() {
    return m_generation_stream->read();
}

void add_output_chunk(std::vector<std::vector<int64_t>>& outputs_chunks, std::vector<uint32_t>& index_table, int64_t token_id, size_t max_new_tokens) {
    std::vector<int64_t> output_chunk;
    output_chunk.reserve(max_new_tokens);
    output_chunk.push_back(token_id);
    index_table.push_back(outputs_chunks.size());
    outputs_chunks.push_back(output_chunk);
}

struct IntermediateResult {
    std::vector<uint32_t> output_chunks_indexes;
    float cumulative_log_prob;

    IntermediateResult(const std::vector<uint32_t>& output_chunks_indexes, const float cumulative_log_prob) :
    output_chunks_indexes(output_chunks_indexes),
    cumulative_log_prob(cumulative_log_prob) {}
};

std::vector<GenerationRawResult> GenerationHandle::read_all() {
    std::vector<GenerationRawResult> results;

    std::vector<std::vector<int64_t>> outputs_chunks; 
    std::unordered_map<uint64_t, IntermediateResult> intermediate_results;

    // keeping last results to track sequences that are discontinued
    std::unordered_map<uint64_t, GenerationOutput> last_iteration_results;
    while (!generation_finished() || can_read()) {
        std::unordered_map<uint64_t, GenerationOutput> iteration_results = read();

        std::unordered_map<uint64_t, IntermediateResult> new_sequences;
        std::unordered_set<uint64_t> forked_sequences_ids;

        // In the first generation sequence #1 is a parent of other sequences, but since the fork happens before tokens are added
        // we treat all of the sequences as roots. We skip to the next iteration once they have separate chunk each.
        if (last_iteration_results.size() == 0) {
            for (auto& result : iteration_results) {
                std::vector<uint32_t> output_chunks_indexes;
                add_output_chunk(outputs_chunks, output_chunks_indexes, result.second.token_id, m_sampling_params.max_new_tokens);
                IntermediateResult intermediate_result{output_chunks_indexes, result.second.cumulative_log_prob};
                intermediate_results.emplace(result.first, intermediate_result);
            }
            last_iteration_results = iteration_results;
            continue;
        }

        // First iterate over new sequences
        // forked sequences will reuse output chunks of their roots
        // root sequences will continue on a new chunks in second phase
        for (auto& result : iteration_results) {
            uint64_t sequence_id = result.first;
            GenerationOutput generation_output = result.second;
            if (intermediate_results.find(sequence_id) == intermediate_results.end()) {
                std::vector<uint32_t> output_chunks_indexes;
                if (generation_output.parent_id) {
                    output_chunks_indexes = intermediate_results.at(generation_output.parent_id).output_chunks_indexes;
                    forked_sequences_ids.emplace(generation_output.parent_id);
                }
                add_output_chunk(outputs_chunks, output_chunks_indexes, generation_output.token_id, m_sampling_params.max_new_tokens);
                IntermediateResult intermediate_result{output_chunks_indexes, result.second.cumulative_log_prob};
                new_sequences.emplace(sequence_id, intermediate_result);
            }
        }

        // Second iteration will update results of sequences already in the map
        // iteration happens twice, because we need to iterate over new sequences only first 
        // to determine which already processed sequences had been forked
        for (auto& iteration_result : iteration_results) {
            uint64_t sequence_id = iteration_result.first;
            GenerationOutput generation_output = iteration_result.second;
            auto intermediate_result_iter = intermediate_results.find(sequence_id);
            if (intermediate_result_iter != intermediate_results.end()) {
                auto& output_chunks_indexes = intermediate_result_iter->second.output_chunks_indexes;
                if (forked_sequences_ids.find(sequence_id) != forked_sequences_ids.end()) {
                    add_output_chunk(outputs_chunks, output_chunks_indexes, generation_output.token_id, m_sampling_params.max_new_tokens);
                } else {
                    outputs_chunks[output_chunks_indexes[output_chunks_indexes.size()-1]].push_back(generation_output.token_id);
                }
            }
        }
        // Drop sequences that are discontinued
        for (auto& last_result : last_iteration_results) {
            if (iteration_results.find(last_result.first) == iteration_results.end()) {
                intermediate_results.erase(last_result.first);
            }
        }
        // Insert new sequences
        intermediate_results.insert(new_sequences.begin(), new_sequences.end());
        // Update last results
        last_iteration_results = iteration_results;
    }

    // Gather final results in a single vector 
    for (auto& intermediate_result : intermediate_results) {
        results.emplace_back();
        auto result_index = results.size() - 1;
        auto * sequence_tokens = &results[result_index].generated_token_ids;
        for (uint32_t output_chunk_index: intermediate_result.second.output_chunks_indexes) {
            sequence_tokens->insert(sequence_tokens->end(), outputs_chunks[output_chunk_index].begin(), outputs_chunks[output_chunk_index].end());
        }
        results[result_index].cumulative_log_prob = intermediate_result.second.cumulative_log_prob;
    }
    return results;
}
