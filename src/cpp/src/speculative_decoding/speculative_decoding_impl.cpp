// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text_callback_streamer.hpp"
#include "speculative_decoding_impl.hpp"
#include "utils.hpp"
#include "utils/paged_attention_transformations.hpp"


namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

bool are_tokenizers_equal(Tokenizer& lhs, Tokenizer& rhs) {
    std::string test_string = "Could you please tell me something about OpenVINO.GenAI?";
    ov::Tensor encoded_string_lhs = lhs.encode(test_string).input_ids,
               encoded_string_rhs = rhs.encode(test_string).input_ids;
    
    ov::Shape shape_lhs = encoded_string_lhs.get_shape(),
              shape_rhs = encoded_string_rhs.get_shape();

    return shape_lhs == shape_rhs && lhs.get_eos_token_id() == rhs.get_eos_token_id() &&
           lhs.get_bos_token_id() == rhs.get_bos_token_id() && lhs.get_pad_token_id() == rhs.get_pad_token_id();
}

ContinuousBatchingPipeline::SpeculativeDecodingImpl::SpeculativeDecodingImpl(const ov::genai::ModelDesc& main_model_desc, 
                                                                             const ov::genai::ModelDesc& draft_model_desc) {
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;

    auto main_scheduler_config = main_model_desc.scheduler_config;
    auto main_device = main_model_desc.device;

    utils::apply_paged_attention_transformations(main_model, main_model_desc.scheduler_config.use_cache_eviction);
    utils::apply_paged_attention_transformations(draft_model, main_model_desc.scheduler_config.use_cache_eviction);
    utils::apply_gather_before_matmul_transformation(main_model);
    utils::apply_gather_before_matmul_transformation(draft_model);

    std::string draft_device = draft_model_desc.device.empty() ? main_model_desc.device : draft_model_desc.device;
    bool is_draft_scheduler_undefined = draft_model_desc.scheduler_config == SchedulerConfig();

    ov::genai::SchedulerConfig main_scheduler_config_updated = main_scheduler_config,
                               draft_scheduler_config = is_draft_scheduler_undefined ? main_scheduler_config : draft_model_desc.scheduler_config;

    if (is_draft_scheduler_undefined) {
        // split KV cache to 2 caches for main and draft models
        size_t main_model_hidden_size = utils::get_hidden_size(main_model),
               draft_model_hidden_size = utils::get_hidden_size(draft_model);
        auto k = static_cast<float>(draft_model_hidden_size) / (main_model_hidden_size + draft_model_hidden_size);

        size_t main_cache_size = std::ceil(main_scheduler_config.cache_size * (1.f - k)),
               draft_cache_size = main_scheduler_config.cache_size - main_cache_size;
        if (draft_cache_size == 0 && main_cache_size > 0) {
            main_cache_size -= (main_cache_size > 1 ? 1 : 0);
            draft_cache_size = 1;
        }

        main_scheduler_config_updated.cache_size = main_cache_size;
        draft_scheduler_config.cache_size = draft_cache_size;
    }

    ov::AnyMap draft_properties = draft_model_desc.properties.empty() ? main_model_desc.properties : draft_model_desc.properties;

    ov::Core core = utils::singleton_core();
    DeviceConfig main_device_config(core, main_scheduler_config_updated, main_device, main_model_desc.properties),
                 draft_device_config(core, draft_scheduler_config, draft_device, draft_properties);

    utils::set_kv_cache_type_and_shape(main_model, main_device_config);
    utils::set_kv_cache_type_and_shape(draft_model, draft_device_config);

    // main and draft model can have different tokenizers
    // to do: support retokenization: 154103
    Tokenizer main_model_tokenizer = main_model_desc.tokenizer;
    Tokenizer draft_model_tokenizer = draft_model_desc.tokenizer;

    // todo: remove this condition after support of CVS-154103
    OPENVINO_ASSERT(are_tokenizers_equal(main_model_tokenizer, draft_model_tokenizer), "Tokenizers for draft and main models are different!");
    
    m_tokenizer = main_model_tokenizer;

    // to create `main_pipeline` with enabled validation_mode and `draft_pipeline` with disabled validation mode
    m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(core,
        main_model, main_model_tokenizer, main_model_desc.generation_config,
        main_device_config, main_scheduler_config_updated, main_device, main_model_desc.properties, true);
    m_draft_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(core,
        draft_model, draft_model_tokenizer, draft_model_desc.generation_config,
        draft_device_config, draft_scheduler_config, draft_device, draft_properties, false);
}

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const ov::Tensor& input_ids,
                                                                 ov::genai::GenerationConfig sampling_params) {
    m_sd_metrics.set_generated_len(request_id, sampling_params.max_new_tokens);
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, input_ids, draft_sampling_params)});
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params);
};

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const std::string& prompt,
                                                                 ov::genai::GenerationConfig sampling_params) {
    m_sd_metrics.set_generated_len(request_id, sampling_params.max_new_tokens);
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, prompt, draft_sampling_params)});
    return m_main_pipeline->add_request(request_id, prompt, sampling_params);
}

bool ContinuousBatchingPipeline::SpeculativeDecodingImpl::has_non_finished_requests() {
    return m_main_pipeline->has_non_finished_requests();
}

void print_generated_request(const ov::genai::GeneratedRequests& requests) {
    for (const auto& request : requests) {
        for (const auto& sequence : request.second) {
            std::cout << "request_id: " << request.first << " | sequence_id: " << sequence.first << " | ";
            for (const auto& token_id : sequence.second.token_ids) {
                std::cout << token_id << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void ContinuousBatchingPipeline::SpeculativeDecodingImpl::step() {
    // this blocks adding new requests during step as it may break coherence between main and draft models
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};
    m_draft_pipeline->pull_awaiting_requests(true);
    m_main_pipeline->pull_awaiting_requests();

    // generate candidates by draft model
    ManualTimer draft_timer("speculative_decoding: draft_model: multistep()");
    draft_timer.start();
    m_draft_pipeline->multistep();
    draft_timer.end();
    m_sd_metrics.draft_duration += draft_timer.get_duration();
    m_pipeline_metrics = m_main_pipeline->get_metrics();

    // to generate num_matches statistic
    std::map<int64_t, UpdateRequestResult> update_sequence_info;
    // put candidates to model KV cache
    auto draft_generated_requests = m_draft_pipeline->get_generated_requests();
    for (const auto& candidate : m_draft_pipeline->get_generated_requests()) {
        auto update_result = m_main_pipeline->update_request(candidate.first, candidate.second, false);
        update_sequence_info.insert({{candidate.first, update_result}});
    }

    ManualTimer main_timer("speculative_decoding: main_model: step()");
    main_timer.start();
    m_main_pipeline->step();
    main_timer.end();
    m_sd_metrics.main_duration += main_timer.get_duration();
    m_pipeline_metrics = m_main_pipeline->get_metrics();

    auto main_generated_requests = m_main_pipeline->get_generated_requests();
    for (const auto& checked_sequence : main_generated_requests) {
        auto update_result = m_draft_pipeline->update_request(checked_sequence.first, checked_sequence.second, true);
        update_sequence_info[checked_sequence.first].removed_tokens_cnt = update_result.removed_tokens_cnt;
    }

    // finish draft request if the generation was completed
    for (const auto& draft_request : draft_generated_requests) {
        auto request_id = draft_request.first;
        if (!main_generated_requests.count(request_id)) {
            m_draft_pipeline->finish_request(request_id);
            // remove draft_generation_handle from queue
            m_draft_generations.erase(request_id);
        }
        auto updated_seq_info = update_sequence_info[request_id];
        // several prompt phase
        if (updated_seq_info.inserted_tokens_cnt == 0) {
            continue;
        }
        float acceptance_rate = 1 - static_cast<float>(updated_seq_info.removed_tokens_cnt) / updated_seq_info.inserted_tokens_cnt;
        m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate * 100);
        m_sd_metrics.update_draft_accepted_tokens(request_id, (updated_seq_info.inserted_tokens_cnt - updated_seq_info.removed_tokens_cnt));
    }

    if (main_generated_requests.empty() && 0) {
        m_sd_metrics.print(true);
        m_sd_metrics.clean_up();
    }
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::SpeculativeDecodingImpl::generate(const std::vector<ov::Tensor>& input_ids,
                                                              const std::vector<GenerationConfig>& sampling_params,
                                                              const StreamerVariant& streamer) {
    OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());

    ManualTimer generate_timer("speculative_decoding: generate()");
    generate_timer.start();

    // checks that all requests has the same LoRA adapters property value
    for (size_t i = 1; i < sampling_params.size(); ++i) {
        OPENVINO_ASSERT(sampling_params[i - 1].adapters == sampling_params[i].adapters,
            "LoRA adapters value must be the same for all requests");
    }
    m_main_pipeline->set_adapters(sampling_params[0].adapters);
    m_draft_pipeline->set_adapters(sampling_params[0].adapters);

    const std::shared_ptr<StreamerBase>& streamer_ptr = std::visit(overloaded{
        [](std::monostate) -> std::shared_ptr<StreamerBase> {
            return nullptr;
        },
        [](const std::shared_ptr<StreamerBase>& streamer) {
            return streamer;
        },
        [this](const std::function<bool(std::string)>& streamer) -> std::shared_ptr<StreamerBase> {
            return std::make_unique<TextCallbackStreamer>(m_tokenizer, streamer);
        }
    }, streamer);

    OPENVINO_ASSERT(streamer_ptr == nullptr || input_ids.size() == 1 && (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_multinomial()),
        "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    std::vector<GenerationHandle> main_generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        m_sd_metrics.set_generated_len(request_id, sampling_params[request_id].max_new_tokens);
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        main_generations.push_back(m_main_pipeline->add_request(request_id, input_ids[request_id], sampling_params[request_id]));

        auto draft_sampling_params = sampling_params[request_id];
        // set the parameters do not stop draft generation without stopping of the same request for main pipeline
        draft_sampling_params.ignore_eos = true;
        std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
        m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, input_ids[request_id], draft_sampling_params)});
    }

    std::vector<EncodedGenerationResult> results;
    results.reserve(input_ids.size());

    bool continue_generation = true;
    while (has_non_finished_requests() && continue_generation) {
        step();
        if (streamer_ptr) {
            // not generated tokens like several prompt phase
            if (!main_generations.at(0).get()->can_read()) {
                continue;
            }
            std::unordered_map<uint64_t, GenerationOutput> token = main_generations.at(0).get()->back();
            for (const auto& gen_token : token.begin()->second.generated_ids) {
                continue_generation = !streamer_ptr->put(gen_token);
                if (!continue_generation) {
                    break;
                }
            }
        }
    }
    if (streamer_ptr) {
        streamer_ptr->end();
    }

    for (size_t generation_idx = 0; generation_idx < main_generations.size(); ++generation_idx) {
        const auto& generation = main_generations[generation_idx];
        EncodedGenerationResult result;
        result.m_request_id = 1;
        std::vector<GenerationOutput> generation_outputs = generation->read_all();
        std::sort(generation_outputs.begin(), generation_outputs.end(), [=] (GenerationOutput& r1, GenerationOutput& r2) {
            return r1.score > r2.score;
        });

        auto num_outputs = std::min(sampling_params[generation_idx].num_return_sequences, generation_outputs.size());
        for (size_t generation_output_idx = 0; generation_output_idx < num_outputs; ++generation_output_idx) {
            const auto& generation_output = generation_outputs[generation_output_idx];
            m_sd_metrics.set_generated_len(generation_idx, generation_outputs[generation_output_idx].generated_ids.size());
            result.m_generation_ids.push_back(std::move(generation_output.generated_ids));
            result.m_scores.push_back(generation_output.score);
        }
        result.m_status = generation->get_status();
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    generate_timer.end();
    return results;
}

SpeculativeDecodingMetrics
ContinuousBatchingPipeline::SpeculativeDecodingImpl::get_speculative_decoding_metrics() {
    return m_sd_metrics;
};
}
