# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import math
from typing import Dict

from pathlib import Path
from openvino_genai import ContinuousBatchingPipeline, GenerationConfig, Tokenizer

from common import get_default_properties, get_hugging_face_models, convert_models, generate_and_compare_with_reference_text, \
    get_scheduler_config, get_greedy, run_cb_pipeline_with_ref, get_beam_search, get_greedy, \
    get_multinomial_all_parameters, get_multinomial_temperature_and_num_return_sequence, \
    get_multinomial_temperature_and_top_k, get_multinomial_temperature, get_multinomial_temperature_and_top_p
from test_sampling import RandomSamplingTestStruct, get_current_platform_ref_texts

from ov_genai_test_utils import (
    get_chat_models_list,
    read_model,
    get_continuous_batching,
)

def read_models_list(file_name: str):
    models = []
    with open(file_name) as f:
        for model_name in f:
            model_name = model_name.strip()
            # skip comment in model scope file
            if model_name.startswith('#'):
                continue
            models.append(model_name)
    return models

#
# e2e tests on random and real models
#

@pytest.mark.precommit
@pytest.mark.parametrize("model_id", read_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "precommit")))
def test_e2e_precommit(tmp_path, model_id):
    run_cb_pipeline_with_ref(tmp_path, model_id)


@pytest.mark.nightly
@pytest.mark.parametrize("model_id", read_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "nightly")))
def test_e2e_nightly(tmp_path, model_id):
    run_cb_pipeline_with_ref(tmp_path, model_id)


@pytest.mark.real_models
@pytest.mark.parametrize("model_id", read_models_list(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "real_models")))
def test_e2e_real_models(tmp_path, model_id):
    run_cb_pipeline_with_ref(tmp_path, model_id)

#
# Comparison with stateful
# TODO: remove these tests once test_llm_pipeline.py are generalized and parametrized to test both Stateful and PA paths
#

test_configs = [
    dict(max_new_tokens=20),
    dict(max_new_tokens=200, ignore_eos=True),
    dict(max_new_tokens=20, num_beam_groups=3, num_beams=15, diversity_penalty=1.0)
]
batched_prompts = [
    ['table is made', 'They sky is blue because', 'Difference between Jupiter and Mars is that'],
    ['hello', 'Here is the longest nowel ever: '],
    ['Alan Turing was a', 'return 0', '你好！ 你好嗎？'],
    ['table is made', 'table is made [force left pad tokens]']
]
@pytest.mark.parametrize("generation_config", test_configs)
@pytest.mark.parametrize("prompt", batched_prompts[1:])  # num_beams=15 diverges on the first prompt.
@pytest.mark.precommit
def test_continuous_batching_vs_stateful(prompt, generation_config):
    model_id, path, tokenizer, model, stateful = read_model((
        "facebook/opt-125m",
        Path("opt-125m")
    ))
    cb_pipe = get_continuous_batching(path)
    generated = cb_pipe.generate(prompt, **generation_config)
    reference = stateful.generate(prompt, **generation_config)
    assert generated.texts == reference.texts
    if 1 != generation_config.get("num_return_sequences", 1):
        # Stateful puts zeroes to generated.scores. Don't compare them.
        for gen, ref in zip(generated.scores, reference.scores):
            assert math.isclose(gen, ref, abs_tol=0.0003)


prompts = ['The Sun is yellow because', 'Difference between Jupiter and Mars is that', 'table is made of']
@pytest.mark.parametrize("prompt", prompts)
@pytest.mark.precommit
def test_cb_streamer_vs_return_vs_stateful(prompt):
    model_id, path, hf_tokenizer, opt_model, ov_pipe = read_model((
        "facebook/opt-125m",
        Path("opt-125m")
    ))
    cb_pipe = get_continuous_batching(path)
    streamed = []
    generated = cb_pipe.generate(prompt, max_new_tokens=20, streamer=lambda subword: streamed.append(subword))
    reference = ov_pipe.generate(prompt, max_new_tokens=20)
    assert generated == "".join(streamed)
    assert "".join(streamed) == reference


generation_configs = [
    dict(do_sample=False, max_new_tokens=20),
    dict(do_sample=False, num_beam_groups=3, num_beams=15, num_return_sequences=1, max_new_tokens=10, diversity_penalty=1.0, repetition_penalty=1.0)
]
questions = [
    '1+1=',
    'What is the previous answer?',
    'Why is the Sun yellow?',
    'What was my first question?'
]
@pytest.mark.parametrize("generation_config_kwargs", generation_configs[1:])
@pytest.mark.parametrize("model_descr", get_chat_models_list())
@pytest.mark.precommit
def test_chat_scenario_vs_stateful(model_descr, generation_config_kwargs: Dict):
    model_id, models_path, hf_tokenizer, opt_model, ov_pipe = read_model((model_descr[0], model_descr[1] / '_test_chat'))
    cb_pipe = get_continuous_batching(models_path)

    ov_pipe.start_chat()
    cb_pipe.start_chat()

    generation_config = GenerationConfig(**generation_config_kwargs)
    ov_pipe.set_generation_config(generation_config)

    for question in questions:
        generated = cb_pipe.generate(question, generation_config=generation_config)
        reference = ov_pipe.generate(question)
        assert generated == reference

    # Test that finish_chat() doesn't fail just in case.
    cb_pipe.finish_chat()

#
# Stress tests to check OOM case
#

@pytest.mark.precommit
@pytest.mark.parametrize("sampling_config", [get_greedy(), get_beam_search(), get_multinomial_all_parameters()],
                         ids=["greedy", "beam_search", "multinomial_all_parameters"])
def test_post_oom_health(tmp_path, sampling_config):
    generation_config = sampling_config
    generation_config.ignore_eos = True
    generation_config.max_new_tokens = 1000000

    scheduler_config = get_scheduler_config()
    scheduler_config.num_kv_blocks = 10 # Low cache size to trigger OOM quickly

    model_id : str = "facebook/opt-125m"
    opt_model, hf_tokenizer = get_hugging_face_models(model_id)

    models_path : Path = tmp_path / model_id
    convert_models(opt_model, hf_tokenizer, models_path)

    cb_pipe = ContinuousBatchingPipeline(models_path, Tokenizer(models_path), scheduler_config, "CPU", **get_default_properties())

    # First run should return incomplete response
    output = cb_pipe.generate(["What is OpenVINO?"], [generation_config])
    assert (len(output))
    assert (len(output[0].m_generation_ids))

    # Same for the second run, here we want to make sure the cleanup works and we have free blocks after recent OOM
    output = cb_pipe.generate(["What is OpenVINO?"], [generation_config])
    assert (len(output))
    assert (len(output[0].m_generation_ids))

#
# Pre-emption
#

def get_parallel_sampling_seq_len_300() -> GenerationConfig:
    generation_config = GenerationConfig()
    # TODO: add generation_config.generator and return parameters below
    # generation_config.num_return_sequences = 3
    # generation_config.do_sample = True
    # generation_config.top_k = 10
    # generation_config.top_p = 0.5
    generation_config.max_new_tokens = 300
    return generation_config

def get_beam_search_seq_len_300() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_beam_groups = 3
    generation_config.num_beams = 6
    generation_config.diversity_penalty = 1
    generation_config.max_new_tokens = 300
    generation_config.num_return_sequences = generation_config.num_beams
    return generation_config

scheduler_params_list = [({"num_kv_blocks": 2, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_greedy()),
                         ({"num_kv_blocks": 2, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_greedy()),
                         ({"num_kv_blocks": 10, "dynamic_split_fuse": True}, get_parallel_sampling_seq_len_300()),
                         ({"num_kv_blocks": 10, "dynamic_split_fuse": False}, get_parallel_sampling_seq_len_300()),
                         ({"num_kv_blocks": 34, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_beam_search()),
                         ({"num_kv_blocks": 34, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}, get_beam_search()),
                         ({"num_kv_blocks": 100, "dynamic_split_fuse": True}, get_beam_search_seq_len_300()),
                         ({"num_kv_blocks": 100, "dynamic_split_fuse": False}, get_beam_search_seq_len_300())]
@pytest.mark.parametrize("params", scheduler_params_list)
@pytest.mark.precommit
def test_preemption(tmp_path, params):
    run_cb_pipeline_with_ref(tmp_path, "facebook/opt-125m", scheduler_params=params[0], generation_config=params[1])


multinomial_params = RandomSamplingTestStruct(
    generation_config=[
        get_multinomial_temperature(),
        get_multinomial_temperature_and_top_p(),
        get_multinomial_temperature_and_top_k(),
    ],
    prompts=[
        "What is OpenVINO?",
        "How are you?",
        "Tell me something about Canada?",
    ],
    ref_texts=get_current_platform_ref_texts({
        "linux": [
            [
                "\n\nOpenVINO is a live platform that allows users to create and manage a new library for open source applications.\n\nOpenVINO is"
            ],
            [
                "  You're getting much better results from doing this, than you are by not doing this.  I have a BH and I was so far"
            ],
            [
                "\nI'm from Canada, and I'm from the US, so I'm not sure.\nI think you mean the Canadian version."
            ],
        ],
        "win32": [
            [
                "\n\nOpenVINO is a live platform that allows users to create and manage a new library of applications on the Virtuoso server, which can"
            ],
            [
                "  You're getting much better results from doing this, than you are by not doing this.  If you are truly trying to do something good,"
            ],
            [
                "\nI'm from Canada, and I'm from the US, so I'm not sure what you're talking about.\nI'm Canadian and I"
            ],
        ],
    }),
)


# todo: Anastasiia Pnevskaya: fix the test because it is hanging according max_new_tokens = std::numeric_limits<std::size_t>::max()
@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.precommit
@pytest.mark.skip(reason="Random sampling results are non deterministic due to: discrete_distribution impl depends on platform, model inference results may depend on CPU. Test passes on CI but fails locally.")
def test_preemption_with_multinomial(tmp_path, dynamic_split_fuse):
    generation_configs = multinomial_params.generation_config
    for config in generation_configs:
        config.max_new_tokens = 30
    model_id : str = "facebook/opt-125m"
    model, hf_tokenizer = get_hugging_face_models(model_id)

    models_path : Path = tmp_path / model_id
    convert_models(model, hf_tokenizer, models_path)

    scheduler_config = get_scheduler_config({"num_kv_blocks": 3, "dynamic_split_fuse": dynamic_split_fuse, "max_num_batched_tokens": 256, "max_num_seqs": 256})
    generate_and_compare_with_reference_text(models_path, multinomial_params.prompts, multinomial_params.ref_texts, generation_configs, scheduler_config)


multinomial_params_n_seq = RandomSamplingTestStruct(
    generation_config=[
        get_multinomial_temperature(),
        get_multinomial_temperature_and_num_return_sequence(),
        get_multinomial_all_parameters(),
    ],
    prompts=[
        "Artificial intelligence ",
        "What is the current",
        "Tell me something about UAE?",
    ],
    ref_texts=get_current_platform_ref_texts({
        "linux": [
            [
                "\nI've seen this expression used too many times without making sense.\nAs an AI engineer, and as a scientist, we should make everything easier"
            ],
            [
                " position of the Z-shaped groove?\n0.41\nWhat is the current position of the Z-shaped groove?\n0.11\n",
                " status of all of this? I can't stop thinking about it.\nIt's been a while since I've seen it. I found it a",
                " status of your blog? Do you accept feedback?\nYes, I’m happy to accept feedback at this time (I’m a"
            ],
            [
                "\nIt's in the middle of nowhere if you haven’t seen one yet! It might be more convenient there than anywhere else.. maybe take",
                "\nUAE is a country with some great culture that has been living under Islamic oppression for almost 60 years now (including 20 years as part of Arab",
                "\nNope, just wanted to say how awesome and beautiful it was when my brother came back from an adventure trip across Asia - our 2nd year",
                "\nI don't know anything.  I'm not sure what kind this sub wants though... but apparently they are pretty bad at making videos/photos",
            ],
        ],
        "win32": [
            [
                "\nI've had a friend with the capacity to test this in his own words.\nThe big problem with real-world results is the economics of"
            ],
            [
                " position of the patent application number of the present invention?\n\nIn the present invention, the present invention relates to an improved method for manufacturing a semic",
                " status of your town? How many houses do you have?\nThere are about three houses in our town. The closest place to us is about 25",
                " status of all the other passengers?\nWe're the only ones left, so no...\nI don't think they'll really leave.\nThey"
            ],
            [
                "\nI don't have any knowledge on them. We are based out near Dubai so hopefully they will take care of us soon enough :) thanks though :",
                "\nUAE is not one of the richest countries in Asia but definitely among those most corrupt nations because this corruption (and its own endemic practices) still",
                "\nNope, I'm just going through my first semester there right now and it was nice to see some people who were doing well haha - we",
                "\nIt's a country where your parents can never give you anything at all!  It also has an extremely low education system for many years... You",
            ],
        ],
        "darwin": [
            [
                "\nI've had a friend with the capacity to test this in his own words.\nThe big problem with real-world results is the rigidity"
            ],
            [
               " position of the patent application number of the present invention?\n\nIn the present invention, the present invention relates to an improved method for manufacturing a semic",
               " status of your town? How many houses do you have?\nThere are about three houses in our town. The closest place to us is about 25",
               " status of all the other passengers?\nWe're the only ones left, so no...\nI don't think they'll really leave.\nThey"
            ],
            [
                "\nI don't have any knowledge on them. We are based out near Dubai so hopefully they will take care of us soon enough :) thanks though :",
                "\nUAE is not one of the richest countries in Asia but definitely among those most corrupt nations because this corruption (and its own endemic practices) still",
                "\nNope, I'm just going through my first semester there right now and it was nice to see some people who were doing well haha - we",
                "\nIt's a country where your parents can never give you anything at all!  It also has an extremely low education system for many years... You",
            ],
        ],
    }),
)


@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.precommit
@pytest.mark.skip(reason="Random sampling results are non deterministic due to: discrete_distribution impl depends on platform, model inference results may depend on CPU. Test passes on CI but fails locally.")
def test_preemption_with_multinomial_n_seq(tmp_path, dynamic_split_fuse):
    model_id : str = "facebook/opt-125m"
    opt_model, hf_tokenizer = get_hugging_face_models(model_id)

    models_path : Path = tmp_path / model_id
    convert_models(opt_model, hf_tokenizer, models_path)

    # needed kv_blocks - 16 (2 blocks per sequence (30 tokens to generated text + prompt (> 2 tokens)) * (1 + 3 + 4) seq )
    scheduler_config = get_scheduler_config({"num_kv_blocks": 8, "dynamic_split_fuse": dynamic_split_fuse, "max_num_batched_tokens": 256, "max_num_seqs": 256})
    generate_and_compare_with_reference_text(models_path, multinomial_params_n_seq.prompts, multinomial_params_n_seq.ref_texts, multinomial_params_n_seq.generation_config, scheduler_config)
