import json
import os
import time
from typing import Any, Dict

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

import ray
import requests
from transformers import LlamaTokenizerFast

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics

@ray.remote
class TrtLLMClient(LLMClient):
    """Client for TensorRT LLM API."""

    def __init__(self):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        start_time = time.monotonic()

        generated_text = self.runner.generate(
            batch_input_ids=decoder_input_ids
            if is_enc_dec else batch_input_ids,
            encoder_input_ids=encoder_input_ids if is_enc_dec else None,
            encoder_input_features=encoder_input_features
            if is_enc_dec else None,
            encoder_output_lengths=encoder_output_lengths
            if is_enc_dec else None,
            max_new_tokens=args.max_output_len,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            end_id=end_id,
            pad_id=pad_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            output_cum_log_probs=(args.output_cum_log_probs_npy != None),
            output_log_probs=(args.output_log_probs_npy != None),
            random_seed=args.random_seed,
            lora_uids=args.lora_task_uids,
            prompt_table=args.prompt_table_path,
            prompt_tasks=args.prompt_tasks,
            streaming=args.streaming,
            output_sequence_lengths=True,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            return_dict=True,
            medusa_choices=args.medusa_choices,
            return_all_generated_tokens=args.return_all_generated_tokens)

        tokens_received = len(self.tokenizer.encode(generated_text))

        total_request_time = time.monotonic() - start_time
        output_throughput = tokens_received / total_request_time

        metrics = {}
        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config
