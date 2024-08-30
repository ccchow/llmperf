import json
import os
import time
from typing import Any, Dict

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner, ModelRunnerCpp

import ray
import requests
from transformers import LlamaTokenizerFast

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics

@ray.remote
class TrtLLMClient(LLMClient):
    """Client for TensorRT LLM API."""

    runner = None
    tokenizer = None

    def __init__(self):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )

        runtime_rank = tensorrt_llm.mpi_rank()
        runner_cls = ModelRunnerCpp
        runner_kwargs = dict(
            engine_dir=args.engine_dir,
            lora_dir=args.lora_dir,
            rank=runtime_rank,
            debug_mode=args.debug_mode,
            lora_ckpt_source=args.lora_ckpt_source,
            max_output_len=args.max_output_len,
        )

        runner = runner_cls.from_dir(**runner_kwargs)

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

        with torch.no_grad():
            outputs = runner.generate(
                batch_input_ids,
                max_new_tokens=args.max_output_len,
                max_attention_window_size=args.max_attention_window_size,
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
                output_cum_log_probs=(args.output_cum_log_probs_npy !=
                                        None),
                output_log_probs=(args.output_log_probs_npy != None),
                random_seed=args.random_seed,
                lora_uids=args.lora_task_uids,
                prompt_table=args.prompt_table_path,
                prompt_tasks=args.prompt_tasks,
                streaming=args.streaming,
                output_sequence_lengths=True,
                return_dict=True,
                return_all_generated_tokens=args.return_all_generated_tokens
            )

            output_text = self.tokenizer.decode(outputs)
        
            if tensorrt_llm.mpi_rank() == 0:
                output_ids = outputs['output_ids']
                sequence_lengths = outputs['sequence_lengths']
                context_logits = None
                generation_logits = None
                cum_log_probs = None
                log_probs = None
                if self.runner.gather_context_logits:
                    context_logits = outputs['context_logits']
                if self.runner.gather_generation_logits:
                    generation_logits = outputs['generation_logits']
                if args.output_cum_log_probs_npy != None:
                    cum_log_probs = outputs['cum_log_probs']
                if args.output_log_probs_npy != None:
                    log_probs = outputs['log_probs']
                print_output(self.tokenizer,
                            output_ids,
                            input_lengths,
                            sequence_lengths,
                            output_csv=args.output_csv,
                            output_npy=args.output_npy,
                            context_logits=context_logits,
                            generation_logits=generation_logits,
                            output_logits_npy=args.output_logits_npy,
                            cum_log_probs=cum_log_probs,
                            log_probs=log_probs,
                            output_cum_log_probs_npy=args.output_cum_log_probs_npy,
                            output_log_probs_npy=args.output_log_probs_npy)

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

        with torch.no_grad():
            tensorrt_llm_llama.setup(
                batch_size=args.batch_size,
                max_context_length=max_length,
                max_new_tokens=args.output_len,
                beam_width=args.num_beams,
                max_attention_window_size=args.max_attention_window_size,
                multi_block_mode=args.multi_block_mode,
                enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc)
            logger.info(f"Generation session set up with the parameters: \
                batch_size: {tensorrt_llm_llama.batch_size}, \
                max_context_length: {tensorrt_llm_llama.max_context_length}, \
                max_new_tokens: {tensorrt_llm_llama.max_new_tokens}, \
                beam_width: {tensorrt_llm_llama.beam_width}, \
                max_attention_window_size: {tensorrt_llm_llama.max_attention_window_size}, \
                multi_block_mode: {tensorrt_llm_llama.multi_block_mode}, \
                enable_context_fmha_fp32_acc: {tensorrt_llm_llama.enable_context_fmha_fp32_acc}"
                        )

            if tensorrt_llm_llama.remove_input_padding:
                output_ids = tensorrt_llm_llama.decode_batch(
                    line_encoded, sampling_config)
            else:
                output_ids = tensorrt_llm_llama.decode(
                    line_encoded,
                    input_lengths,
                    sampling_config,
                )
                torch.cuda.synchronize()

        return metrics, generated_text, request_config
