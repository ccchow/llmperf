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

import torch

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics

@ray.remote
class TrtLLMClient(LLMClient):
    """Client for TensorRT LLM API."""

    runner = None
    tokenizer = None

    engine_dir = None
    max_output_len = 2048

    def __init__(self):
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )

        runtime_rank = tensorrt_llm.mpi_rank()
        runner_cls = ModelRunnerCpp
        runner_kwargs = dict(
            engine_dir=self.engine_dir,
            rank=runtime_rank,
            max_output_len=self.max_output_len,
        )

        self.runner = runner_cls.from_dir(**runner_kwargs)

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

        batch_input_ids = parse_input(tokenizer=self.tokenizer,
                                    input_text=request_config.prompt,
                                    add_special_tokens=False,
                                    max_input_length=2048)

        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids,
                max_new_tokens=2048,
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

        return metrics, generated_text, request_config

def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None,
                model_version=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if input_file is None:
        if 'whisper' in model_name.lower():
            batch_input_ids.append(tokenizer.prefix_tokens)
        else:
            for curr_text in input_text:
                if prompt_template is not None:
                    curr_text = prompt_template.format(input_text=curr_text)
                input_ids = tokenizer.encode(
                    curr_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length)
                batch_input_ids.append(input_ids)
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_ids = np.array(line, dtype='int32')
                    batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                input_ids = row[row != pad_id]
                batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.txt'):
            with open(input_file, 'r', encoding='utf-8',
                      errors='replace') as txt_file:
                input_text = txt_file.readlines()
                batch_input_ids = tokenizer(
                    input_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length)["input_ids"]
        else:
            print('Input file format not supported.')
            raise SystemExit

    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(
            tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(
                range(base_vocab_size,
                      base_vocab_size + length)) + batch_input_ids[i]

    if input_file is None and 'GLM' in model_name and model_version == 'glm':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids