import time
from typing import Any, Dict

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.runtime import ModelRunnerCpp
import numpy as np
import csv

import ray
from transformers import LlamaTokenizerFast

import torch

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote(num_gpus=8)
class TrtLLMClient(LLMClient):
    """Client for TensorRT LLM API."""

    runner = None
    tokenizer = None

    engine_dir = ""  # path to the engine directory
    max_output_len = 2048

    def __init__(self, engine_dir: str = ""):

        # read engine dir from config file
        # example of config file:
        # /path/to/engine/dir
        if engine_dir is not None:
            self.engine_dir = engine_dir
        else:
            with open("trtllm_config.txt", "r") as f:
                self.engine_dir = f.read().strip()

        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )

        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices available: {num_devices}\n")

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

        metrics = {}
        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        generated_text = ""
        output_throughput = 0
        total_request_time = 0

        start_time = time.monotonic()

        batch_input_ids = parse_input(tokenizer=self.tokenizer,
                                      input_text=request_config.prompt,
                                      add_special_tokens=False,
                                      model_name=request_config.model,
                                      max_input_length=2048)

        try:
            with torch.no_grad():
                outputs = self.runner.generate(
                    batch_input_ids,
                    max_new_tokens=2048
                )
        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = str(e)

        output_ids = outputs['output_ids']
        generated_text = self.tokenizer.decode(output_ids)

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
            # for curr_text in input_text:
            print(input_text)
            if prompt_template is not None:
                input_text = prompt_template.format(input_text=input_text)
            input_ids = tokenizer.encode(
                input_text,
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
