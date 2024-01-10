# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import argparse
from pathlib import Path
import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import ModelRunnerCpp
from transformers import AutoTokenizer

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        help=
        'The attention window size that controls the sliding window attention / cyclic kv cache behaviour'
    )
    parser.add_argument('--sink_token_length',
                        type=int,
                        default=None,
                        help='The sink token length.')
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='engine_outputs')
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["Born in north-east France, Soyer trained as a"])
    parser.add_argument(
        '--no_prompt_template',
        dest='use_prompt_template',
        default=True,
        action='store_false',
        help=
        "Whether or not to use default prompt template to wrap the input text.")
    parser.add_argument(
        '--input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--output_logits_npy',
        type=str,
        help=
        'Numpy file where the generation logits are stored. Use only when num_beams==1',
        default=None)
    parser.add_argument('--tokenizer_dir',
                        help="HF tokenizer config path",
                        default='gpt2')
    parser.add_argument(
        '--tokenizer_type',
        help=
        'Specify that argument when providing a .model file as the tokenizer_dir. '
        'It allows AutoTokenizer to instantiate the correct tokenizer type.')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers")
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--debug_mode',
                        default=False,
                        action='store_true',
                        help="Whether or not to turn on the debug mode")
    parser.add_argument('--no_add_special_tokens',
                        dest='add_special_tokens',
                        default=True,
                        action='store_false',
                        help="Whether or not to add special tokens")
    parser.add_argument('--streaming', default=False, action='store_true')
    parser.add_argument('--streaming_interval',
                        type=int,
                        help="How often to return tokens when streaming.",
                        default=5)
    parser.add_argument(
        '--prompt_table_path',
        type=str,
        help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
        '--prompt_tasks',
        help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
    parser.add_argument('--lora_dir',
                        type=str,
                        default=None,
                        help="The directory of LoRA weights")
    parser.add_argument(
        '--lora_task_uids',
        type=str,
        default=None,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module")
    parser.add_argument('--lora_ckpt_source',
                        type=str,
                        default="hf",
                        choices=["hf", "nemo"],
                        help="The source of lora checkpoint.")

    parser.add_argument(
        '--num_prepend_vtokens',
        nargs="+",
        type=int,
        help="Number of (default) virtual tokens to prepend to each sentence."
        " For example, '--num_prepend_vtokens=10' will prepend the tokens"
        " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")

    return parser.parse_args(args=args)

def parse_input(tokenizer,
                input_text=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=128,
                pad_id=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if input_file is None:
        for curr_text in input_text:
            input_ids = tokenizer.encode(curr_text,
                                         add_special_tokens=add_special_tokens,
                                         truncation=True,
                                         max_length=max_input_length)
            batch_input_ids.append(input_ids)
    else:
        with open(input_file, 'r', encoding='utf-8',
                    errors='replace') as txt_file:
            input_text = txt_file.read()
            input_ids = tokenizer.encode(
                input_text,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=max_input_length)
            batch_input_ids.append(input_ids)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids

def print_output(tokenizer,
                 output_ids,
                 input_lengths,
                 sequence_lengths):
    batch_size, num_beams, _ = output_ids.size()
    for batch_idx in range(batch_size):
        inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
        )
        input_text = tokenizer.decode(inputs)
        print(f'Input [Text {batch_idx}]: \"{input_text}\"')
        for beam in range(num_beams):
            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[batch_idx][beam]
            outputs = output_ids[batch_idx][beam][
                output_begin:output_end].tolist()
            output_text = tokenizer.decode(outputs)
            print(
                f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')

def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id

    batch_input_ids = parse_input(tokenizer=tokenizer,
                                  input_text=args.input_text,
                                  input_file=args.input_file,
                                  add_special_tokens=args.add_special_tokens,
                                  max_input_length=args.max_input_length,
                                  pad_id=pad_id)
    input_lengths = [x.size(0) for x in batch_input_ids]

    runner_kwargs = dict(engine_dir=args.engine_dir,
                         lora_dir=args.lora_dir,
                         rank=runtime_rank,
                         debug_mode=args.debug_mode,
                         lora_ckpt_source=args.lora_ckpt_source)
    runner = ModelRunnerCpp.from_dir(**runner_kwargs)

    start = time.time()
    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids,
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
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            lora_uids=args.lora_task_uids,
            prompt_table_path=args.prompt_table_path,
            prompt_tasks=args.prompt_tasks,
            streaming=args.streaming,
            output_sequence_lengths=True,
            return_dict=True)
        torch.cuda.synchronize()
    end = time.time() 
    print(f'Time: {end - start}')
    
    output_ids = outputs['output_ids']
    sequence_lengths = outputs['sequence_lengths']
    print_output(tokenizer, output_ids, input_lengths, sequence_lengths)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
