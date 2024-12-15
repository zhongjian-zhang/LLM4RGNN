#!/bin/bash

datasets=('cora')
attacks=('dice')
llm="mistral-7b-merge"
for dataset in "${datasets[@]}"; do
  for attack in "${attacks[@]}"; do
      if [ "$attack" == "meta" ]; then
          ptb_rates=(0 0.05 0.1 0.2)
      else
          ptb_rates=(0.1 0.2 0.4)
      fi
      for ptb_rate in "${ptb_rates[@]}"; do
          python create_instruction.py --llm ${llm} --dataset "${dataset}" --attack "${attack}" --ptb_rate "${ptb_rate}"

          CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.run_batch \
              -i ./instruction/"${dataset}"_"${attack}"_"${ptb_rate}".jsonl \
              -o ./output/"${dataset}"_"${attack}"_"${ptb_rate}".jsonl \
              --tensor_parallel_size 2 \
              --gpu-memory-utilization 0.95 \
              --model ../../saved_model/llm/${llm}

          python process_output.py --llm ${llm} --dataset "${dataset}" --attack "${attack}" --ptb_rate "${ptb_rate}"
      done
  done
done