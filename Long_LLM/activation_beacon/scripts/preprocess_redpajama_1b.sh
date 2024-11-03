python -m main.preprocess_redpajama_1b \
    --train_data data/redpajama_1b.py \
    --output_path data/redpajama_1b_llama2_8k \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --num_token_per_example 8192 \
    --add_bos \
    --add_eos
