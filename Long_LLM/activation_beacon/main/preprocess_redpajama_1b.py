from dataclasses import dataclass, field

import datasets
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class Args:
    output_path: str = field(
        metadata={"help": "Output path for preprocessed dataset."},
    )
    tokenizer_name_or_path: str = field(
        metadata={"help": "Tokenizer name."},
    )
    train_data: str = field(
        default="data/redpajama_1b.py",
        metadata={
            "help": "Directory of training data (multiple json files whose name correspond to the ones in config)."
        },
    )

    num_token_per_example: int = field(
        default=8192, metadata={"help": "Number of tokens per example."}
    )
    add_bos: bool = field(
        default=False, metadata={"help": "Add bos at the end of each document?"}
    )
    add_eos: bool = field(
        default=True, metadata={"help": "Add eos at the end of each document?"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed."})

    num_proc: int = field(default=64, metadata={"help": "Number of processes to use."})


def tokenize_fn(x, tokenizer):
    input_ids = tokenizer(x["text"], add_special_tokens=False)["input_ids"]
    return {"input_ids": input_ids}


def main():
    parser = HfArgumentParser([Args])
    args: Args = parser.parse_args_into_dataclasses()[0]

    raw_dataset = load_dataset(args.train_data)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    outputs = {"input_ids": [], "attention_mask": [], "labels": [], "length": []}
    for split in raw_dataset.keys():
        tokenized_dataset = raw_dataset[split].map(
            tokenize_fn,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=raw_dataset[split].column_names,
            fn_kwargs={"tokenizer": tokenizer},
            desc=f"Tokenizing {split} split...",
        )

        input_ids = []
        for x in tqdm(tokenized_dataset, desc=f"Processing {split} split..."):
            sample_input_ids = x["input_ids"]
            if args.add_bos:
                assert (
                    tokenizer.bos_token_id is not None
                ), "Make sure the bos_token_id exists when enable add_eos."
                sample_input_ids = [tokenizer.bos_token_id] + sample_input_ids
            if args.add_eos:
                assert (
                    tokenizer.eos_token_id is not None
                ), "Make sure the eos_token_id exists when enable add_eos."
                sample_input_ids = sample_input_ids + [tokenizer.eos_token_id]

            input_ids.extend(sample_input_ids)

            if len(input_ids) >= args.num_token_per_example:
                cursor = 0
                while cursor + args.num_token_per_example <= len(input_ids):
                    instance_input_ids = input_ids[
                        cursor : cursor + args.num_token_per_example
                    ].copy()
                    instance_attention_mask = [1 for _ in instance_input_ids]
                    instance_labels = instance_input_ids.copy()

                    # move the cursor
                    cursor += args.num_token_per_example

                    # add to final data
                    outputs["input_ids"].append(instance_input_ids)
                    outputs["attention_mask"].append(instance_attention_mask)
                    outputs["labels"].append(instance_labels)
                    outputs["length"].append(args.num_token_per_example)

                # remove input_ids that have been saved in outputs
                input_ids = input_ids[cursor:]

    print(f"Saving preprocessed datasets to {args.output_path}...")
    preprocessed_dataset = datasets.Dataset.from_dict(outputs)
    preprocessed_dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
