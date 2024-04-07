import argparse

import transformers
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="huggungface transformers training")

    # transformers
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--evaluation_strategy", type=str, default="epoch")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--fp16", type=bool, default=False)
    # paths
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()

    # model, tokenizer のロード
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
    )

    # example データセットのロード
    raw_datasets = load_dataset("glue", "mrpc")

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # トレーニングの設定
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy=args.evaluation_strategy,
        logging_dir=args.logging_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        weight_decay=args.weight_decay,
        save_strategy=args.save_strategy,
        fp16=args.fp16,
    )

    trainer = transformers.Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
