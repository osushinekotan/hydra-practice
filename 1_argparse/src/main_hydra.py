import hydra
import transformers
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # model, tokenizer のロード
    tokenizer = AutoTokenizer.from_pretrained(cfg.transformers.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.transformers.model_name, num_labels=cfg.transformers.num_labels
    )

    # example データセットのロード
    raw_datasets = load_dataset("glue", "mrpc")

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    # トレーニングの設定
    training_args = transformers.TrainingArguments(
        output_dir=cfg.paths.output_dir,
        per_device_train_batch_size=cfg.transformers.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.transformers.per_device_eval_batch_size,
        evaluation_strategy=cfg.transformers.evaluation_strategy,
        logging_dir=cfg.paths.logging_dir,
        num_train_epochs=cfg.transformers.num_epochs,
        learning_rate=cfg.transformers.learning_rate,
        warmup_ratio=cfg.transformers.warmup_ratio,
        gradient_accumulation_steps=cfg.transformers.gradient_accumulation_steps,
        eval_accumulation_steps=cfg.transformers.eval_accumulation_steps,
        weight_decay=cfg.transformers.weight_decay,
        save_strategy=cfg.transformers.save_strategy,
        fp16=cfg.transformers.fp16,
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
