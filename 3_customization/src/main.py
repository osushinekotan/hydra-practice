import hydra
from datasets import load_dataset
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # model, tokenizer のロード
    model = hydra.utils.get_method(cfg.transformers.model)(
        cfg.transformers.model_name,
        num_labels=cfg.transformers.num_labels,
    )
    tokenizer = hydra.utils.get_method(cfg.transformers.tokenizer)(cfg.transformers.model_name)

    # example データセットのロード
    raw_datasets = load_dataset("glue", "mrpc")

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    trainer = hydra.utils.instantiate(
        cfg.transformers.trainer,
        model=model,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
