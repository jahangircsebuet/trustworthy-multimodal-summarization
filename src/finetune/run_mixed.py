import os
from trainer.utils import load_yaml
from data_mix.dataset_loaders import load_social_datasets, load_news_datasets
from data_mix.mixer import build_mixed_dataset
from trainer.train_sft import train

def main():
    cfg = load_yaml("configs/mixed_domain.yaml")

    social = list(load_social_datasets(cfg))
    news   = list(load_news_datasets(cfg))

    dset = build_mixed_dataset(
        social_sources=social,
        news_sources=news,
        use_domain_tags=cfg["sampling"]["use_domain_tags"],
        social_to_news_ratio=tuple(cfg["sampling"]["social_to_news_ratio"]),
        upsample_social=cfg["sampling"].get("upsample_social", 1),
        max_target_len=cfg["max_target_len"],
        train_ratio=cfg["train_ratio"],
    )
    train(dset, cfg)

if __name__ == "__main__":
    main()
