import os
from trainer.utils import load_yaml
from data_mix.dataset_loaders import load_social_datasets
from data_mix.mixer import build_mixed_dataset
from trainer.train_sft import train

def main(base_model=None):
    cfg = load_yaml("configs/social_only_large.yaml")
    
    # Optional override from CLI
    if base_model:
        cfg["base_model"] = base_model
    
    # Load social datasets (Reddit TIFU + optional TweetSumm)
    social = list(load_social_datasets(cfg))
    news = []
    
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", "--base-model", dest="base_model", default=None,
                   help="Hugging Face model id to use (overrides cfg['base_model'])")
    args = p.parse_args()
    main(args.base_model)


#CUDA_VISIBLE_DEVICES=2 nohup python run_social_large.py > /home/tahad/trustworthy-multimodal-summarization/outputs/results/logs/fine_tune_social_large_Llama-2-13b-chat-hf.log 2>&1 &