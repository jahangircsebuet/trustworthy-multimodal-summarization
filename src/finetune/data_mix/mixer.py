# mixer.py
import math, random
from typing import List, Tuple
from datasets import concatenate_datasets, Dataset, DatasetDict
from .formatters import apply_domain_tags, format_chat

def split_train_val(ds: Dataset, train_ratio=0.97) -> Tuple[Dataset, Dataset]:
    n = len(ds)
    n_train = int(n * train_ratio)
    ds = ds.shuffle(seed=42)
    return ds.select(range(n_train)), ds.select(range(n_train, n))

def upsample(ds: Dataset, factor: int) -> Dataset:
    parts = [ds]
    for _ in range(factor-1):
        parts.append(ds.shuffle(seed=42))
    return concatenate_datasets(parts)

def build_mixed_dataset(
    social_sources: List[Dataset],
    news_sources: List[Dataset],
    use_domain_tags: bool = True,
    social_to_news_ratio: Tuple[float, float] = (0.7, 0.3),
    upsample_social: int = 1,
    max_target_len: int = 128,
    train_ratio: float = 0.97,
):
    # merge groups
    social = concatenate_datasets(social_sources) if social_sources else None
    news   = concatenate_datasets(news_sources) if news_sources else None

    if social is None and news is None:
        raise ValueError("No datasets selected.")

    # Apply domain tags
    def tag_map(domain):
        return lambda ex: apply_domain_tags(ex, domain, use_domain_tags)

    if social is not None:
        social = social.map(tag_map("social"), desc="tag social")
    if news is not None:
        news = news.map(tag_map("news"), desc="tag news")

    # Optional upsample social
    if social is not None and upsample_social > 1:
        social = upsample(social, upsample_social)

    # Ratio-based downsampling to target mix
    if social is not None and news is not None:
        total = len(social) + len(news)
        tgt_social = int(total * social_to_news_ratio[0])
        tgt_news   = int(total * social_to_news_ratio[1])

        if len(social) > tgt_social:
            social = social.shuffle(seed=123).select(range(tgt_social))
        if len(news) > tgt_news:
            news = news.shuffle(seed=123).select(range(tgt_news))

        combined = concatenate_datasets([social, news]).shuffle(seed=7)
    else:
        combined = (social or news).shuffle(seed=7)

    # train/val split
    train, val = split_train_val(combined, train_ratio=train_ratio)

    # Format into chat-style fields
    train = train.map(lambda ex: format_chat(ex, max_target_len), desc="format train", remove_columns=train.column_names)
    val   = val.map(lambda ex: format_chat(ex, max_target_len),   desc="format val",   remove_columns=val.column_names)

    return DatasetDict({"train": train, "validation": val})
