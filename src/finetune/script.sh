# Fine-tune on social datasets only:
# will run only on social network datasets
CUDA_VISIBLE_DEVICES=2 python /home/malam10/projects/trustworthy-multimodal-summarization/src/finetune/run_social.py

# Fine-tune on social + news datasets (70:30 ratio with domain tags):
# will run on social network  + news/media datasets
# CUDA_VISIBLE_DEVICES=2 python /home/malam10/projects/trustworthy-multimodal-summarization/src/finetune/run_mixed.py
