# CUDA_VISIBLE_DEVICES=2 python test_easy_retrieval_generate.py

# run all Idea-1 integration tests
# CUDA_VISIBLE_DEVICES=2 pytest -q tests_integration

# run a single test and show logs
# CUDA_VISIBLE_DEVICES=2 pytest -q tests_integration/test_easy_retrieval_generate.py -s
# CUDA_VISIBLE_DEVICES=2 pytest -q tests_integration/test_easy_hierarchy.py -s
# CUDA_VISIBLE_DEVICES=2 pytest -q tests_integration/test_medium_verify_and_revise.py -s

# CUDA_VISIBLE_DEVICES=2 python perception/caption.py
CUDA_VISIBLE_DEVICES=2 python perception/asr.py
# CUDA_VISIBLE_DEVICES=2 python tests_integration/test_textbag_integration.py