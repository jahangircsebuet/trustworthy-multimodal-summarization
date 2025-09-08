

def test_budgeted_decoding_with_small_model():
    from src.decoding.budget import BudgetDecoder
    # Use a small HF model as the "model_dir"
    dec = BudgetDecoder("google/flan-t5-small")
    inp = (
        "Bullet points:\n"
        "- The council approved funding for park renovations.\n"
        "- Construction will begin in September.\n"
        "- A public forum is scheduled next week.\n"
        "Write a concise summary."
    )
    out, log = dec.generate_with_budget(inp, budget_tokens=96, profile="fast")
    assert isinstance(out, str) and len(out.split()) > 5
    assert log["budget_tokens"] == 96
