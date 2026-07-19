"""DeepSeek integration: model resolution, token budgeting, cost logging.

Separate from the search pipeline on purpose. The pipeline runs with no API key
and no cost; only resume tailoring will need what lives here, so importing this
package should never be a prerequisite for searching.

    from jobbuddy.deepseek import model_config, deepseek_client

    model = model_config.resolve("fast")        # never hardcode a model id
    reply = deepseek_client.complete(prompt, profile="extract")
"""

__all__ = [
    "deepseek_client", "deepseek_common", "model_config", "token_budget",
    "update_models", "calibrate_budgets", "fetch_tokenizer",
]
