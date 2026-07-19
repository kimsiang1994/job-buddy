"""Job Buddy: find Singapore jobs, score them against a verified profile.

Most work goes through two modules:

    from jobbuddy import pipeline, user_input

    profile = user_input.build_profile(resume_path=..., target_roles=...)
    result  = pipeline.run(scope, config)

`jobbuddy.deepseek` holds the LLM layer -- model resolution, token budgeting,
cost logging. The search pipeline does not use it and runs with no API key.
"""

__version__ = "0.2.0"

__all__ = [
    "job_schema", "job_store", "net", "pipeline", "scoring",
    "skills_taxonomy", "source_mcf", "user_input",
]
