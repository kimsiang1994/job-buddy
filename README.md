# job-buddy

Starting point for the Job Buddy project. Right now it holds a secure setup for the
DeepSeek API key plus a small connectivity test.

## Secrets

The DeepSeek API key lives in a local **`.env`** file that is **git-ignored** and never
committed. `.env.example` is the template that *is* committed:

    copy .env.example .env      # Windows (or `cp` on macOS/Linux)
    # then edit .env and paste your real key

Get a key from the DeepSeek console: https://platform.deepseek.com/

## Test the key

    py test_deepseek.py

Expected output:

- `Authorization successful` — the key is valid (checked via the free `/user/balance`
  endpoint, so no tokens are spent).
- A short reply from the `deepseek-v4-flash` model — end-to-end inference works.

If you see `Authorization FAILED`, the key is invalid or expired: regenerate it in the
DeepSeek console, update `.env`, and re-run.

## Notes

- Models: this project targets `deepseek-v4-flash` / `deepseek-v4-pro`. The older
  `deepseek-chat` / `deepseek-reasoner` names are deprecated as of 2026-07-24.
- The test script uses only the Python standard library — no `pip install` required.
