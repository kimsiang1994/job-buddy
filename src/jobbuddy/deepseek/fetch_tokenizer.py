"""Download DeepSeek's official offline tokenizer into ./tokenizer/.

    py fetch_tokenizer.py

Optional. Without it, token_budget falls back to DeepSeek's documented char-ratio
heuristic and everything still works with zero dependencies.

Caveat worth knowing: the only tokenizer DeepSeek publishes is the **v3** one,
while the models in use are **v4**. Its accuracy for v4 is therefore not
guaranteed -- which is exactly why deepseek_client logs the predicted token count
next to the real one, and `py calibrate_budgets.py --report` prints the measured
error. Keep the dependency only if that number is small.
"""

import io
import os
import shutil
import sys
import urllib.request
import zipfile

URL = "https://cdn.deepseek.com/api-docs/deepseek_v3_tokenizer.zip"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer")
WANTED = ("tokenizer.json", "tokenizer_config.json")


def main():
    print(f"downloading {URL}")
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "job-buddy/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            blob = response.read()
    except Exception as err:                      # noqa: BLE001
        print(f"download failed: {err}")
        return 1
    print(f"  {len(blob):,} bytes")

    try:
        archive = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as err:
        print(f"not a valid zip: {err}")
        return 1

    with archive:
        names = archive.namelist()
        print(f"archive contains {len(names)} entries:")
        for name in names:
            print(f"  {name}")

        # Flatten to basenames on extract -- never join a zip-supplied path, or a
        # crafted archive could write outside the destination directory.
        targets = [n for n in names if os.path.basename(n) in WANTED]
        if not targets:
            print(f"\nNone of {WANTED} found in the archive; nothing extracted.")
            return 1

        os.makedirs(DEST, exist_ok=True)
        for name in targets:
            base = os.path.basename(name)
            with archive.open(name) as src, open(os.path.join(DEST, base), "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"\nextracted {base} -> tokenizer/{base}")

    print("\nNow install the loader:  pip install -r requirements.txt")
    print("Then check it took:      py token_budget.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
