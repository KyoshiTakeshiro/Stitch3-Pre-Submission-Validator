# Stitch3 Validator

Stitch3 Validator helps Bitcast creators check whether a draft X post is likely to satisfy a campaign brief before publishing. Instead of posting first and discovering afterward that a tweet failed validation, users can paste their draft, select a campaign, and receive an instant pass/fail evaluation using the same evaluation logic as the Bitcast/Stitch3 validator.

**Why this project is useful**

Creators often lose engagement when a tweet fails brief validation. By the time they discover the issue, the original post has already gained traction, and a replacement tweet typically performs worse.

This tool reduces that risk by providing immediate feedback before a tweet is published, allowing creators to improve compliance while preserving the opportunity for maximum engagement.

**How this matches the real Bitcast validator**

This tool is built to replicate the actual validator's brief-evaluation logic as closely as possible, not just approximate it:

- **Same model and provider.** Evaluation runs on `Qwen/Qwen3-32B` via [Chutes](https://chutes.ai) — the same model and provider the production validator uses, not a different LLM standing in for it.
- **Same prompts.** All 4 prompt versions are transcribed directly from the validator's source, and the correct version is selected automatically per brief (via each brief's `prompt_version` field), the same way the real validator does.
- **Same optimistic multi-check strategy.** Just like the real validator, this tool runs up to 3 independent evaluation checks per tweet and accepts it as compliant if *any* check passes — including appending the same per-check text differentiator the validator uses internally so each of the 3 checks is a genuinely independent judgment rather than 3 repeats of the same answer.

Because of that last point, results can vary slightly between runs on borderline tweets — that's expected, and mirrors how the live validator itself behaves, not a bug in this tool.

**Getting Started**

1. **Clone this repository.**

   ```bash
   git clone https://github.com/KyoshiTakeshiro/Stitch3-Validator.git
   cd Stitch3-Validator
   ```

2. **Install the project dependencies.**

   Create a virtual environment and install the Python packages from `requirements.txt`:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure your Chutes API key.**

   Evaluation runs on [Chutes](https://chutes.ai) (`Qwen/Qwen3-32B`) — the same model and provider the real Bitcast validator uses. Copy the example env file and fill in your key:

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and set `CHUTES_API_KEY` to a real key from your [Chutes account](https://chutes.ai).

4. **Start the FastAPI backend.**

   ```bash
   uvicorn main:app --reload
   ```

   This serves both the API and the frontend on `http://localhost:8000`.

5. **Open the web interface in your browser.**

   Visit [http://localhost:8000](http://localhost:8000). You should see the Stitch3 Validator UI.

6. **Select a campaign brief, paste your draft tweet, and run the evaluation.**

   - Pick an ecosystem (Bittensor / Perp DEXs / Prediction Markets), then choose a campaign from the **Campaign brief** dropdown.
   - Paste your draft tweet into the **Draft tweet** field.
   - Click **Check against brief**. The tool runs up to 3 checks (mirroring the real validator's optimistic multi-check strategy) and returns a pass/fail verdict, with a reason shown if it fails.

**Maintainers & Contributions**

This project is maintained by @KyoshiTakeshiro

Contributions are welcome. If you find a bug, have ideas for improvements, or want to add new functionality, feel free to open an issue or submit a pull request.
