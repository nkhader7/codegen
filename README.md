# codegen

Generate Secure Code

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Install [Checkov](https://www.checkov.io/) if you plan to run local scans:
   ```bash
   pip install checkov
   ```
3. Copy the `.env.example` file to `.env` and update the values so the app can reach your self-hosted LLM endpoint:
   ```bash
   cp .env.example .env
   ```
   * `LLM_API_URL` / `LLM_API_KEY` configure how the app talks to your gateway. Supply the base URL for your deployment; the app automatically targets `POST /v1/chat/completions` on that host.
   * `LLM_MODEL` is the default model identifier used when no explicit selection is made.
   * (Optional) `LLM_MODELS_URL` points to an endpoint that lists available models/tags.

## Running the Streamlit application

```bash
streamlit run app.py
```

The application provides three workflows:

The application provides three workflows. All of them require at least one uploaded code file so the LLM and Checkov have concrete artefacts to analyse.

### 1. Secure code regeneration (code only)

* Upload one or more application or infrastructure files.
* Press **Generate secure code recommendations**. The LLM reviews the code, identifies weaknesses, and responds with hardened replacements.
* The UI renders a side-by-side comparison so you can inspect the original file on the left and the LLM’s remediated version on the right.

### 2. Scan-assisted remediation (code + findings)

* Upload the same code files plus any static/dynamic scan reports.
* The RAG pipeline indexes both data sources and highlights the most relevant snippets when querying the LLM.
* The resulting response still appears in the left/right comparison view, but the guidance and regenerated code incorporate the scan results to prioritise and justify fixes.

### 3. Iterative Checkov remediation loop

* Upload your code and press **Run Checkov** to execute a scan inside the temporary workspace.
* Either request a single Checkov-guided remediation (one pass through the LLM) or trigger **Iterative Checkov remediation**. The latter follows the flow from the provided diagram: run Checkov, retrieve findings, build a RAG-informed prompt, query the LLM for fixed code, merge the output, and re-run Checkov. The loop continues until Checkov reports a clean result or the retry budget (default: three passes) is exhausted.
* Each iteration logs the Checkov output and full LLM response. The final tab displays the remediated files next to the originals plus the validation scan output.

Behind the scenes the app performs lightweight retrieval-augmented generation (RAG). Uploaded artefacts are chunked, scored for relevance, and only the most pertinent snippets are forwarded to the chat completions API. This keeps prompts focused while still grounding the response in your actual code and findings.

The top of the app now includes an **LLM configuration** panel. When your deployment exposes a discovery endpoint (for example a `/models` or `/api/tags` route), the app queries it and renders the advertised identifiers so you can switch between any tags provided by your self-hosted stack. If discovery fails, you can still enter a custom model name manually. The LLM response uses the API URL and key defined in your `.env` file.

Every LLM interaction targets the OpenAI-compatible `v1/chat/completions` route, so make sure your self-hosted stack exposes that contract. The request combines a system message that defines the secure-coding role and a user message that embeds the RAG context alongside the remediation task description.
