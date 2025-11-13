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
   * `LLM_API_URL` / `LLM_API_KEY` configure how the app talks to your gateway.
   * `LLM_MODEL` is the default model identifier used when no explicit selection is made.
   * (Optional) `LLM_MODELS_URL` points to an endpoint that lists available models/tags.

## Running the Streamlit application

```bash
streamlit run app.py
```

The application provides three workflows:

1. Upload source code (IaC, Java, or any other language) files.
2. Upload security scan reports to combine their findings with the code context.
3. Optionally run Checkov against the uploaded code and ask the LLM for remediation guidance based on the findings.

If you only upload code, the LLM analyses that code to surface security issues and returns hardened replacements. When you also provide scan results (uploaded files or Checkov output), the LLM cross-references those findings while generating its secure code recommendations.

The top of the app now includes an **LLM configuration** panel. When your deployment
exposes a discovery endpoint (for example a `/models` or `/api/tags` route), the app
queries it and renders the advertised identifiers so you can switch between any tags
provided by your self-hosted stack. If discovery fails, you can still enter a custom
model name manually. The LLM response uses the API URL and key defined in your `.env`
file.
