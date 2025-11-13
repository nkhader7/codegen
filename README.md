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

## Running the Streamlit application

```bash
streamlit run app.py
```

The application provides three workflows:

1. Upload source code (IaC, Java, or any other language) files.
2. Upload security scan reports to combine their findings with the code context.
3. Optionally run Checkov against the uploaded code and ask the LLM for remediation guidance based on the findings.

The LLM response uses the API URL and key defined in your `.env` file.
