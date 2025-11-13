"""Streamlit app for secure code recommendations using a self-hosted LLM."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import urljoin, urlparse

import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "")
MODELS_ENDPOINT = os.getenv("LLM_MODELS_URL", "")

st.set_page_config(page_title="Secure Code Generation", layout="wide")


def ensure_env_variables() -> Tuple[bool, List[str]]:
    """Validate that required environment variables exist."""
    missing = []
    if not LLM_API_URL:
        missing.append("LLM_API_URL")
    if not LLM_API_KEY:
        missing.append("LLM_API_KEY")
    return len(missing) == 0, missing


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _extract_models_from_payload(payload) -> List[str]:
    """Best-effort extraction of model identifiers from various API payloads."""
    names: List[str] = []

    def collect_from_candidate(candidate) -> None:
        if isinstance(candidate, str):
            names.append(candidate)
        elif isinstance(candidate, dict):
            for key in ("name", "id", "model", "slug", "tag"):
                value = candidate.get(key)
                if isinstance(value, str):
                    names.append(value)

    if isinstance(payload, dict):
        for key in ("models", "data", "result", "items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                for item in candidate:
                    collect_from_candidate(item)
            elif isinstance(candidate, dict):
                collect_from_candidate(candidate)
        # Some APIs return {"model": "name"}
        collect_from_candidate(payload)
    elif isinstance(payload, list):
        for item in payload:
            collect_from_candidate(item)

    return _unique_preserve_order(names)


def derive_model_endpoints(api_url: str, override_url: str = "") -> List[str]:
    """Guess potential endpoints that return a list of models."""
    candidates: List[str] = []
    if override_url:
        candidates.append(override_url)

    if not api_url:
        return _unique_preserve_order(candidates)

    parsed = urlparse(api_url)
    if not parsed.scheme or not parsed.netloc:
        return _unique_preserve_order(candidates)

    base = f"{parsed.scheme}://{parsed.netloc}"
    candidates.extend(
        [
            urljoin(base, "/models"),
            urljoin(base, "/v1/models"),
            urljoin(base, "/api/models"),
            urljoin(base, "/api/tags"),
        ]
    )

    if parsed.path:
        path_parts = parsed.path.rstrip("/").split("/")
        if len(path_parts) > 1:
            parent_path = "/".join(path_parts[:-1])
            if not parent_path.startswith("/"):
                parent_path = "/" + parent_path
            candidates.extend(
                [
                    urljoin(base, parent_path + "/models"),
                    urljoin(base, parent_path + "/tags"),
                ]
            )

    return _unique_preserve_order(candidates)


def discover_llm_models(api_url: str, override_url: str = "") -> Tuple[List[str], List[str]]:
    """Attempt to fetch the list of available models from the LLM API."""

    headers = {"Accept": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    discovered: List[str] = []
    logs: List[str] = []

    for endpoint in derive_model_endpoints(api_url, override_url):
        try:
            response = requests.get(endpoint, headers=headers, timeout=15)
        except requests.RequestException as exc:
            logs.append(f"Failed to reach {endpoint}: {exc}")
            continue

        if response.status_code != 200:
            logs.append(
                f"{endpoint} returned status {response.status_code}; skipping."
            )
            continue

        try:
            payload = response.json()
        except ValueError as exc:  # JSON decoding error
            logs.append(f"{endpoint} did not return JSON: {exc}")
            continue

        models = _extract_models_from_payload(payload)
        if models:
            discovered.extend(models)
            logs.append(
                f"Discovered {len(models)} model(s) from {endpoint}."
            )
        else:
            logs.append(f"{endpoint} responded but no model identifiers were found.")

    return _unique_preserve_order(discovered), logs


def read_uploaded_file(file) -> str:
    """Decode an uploaded file into text."""
    raw_bytes = file.getvalue()
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    # Fallback to repr of bytes if decoding fails
    return repr(raw_bytes)


def summarise_files(files: Iterable) -> Tuple[str, bool]:
    """Create a textual summary of uploaded files for LLM prompting."""
    parts: List[str] = []
    for file in files:
        content = read_uploaded_file(file)
        parts.append(
            f"# File: {file.name}\n"
            + "```\n"
            + content
            + "\n```"
        )
    if not parts:
        return "", False
    return "\n\n".join(parts), True


def call_llm(prompt: str, model: str | None = None) -> str:
    """Send a prompt to the self-hosted LLM and return the response text."""
    if not prompt.strip():
        return "Prompt was empty; nothing to send to the LLM."

    if not LLM_API_URL:
        raise ValueError(
            "LLM_API_URL is not configured. Update your .env file before querying the LLM."
        )

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {"prompt": prompt, "max_tokens": 1024}
    effective_model = model or DEFAULT_MODEL
    if effective_model:
        payload["model"] = effective_model

    response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict):
        if "completion" in data:
            return data["completion"].strip()
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                text = choice.get("text") or choice.get("message", {}).get("content")
                if text:
                    return str(text).strip()
        if "message" in data:
            return str(data["message"]).strip()
    return json.dumps(data, indent=2)


def build_prompt(
    code_summary: str,
    has_code: bool,
    findings_summary: str,
    has_findings: bool,
    findings_label: str = "Scan Findings",
) -> str:
    """Construct a prompt combining source code and (optional) scan findings."""

    guidance: List[str] = [
        "You are an expert secure code assistant tasked with producing secure "
        "remediation guidance and replacement code.",
    ]

    if has_code:
        guidance.append(
            "Identify vulnerabilities, misconfigurations, and insecure patterns in the "
            "provided source code. Explain the risks and generate remediated code snippets."
        )
    else:
        guidance.append(
            "No source code artifacts were supplied; use the available findings to craft "
            "secure guidance."
        )

    if has_findings:
        guidance.append(
            f"Incorporate the {findings_label.lower()} to prioritise fixes and validate the "
            "recommended changes."
        )
    else:
        guidance.append(
            f"No {findings_label.lower()} are provided; rely entirely on the code review to "
            "infer vulnerabilities and suggest secure implementations."
        )

    sections = ["\n".join(guidance)]

    if has_code:
        sections.append("<Code Base>\n" + code_summary)
    else:
        sections.append("<Code Base>\nNo code artifacts were uploaded.")

    if has_findings:
        sections.append(f"<{findings_label}>\n" + findings_summary)
    else:
        sections.append(f"<{findings_label}>\nNo additional findings were provided.")

    sections.append(
        "Respond with structured recommendations, referencing affected files and "
        "including improved code blocks that resolve the identified security issues."
    )

    return "\n\n".join(sections)


def write_files_to_temp(files: Iterable) -> Tuple[tempfile.TemporaryDirectory, Path]:
    """Persist uploaded files to a temporary directory for scanning."""
    temp_dir = tempfile.TemporaryDirectory()
    base_path = Path(temp_dir.name)
    for file in files:
        file_path = base_path / Path(file.name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(file.getvalue())
    return temp_dir, base_path


def run_checkov_scan(code_files: Iterable) -> Tuple[str, str]:
    """Execute Checkov on uploaded code files and return stdout/stderr."""
    if not code_files:
        return "", "No code files provided for Checkov scan."

    temp_dir, base_path = write_files_to_temp(code_files)
    try:
        result = subprocess.run(
            ["checkov", "-d", str(base_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode not in (0, 1):
            # Checkov returns 1 when misconfigurations are found; treat as success
            stderr = f"Checkov exited with status {result.returncode}.\n{stderr}"
        return stdout or "Checkov completed with no output.", stderr
    except FileNotFoundError:
        return "", "Checkov CLI was not found. Please install it inside the environment."
    finally:
        temp_dir.cleanup()


# Layout
st.title("Secure Code Generation Workbench")
st.markdown(
    "This application aggregates infrastructure or application code, combines it with "
    "scan findings, and queries your self-hosted LLM to provide secure remediation "
    "guidance."
)

env_ok, missing_env = ensure_env_variables()
if not env_ok:
    st.warning(
        "Missing required environment variables: " + ", ".join(missing_env) +
        ". Update your .env file before attempting to query the LLM."
    )

st.header("LLM configuration")
st.write(
    "Select the model hosted by your API. The app can attempt to read the API "
    "metadata to list available identifiers advertised by your deployment."
)

if "model_options" not in st.session_state:
    st.session_state["model_options"] = []
if "model_discovery_logs" not in st.session_state:
    st.session_state["model_discovery_logs"] = []
if "model_discovery_attempted" not in st.session_state:
    st.session_state["model_discovery_attempted"] = False
if "active_model" not in st.session_state:
    st.session_state["active_model"] = DEFAULT_MODEL.strip()


def refresh_model_options() -> None:
    models, logs = discover_llm_models(LLM_API_URL or "", MODELS_ENDPOINT or "")
    st.session_state["model_options"] = models
    st.session_state["model_discovery_logs"] = logs
    st.session_state["model_discovery_attempted"] = True
    if models:
        if st.session_state.get("active_model") not in models:
            st.session_state["active_model"] = models[0]


if env_ok and not st.session_state["model_discovery_attempted"]:
    with st.spinner("Discovering available models..."):
        refresh_model_options()

st.button(
    "Refresh available models",
    on_click=refresh_model_options,
    disabled=not env_ok,
)

model_options = st.session_state.get("model_options", [])
active_model_default = st.session_state.get("active_model") or DEFAULT_MODEL

if model_options:
    option_labels = model_options + ["Custom model identifier…"]
    default_label = (
        active_model_default
        if active_model_default in model_options
        else "Custom model identifier…"
    )
    selected_label = st.selectbox(
        "Available models detected from the API",
        option_labels,
        index=option_labels.index(default_label) if default_label in option_labels else 0,
        help="Pick an advertised model (refresh if you recently pulled a new image).",
    )
    if selected_label == "Custom model identifier…":
        custom_value = st.text_input(
            "Model identifier",
            value=active_model_default or "",
            key="custom_model_input",
            help=(
                "Type the identifier expected by your deployment (for example: "
                "secure-code-model or any published tag)."
            ),
        ).strip()
        st.session_state["active_model"] = custom_value
    else:
        st.session_state["active_model"] = selected_label
else:
    custom_value = st.text_input(
        "Model identifier",
        value=active_model_default or DEFAULT_MODEL,
        key="custom_model_input",
        help=(
            "Provide the model name or tag offered by your backend (for example: "
            "default, secure-assistant, or any published label)."
        ),
    ).strip()
    st.session_state["active_model"] = custom_value

if st.session_state.get("active_model"):
    st.caption(f"Requests will include model '{st.session_state['active_model']}'.")
else:
    st.caption("Requests will omit the model parameter; the backend default will be used.")

discovery_logs = st.session_state.get("model_discovery_logs") or []
if discovery_logs:
    with st.expander("Model discovery details"):
        for entry in discovery_logs:
            st.write(f"- {entry}")

st.header("1. Upload source code or manifests")
code_files = st.file_uploader(
    "Upload IaC, Java, or other source files",
    accept_multiple_files=True,
    type=None,
    help="You can select multiple files to provide context for the LLM and Checkov.",
    key="code_files",
)

st.header("2. Upload security scan results")
scan_files = st.file_uploader(
    "Upload reports from SAST/DAST tools",
    accept_multiple_files=True,
    type=None,
    help="These findings will be summarised and shared with the LLM.",
    key="scan_files",
)

if st.button("Generate secure code recommendations", disabled=not env_ok):
    code_summary, has_code = summarise_files(code_files or [])
    scan_summary, has_scans = summarise_files(scan_files or [])

    if not has_code and not has_scans:
        st.warning("Upload source code and/or scan findings before contacting the LLM.")
    else:
        prompt = build_prompt(
            code_summary,
            has_code,
            scan_summary,
            has_scans,
            findings_label="Scan Findings",
        )
        with st.spinner("Contacting self-hosted LLM..."):
            try:
                selected_model = st.session_state.get("active_model") or None
                response_text = call_llm(prompt, model=selected_model)
                st.subheader("LLM Secure Recommendations")
                st.markdown(response_text)
            except ValueError as exc:
                st.error(str(exc))
            except requests.RequestException as exc:
                st.error(f"Failed to call the LLM API: {exc}")

st.header("3. Run Checkov scan and remediate")
st.write(
    "Optionally execute Checkov against the uploaded code. The results will be "
    "sent to the LLM for remediation advice."
)

if "checkov_output" not in st.session_state:
    st.session_state["checkov_output"] = ""
    st.session_state["checkov_error"] = ""
    st.session_state["checkov_recommendations"] = ""

col1, col2 = st.columns(2)
with col1:
    if st.button("Run Checkov", disabled=not code_files):
        with st.spinner("Executing Checkov scan..."):
            stdout, stderr = run_checkov_scan(code_files or [])
            st.session_state["checkov_output"] = stdout
            st.session_state["checkov_error"] = stderr

with col2:
    if st.button("Generate remediation from Checkov", disabled=(not env_ok)):
        checkov_sections = [
            st.session_state.get("checkov_output", ""),
            st.session_state.get("checkov_error", ""),
        ]
        checkov_context = "\n\n".join(section for section in checkov_sections if section.strip())
        has_checkov_results = bool(checkov_context.strip())

        if not has_checkov_results:
            st.warning("Run Checkov first or upload its results.")
        else:
            code_summary, has_code = summarise_files(code_files or [])
            prompt = build_prompt(
                code_summary,
                has_code,
                checkov_context,
                has_checkov_results,
                findings_label="Checkov Scan Findings",
            )
            with st.spinner("Contacting self-hosted LLM for Checkov remediation..."):
                try:
                    selected_model = st.session_state.get("active_model") or None
                    response_text = call_llm(prompt, model=selected_model)
                    st.session_state["checkov_recommendations"] = response_text
                except ValueError as exc:
                    st.error(str(exc))
                except requests.RequestException as exc:
                    st.error(f"Failed to call the LLM API: {exc}")

if st.session_state["checkov_output"]:
    st.subheader("Checkov Output")
    st.code(st.session_state["checkov_output"], language="bash")

if st.session_state["checkov_error"]:
    st.subheader("Checkov Messages")
    st.code(st.session_state["checkov_error"], language="bash")

if st.session_state["checkov_recommendations"]:
    st.subheader("LLM Remediation for Checkov Findings")
    st.markdown(st.session_state["checkov_recommendations"])

st.caption(
    "Remember to configure your self-hosted LLM endpoint and API key inside the .env file."
)
