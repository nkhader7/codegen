"""Streamlit app for secure code recommendations using a self-hosted LLM."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "")

st.set_page_config(page_title="Secure Code Generation", layout="wide")


def ensure_env_variables() -> Tuple[bool, List[str]]:
    """Validate that required environment variables exist."""
    missing = []
    if not LLM_API_URL:
        missing.append("LLM_API_URL")
    if not LLM_API_KEY:
        missing.append("LLM_API_KEY")
    return len(missing) == 0, missing


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


def summarise_files(files: Iterable) -> str:
    """Create a textual summary of uploaded files for LLM prompting."""
    parts: List[str] = []
    for file in files:
        content = read_uploaded_file(file)
        parts.append(
            f"# File: {file.name}\n" +
            "```\n" + content + "\n```"
        )
    return "\n\n".join(parts) if parts else "No files uploaded."


def call_llm(prompt: str) -> str:
    """Send a prompt to the self-hosted LLM and return the response text."""
    if not prompt.strip():
        return "Prompt was empty; nothing to send to the LLM."

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    payload = {"prompt": prompt, "max_tokens": 1024}
    if DEFAULT_MODEL:
        payload["model"] = DEFAULT_MODEL

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


def build_prompt(code_summary: str, scan_summary: str) -> str:
    """Construct a prompt combining source code and scan findings."""
    return (
        "You are an expert secure code assistant. Review the provided code and "
        "security scan findings. Produce actionable remediation advice and, "
        "when appropriate, supply improved code snippets. \n\n"
        "<Code Base>\n"
        f"{code_summary}\n\n"
        "<Scan Findings>\n"
        f"{scan_summary}\n\n"
        "Respond with structured recommendations that reference the affected "
        "files and explain why each change improves security."
    )


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
    with st.spinner("Contacting self-hosted LLM..."):
        code_summary = summarise_files(code_files)
        scan_summary = summarise_files(scan_files)
        prompt = build_prompt(code_summary, scan_summary)
        try:
            response_text = call_llm(prompt)
            st.subheader("LLM Secure Recommendations")
            st.markdown(response_text)
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
            stdout, stderr = run_checkov_scan(code_files)
            st.session_state["checkov_output"] = stdout
            st.session_state["checkov_error"] = stderr

with col2:
    if st.button("Generate remediation from Checkov", disabled=(not env_ok)):
        if not st.session_state["checkov_output"] and not st.session_state["checkov_error"]:
            st.warning("Run Checkov first or upload its results.")
        else:
            checkov_context = st.session_state["checkov_output"] or st.session_state["checkov_error"]
            prompt = build_prompt(summarise_files(code_files), checkov_context)
            with st.spinner("Contacting self-hosted LLM for Checkov remediation..."):
                try:
                    response_text = call_llm(prompt)
                    st.session_state["checkov_recommendations"] = response_text
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
