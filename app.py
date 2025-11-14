"""Streamlit app for secure code recommendations using a self-hosted LLM."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
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


class VirtualUploadedFile:
    """Lightweight representation of a file for re-use in summarisation/RAG."""

    def __init__(self, name: str, content: str):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content.encode("utf-8")


def collect_uploaded_code(files: Iterable) -> Dict[str, str]:
    """Return a mapping of uploaded file names to decoded source."""

    code_map: Dict[str, str] = {}
    for file in files or []:
        code_map[file.name] = read_uploaded_file(file)
    return code_map


def code_map_to_virtual_files(code_map: Dict[str, str]) -> List[VirtualUploadedFile]:
    """Convert a mapping of file contents into VirtualUploadedFile objects."""

    return [VirtualUploadedFile(name, content) for name, content in code_map.items()]


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


def format_code_map_for_prompt(code_map: Dict[str, str]) -> Tuple[str, bool]:
    """Helper to format in-memory code maps for prompting."""

    virtual_files = code_map_to_virtual_files(code_map)
    return summarise_files(virtual_files)


def resolve_chat_endpoint(api_url: str) -> str:
    """Ensure the target endpoint points at /v1/chat/completions."""

    if not api_url:
        return ""

    trimmed = api_url.rstrip("/")
    if trimmed.endswith("/v1/chat/completions"):
        return api_url

    # Treat provided value as a base URL and append the chat completions path.
    base = api_url if api_url.endswith("/") else api_url + "/"
    return urljoin(base, "v1/chat/completions")


def call_llm(messages: Sequence[dict], model: str | None = None) -> str:
    """Send chat messages to the self-hosted LLM and return the response text."""

    if not messages:
        return "No messages were provided for the LLM request."

    if not LLM_API_URL:
        raise ValueError(
            "LLM_API_URL is not configured. Update your .env file before querying the LLM."
        )

    endpoint = resolve_chat_endpoint(LLM_API_URL)
    if not endpoint:
        raise ValueError("Failed to derive the chat completions endpoint from LLM_API_URL.")

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    effective_model = model or DEFAULT_MODEL
    payload: dict = {
        "messages": list(messages),
        "max_tokens": 1024,
    }
    if effective_model:
        payload["model"] = effective_model

    response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict) and isinstance(data.get("choices"), list):
        for choice in data["choices"]:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
            # Some backends still return "text" for compatibility.
            text = choice.get("text")
            if isinstance(text, str):
                return text.strip()

    if isinstance(data, dict):
        message = data.get("message")
        if isinstance(message, str):
            return message.strip()

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
        "Respond with structured recommendations followed by updated code for every "
        "affected file. Use fenced code blocks labelled with the exact file paths, "
        "for example ```path/to/file.ext\n<remediated code>``` so the fixes can be "
        "applied automatically."
    )

    return "\n\n".join(sections)


@dataclass
class DocumentChunk:
    """A single chunk of context collected for retrieval augmented generation."""

    source: str
    content: str


def split_text_into_chunks(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks to feed the retriever."""

    if chunk_size <= 0:
        return [text]

    cleaned = text.replace("\r\n", "\n")
    chunks: List[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = start + chunk_size
        segment = cleaned[start:end]
        chunks.append(segment)
        if end >= length:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def tokenise(text: str) -> List[str]:
    """Simple whitespace/word tokenizer with lower-casing."""

    return [token.lower() for token in re.findall(r"[A-Za-z0-9_]+", text)]


def chunk_uploaded_files(files: Iterable, category: str) -> List[DocumentChunk]:
    """Convert uploaded files into retrievable chunks."""

    chunks: List[DocumentChunk] = []
    for file in files or []:
        text = read_uploaded_file(file)
        for index, fragment in enumerate(split_text_into_chunks(text)):
            label = f"{category}: {file.name} (chunk {index + 1})"
            chunks.append(DocumentChunk(source=label, content=fragment))
    return chunks


def score_chunk(chunk: DocumentChunk, query_tokens: set[str]) -> float:
    """Calculate a lightweight lexical relevance score for a chunk."""

    if not query_tokens:
        return 0.0

    tokens = set(tokenise(chunk.content))
    if not tokens:
        return 0.0

    overlap = len(tokens & query_tokens)
    coverage = overlap / (len(tokens) + 1)
    density = overlap / (len(query_tokens) + 1)
    length_factor = min(len(chunk.content) / 800.0, 1.0)
    return coverage * 0.6 + density * 0.3 + length_factor * 0.1


def build_rag_context(
    code_files: Iterable,
    findings_files: Iterable,
    supplemental_texts: Sequence[str] | None = None,
    top_k: int = 5,
) -> Tuple[str, List[DocumentChunk]]:
    """Select the most relevant chunks from uploaded artefacts for prompting."""

    supplemental_texts = supplemental_texts or []

    chunks = chunk_uploaded_files(code_files, "Code")
    chunks.extend(chunk_uploaded_files(findings_files, "Findings"))

    for index, text in enumerate(supplemental_texts):
        if text and text.strip():
            label = f"Supplemental Context {index + 1}"
            chunks.append(DocumentChunk(source=label, content=text))

    if not chunks:
        return "", []

    query_seed = " ".join(
        part.strip()[:800]
        for part in supplemental_texts
        if isinstance(part, str) and part.strip()
    )
    if not query_seed:
        # Fall back to a generic security-focused query using file names to anchor relevance.
        file_names = [
            Path(getattr(file, "name", "")).name
            for file in (list(code_files or []) + list(findings_files or []))
        ]
        query_seed = " ".join(file_names) + " security remediation vulnerabilities"

    query_tokens = set(tokenise(query_seed))

    scored_chunks = [
        (score_chunk(chunk, query_tokens), chunk)
        for chunk in chunks
    ]
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    top_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]
    if not top_chunks:
        # If lexical scoring did not yield matches, fall back to the first few chunks.
        top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]

    context_sections = []
    for chunk in top_chunks:
        context_sections.append(
            f"# Source: {chunk.source}\n```\n{chunk.content.strip()}\n```"
        )

    return "\n\n".join(context_sections), top_chunks


LANGUAGE_HINTS = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "java": "java",
    "rb": "ruby",
    "go": "go",
    "tf": "hcl",
    "yaml": "yaml",
    "yml": "yaml",
    "json": "json",
    "sh": "bash",
    "ps1": "powershell",
}


def guess_language_from_name(file_name: str) -> str | None:
    """Infer a code language for syntax highlighting based on the file name."""

    suffix = Path(file_name).suffix.lower().lstrip(".")
    if not suffix:
        return None
    return LANGUAGE_HINTS.get(suffix)


CODE_BLOCK_PATTERN = re.compile(r"```(?P<label>[^\n`]*)\n(?P<body>.*?)```", re.DOTALL)
LANGUAGE_ONLY_LABEL = re.compile(r"^[A-Za-z0-9#+\-_.]+$")


def parse_llm_code_blocks(
    markdown_text: str, fallback_names: Sequence[str] | None = None
) -> Dict[str, str]:
    """Extract labelled code blocks from an LLM response."""

    fallback_iter = iter(fallback_names or [])
    extracted: Dict[str, str] = {}
    seen_labels: set[str] = set()

    for index, match in enumerate(CODE_BLOCK_PATTERN.finditer(markdown_text)):
        label = match.group("label").strip()
        body = match.group("body")
        if not body.strip():
            continue

        if label:
            cleaned_label = label.split()[0]
        else:
            cleaned_label = ""

        looks_like_path = "/" in cleaned_label or "." in Path(cleaned_label).name

        if not cleaned_label or (LANGUAGE_ONLY_LABEL.match(cleaned_label) and not looks_like_path):
            cleaned_label = next(fallback_iter, f"llm_output_{index + 1}.txt")

        if cleaned_label in seen_labels:
            cleaned_label = f"{cleaned_label}_v{index + 1}"

        seen_labels.add(cleaned_label)
        extracted[cleaned_label] = body.strip()

    return extracted


def merge_code_maps(base: Dict[str, str], updates: Dict[str, str]) -> Dict[str, str]:
    """Merge code maps, preferring values from updates while preserving originals."""

    merged = dict(base)
    merged.update(updates)
    return merged


def display_code_comparison(
    original: Dict[str, str],
    remediated: Dict[str, str],
    title: str,
) -> None:
    """Render a side-by-side comparison of original and remediated code."""

    st.subheader(title)
    for file_name, original_content in original.items():
        col_left, col_right = st.columns(2, gap="medium")
        with col_left:
            st.caption(f"Original • {file_name}")
            st.code(
                original_content,
                language=guess_language_from_name(file_name) or "text",
            )
        with col_right:
            st.caption(f"Remediated • {file_name}")
            if file_name in remediated:
                st.code(
                    remediated[file_name],
                    language=guess_language_from_name(file_name) or "text",
                )
            else:
                st.info("The LLM did not return an updated version of this file.")

    additional_files = {k: v for k, v in remediated.items() if k not in original}
    if additional_files:
        st.markdown("---")
        st.markdown("#### Additional files proposed by the LLM")
        for file_name, content in additional_files.items():
            st.caption(file_name)
            st.code(content, language=guess_language_from_name(file_name) or "text")


def perform_checkov_remediation_loop(
    code_map: Dict[str, str],
    model: str | None,
    max_iterations: int = 3,
) -> Tuple[Dict[str, str], List[dict], dict]:
    """Run iterative Checkov -> LLM remediation until clean or limit reached."""

    current_map = dict(code_map)
    history: List[dict] = []

    for iteration in range(1, max_iterations + 1):
        checkov_code, stdout, stderr = run_checkov_scan(
            code_map_to_virtual_files(current_map)
        )
        iteration_log = {
            "iteration": iteration,
            "checkov_return_code": checkov_code,
            "checkov_stdout": stdout,
            "checkov_stderr": stderr,
        }
        history.append(iteration_log)

        if checkov_code == 0:
            iteration_log["status"] = "clean"
            break

        if checkov_code not in (0, 1):
            iteration_log["status"] = "checkov_error"
            break

        checkov_context = "\n\n".join(
            section for section in (stdout, stderr) if section and section.strip()
        )
        code_summary, _ = format_code_map_for_prompt(current_map)
        prompt = build_prompt(
            code_summary,
            has_code=True,
            findings_summary=checkov_context,
            has_findings=bool(checkov_context.strip()),
            findings_label="Checkov Scan Findings",
        )
        rag_context, _ = build_rag_context(
            code_map_to_virtual_files(current_map),
            [],
            supplemental_texts=[code_summary, checkov_context],
        )
        iteration_log["rag_context"] = rag_context
        messages = prepare_chat_messages(prompt, rag_context)
        response_text = call_llm(messages, model=model)
        iteration_log["llm_response"] = response_text

        parsed_updates = parse_llm_code_blocks(
            response_text, fallback_names=list(current_map.keys())
        )
        iteration_log["remediated_files"] = list(parsed_updates.keys())

        if not parsed_updates:
            iteration_log["status"] = "llm_missing_code"
            break

        current_map = merge_code_maps(current_map, parsed_updates)
        iteration_log["status"] = "llm_applied"

    final_code, final_stdout, final_stderr = run_checkov_scan(
        code_map_to_virtual_files(current_map)
    )
    final_validation = {
        "checkov_return_code": final_code,
        "checkov_stdout": final_stdout,
        "checkov_stderr": final_stderr,
    }

    return current_map, history, final_validation


def prepare_chat_messages(prompt: str, rag_context: str) -> List[dict]:
    """Assemble chat messages for the chat completions API."""

    base_system_prompt = (
        "You are a secure code remediation assistant. Analyse infrastructure and "
        "application artefacts, identify vulnerabilities, and return actionable "
        "fixes with secure replacement code and explanations."
    )

    user_parts: List[str] = []
    if rag_context.strip():
        user_parts.append("Context documents:\n" + rag_context)
    user_parts.append("Task:\n" + prompt)

    return [
        {"role": "system", "content": base_system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def write_files_to_temp(files: Iterable) -> Tuple[tempfile.TemporaryDirectory, Path]:
    """Persist uploaded or virtual files to a temporary directory for scanning."""

    temp_dir = tempfile.TemporaryDirectory()
    base_path = Path(temp_dir.name)
    for file in files or []:
        file_path = base_path / Path(file.name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(file.getvalue())
    return temp_dir, base_path


def run_checkov_scan(code_files: Iterable) -> Tuple[int, str, str]:
    """Execute Checkov and return (exit_code, stdout, stderr)."""

    if not code_files:
        return 2, "", "No code files provided for Checkov scan."

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
            stderr = f"Checkov exited with status {result.returncode}.\n{stderr}".strip()
        return result.returncode, stdout or "Checkov completed with no output.", stderr
    except FileNotFoundError:
        return 127, "", "Checkov CLI was not found. Please install it inside the environment."
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
if "secure_generation" not in st.session_state:
    st.session_state["secure_generation"] = None
if "checkov_recommendations" not in st.session_state:
    st.session_state["checkov_recommendations"] = None
if "checkov_loop" not in st.session_state:
    st.session_state["checkov_loop"] = None


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
st.write(
    "Provide at least one source or manifest file. These uploads are required for "
    "secure code regeneration, scan-assisted fixes, and Checkov remediation."
)
code_files = st.file_uploader(
    "Upload IaC, Java, or other source files",
    accept_multiple_files=True,
    type=None,
    help="Select one or more files. Secure generation buttons remain disabled until code is uploaded.",
    key="code_files",
)

uploaded_code_map = collect_uploaded_code(code_files or [])
code_summary, has_code = format_code_map_for_prompt(uploaded_code_map)

st.header("2. Upload security scan results")
scan_files = st.file_uploader(
    "Upload reports from SAST/DAST tools",
    accept_multiple_files=True,
    type=None,
    help="Optional – when supplied the findings guide the LLM fixes.",
    key="scan_files",
)

scan_summary, has_scans = summarise_files(scan_files or [])

if st.button(
    "Generate secure code recommendations",
    disabled=(not env_ok or not code_files),
):
    if not has_code:
        st.error("Upload at least one code file before generating secure code.")
    else:
        prompt = build_prompt(
            code_summary,
            has_code,
            scan_summary,
            has_scans,
            findings_label="Scan Findings",
        )
        rag_context, _rag_chunks = build_rag_context(
            code_files or [],
            scan_files or [],
            supplemental_texts=[code_summary, scan_summary],
        )
        messages = prepare_chat_messages(prompt, rag_context)
        with st.spinner("Contacting self-hosted LLM..."):
            try:
                selected_model = st.session_state.get("active_model") or None
                response_text = call_llm(messages, model=selected_model)
                remediated_map = parse_llm_code_blocks(
                    response_text, fallback_names=list(uploaded_code_map.keys())
                )
                st.session_state["secure_generation"] = {
                    "original_code": uploaded_code_map,
                    "remediated_code": remediated_map,
                    "raw_response": response_text,
                    "rag_context": rag_context,
                    "prompt": prompt,
                    "has_scan": has_scans,
                    "scan_summary": scan_summary if has_scans else "",
                }
                if not remediated_map:
                    st.warning(
                        "The LLM response did not include labelled code blocks. Review the "
                        "response below to apply fixes manually."
                    )
            except ValueError as exc:
                st.error(str(exc))
            except requests.RequestException as exc:
                st.error(f"Failed to call the LLM API: {exc}")

secure_result = st.session_state.get("secure_generation")
if secure_result:
    display_code_comparison(
        secure_result.get("original_code", {}),
        secure_result.get("remediated_code", {}),
        "Secure code regeneration",
    )
    with st.expander("Full LLM response"):
        st.markdown(secure_result.get("raw_response", ""))
    if secure_result.get("rag_context"):
        with st.expander("Context supplied to the LLM (RAG)"):
            st.markdown(secure_result["rag_context"])
    if secure_result.get("has_scan") and secure_result.get("scan_summary"):
        with st.expander("Uploaded scan findings summary"):
            st.markdown(secure_result["scan_summary"])

st.header("3. Run Checkov scan and remediate")
st.write(
    "Execute Checkov on the uploaded code, then feed the results back to the LLM. "
    "You can request a single pass or let the app iterate until the scan is clean "
    "or the retry budget is exhausted."
)

if "checkov_run" not in st.session_state:
    st.session_state["checkov_run"] = {"returncode": None, "stdout": "", "stderr": ""}

run_data = st.session_state.get("checkov_run") or {
    "returncode": None,
    "stdout": "",
    "stderr": "",
}

actions = st.columns(3)
with actions[0]:
    if st.button("Run Checkov", disabled=not code_files):
        with st.spinner("Executing Checkov scan..."):
            return_code, stdout, stderr = run_checkov_scan(code_files or [])
            st.session_state["checkov_run"] = {
                "returncode": return_code,
                "stdout": stdout,
                "stderr": stderr,
            }
            st.session_state["checkov_recommendations"] = None
            st.session_state["checkov_loop"] = None
            run_data = st.session_state["checkov_run"]
        if return_code == 127:
            st.error(stderr)
        elif return_code not in (0, 1):
            st.warning(stderr or "Checkov exited with a non-zero status.")

with actions[1]:
    if st.button(
        "Generate remediation from Checkov",
        disabled=(not env_ok or not code_files),
    ):
        checkov_context = "\n\n".join(
            section
            for section in (run_data.get("stdout"), run_data.get("stderr"))
            if section and section.strip()
        )
        has_checkov_results = bool(checkov_context.strip())

        if not has_checkov_results:
            st.warning("Run Checkov first to capture findings for the LLM.")
        else:
            prompt = build_prompt(
                code_summary,
                has_code,
                checkov_context,
                has_checkov_results,
                findings_label="Checkov Scan Findings",
            )
            rag_context, _rag_chunks = build_rag_context(
                code_files or [],
                [],
                supplemental_texts=[code_summary, checkov_context],
            )
            messages = prepare_chat_messages(prompt, rag_context)
            with st.spinner("Contacting self-hosted LLM for Checkov remediation..."):
                try:
                    selected_model = st.session_state.get("active_model") or None
                    response_text = call_llm(messages, model=selected_model)
                    remediated_map = parse_llm_code_blocks(
                        response_text, fallback_names=list(uploaded_code_map.keys())
                    )
                    st.session_state["checkov_recommendations"] = {
                        "original_code": uploaded_code_map,
                        "remediated_code": remediated_map,
                        "raw_response": response_text,
                        "rag_context": rag_context,
                        "checkov_context": checkov_context,
                    }
                    st.session_state["checkov_loop"] = None
                    if not remediated_map:
                        st.warning(
                            "The LLM response did not include labelled code blocks. "
                            "Review the response below to apply fixes manually."
                        )
                except ValueError as exc:
                    st.error(str(exc))
                except requests.RequestException as exc:
                    st.error(f"Failed to call the LLM API: {exc}")

with actions[2]:
    if st.button(
        "Iterative Checkov remediation",
        disabled=(not env_ok or not code_files),
    ):
        if not has_code:
            st.error("Upload at least one code file before running Checkov remediation.")
        else:
            with st.spinner("Iterating between Checkov and the LLM..."):
                try:
                    selected_model = st.session_state.get("active_model") or None
                    final_code_map, history, final_validation = (
                        perform_checkov_remediation_loop(
                            uploaded_code_map, selected_model, max_iterations=3
                        )
                    )
                    st.session_state["checkov_loop"] = {
                        "original_code": uploaded_code_map,
                        "final_code": final_code_map,
                        "history": history,
                        "final_validation": final_validation,
                    }
                    st.session_state["checkov_recommendations"] = None
                except ValueError as exc:
                    st.error(str(exc))
                except requests.RequestException as exc:
                    st.error(f"Failed to call the LLM API: {exc}")

run_data = st.session_state.get("checkov_run") or {}
if run_data.get("stdout"):
    st.subheader("Checkov Output")
    st.code(run_data.get("stdout"), language="bash")

if run_data.get("stderr"):
    st.subheader("Checkov Messages")
    st.code(run_data.get("stderr"), language="bash")

if run_data.get("returncode") is not None:
    st.caption(f"Last Checkov exit code: {run_data.get('returncode')}")

checkov_result = st.session_state.get("checkov_recommendations")
if checkov_result:
    display_code_comparison(
        checkov_result.get("original_code", {}),
        checkov_result.get("remediated_code", {}),
        "Checkov-guided remediation",
    )
    with st.expander("Full LLM response for Checkov findings"):
        st.markdown(checkov_result.get("raw_response", ""))
    if checkov_result.get("rag_context"):
        with st.expander("Context supplied to the LLM (RAG)"):
            st.markdown(checkov_result["rag_context"])
    if checkov_result.get("checkov_context"):
        with st.expander("Checkov findings shared with the LLM"):
            st.code(checkov_result["checkov_context"], language="text")

loop_result = st.session_state.get("checkov_loop")
if loop_result:
    display_code_comparison(
        loop_result.get("original_code", {}),
        loop_result.get("final_code", {}),
        "Iterative Checkov remediation",
    )
    final_validation = loop_result.get("final_validation", {})
    st.caption(
        "Final Checkov exit code: "
        + str(final_validation.get("checkov_return_code", "unknown"))
    )
    with st.expander("Remediation iterations"):
        for entry in loop_result.get("history", []):
            st.markdown(f"**Iteration {entry.get('iteration')}**")
            st.markdown(
                f"- Checkov exit: {entry.get('checkov_return_code')}\n"
                f"- Status: {entry.get('status')}\n"
            )
            if entry.get("checkov_stdout"):
                with st.expander("Checkov stdout"):
                    st.code(entry["checkov_stdout"], language="bash")
            if entry.get("checkov_stderr"):
                with st.expander("Checkov stderr"):
                    st.code(entry["checkov_stderr"], language="bash")
            if entry.get("llm_response"):
                with st.expander("LLM response"):
                    st.markdown(entry["llm_response"])
    if final_validation.get("checkov_stdout") or final_validation.get("checkov_stderr"):
        with st.expander("Final Checkov validation output"):
            if final_validation.get("checkov_stdout"):
                st.code(final_validation["checkov_stdout"], language="bash")
            if final_validation.get("checkov_stderr"):
                st.code(final_validation["checkov_stderr"], language="bash")

st.caption(
    "Remember to configure your self-hosted LLM endpoint and API key inside the .env file."
)
