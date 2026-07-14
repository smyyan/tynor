"""
Tynor SOP Agent - PRODUCTION v6 (persistent vector store + auto department routing)
SharePoint (Graph) + DeepSeek/DeepInfra + Qdrant Cloud.

How it scales cheaply & accurately:
  - Persistent vector store in Qdrant Cloud; only NEW/CHANGED files are re-embedded.
  - Each chunk is tagged with its DEPARTMENT = the subfolder it came from.
  - On each question, the app AUTO-ROUTES to the most relevant department(s) using the same
    embeddings (no extra AI call, free) and searches mainly within those folders. This keeps the
    search scope small -> low TOP_K works -> cheap AND accurate even with hundreds of SOPs.
  - "All Departments" and "Interdepartmental" folders are always included (cross-cutting content).
  - If routing confidence is weak, it falls back to searching everything.

PROVIDER: works with DeepInfra (US-hosted) OR DeepSeek-direct (China) - just set the two env
vars below. No code change needed to switch hosts.
  DeepInfra:      DEEPSEEK_BASE_URL=https://api.deepinfra.com/v1/openai
                  models like  deepseek-ai/DeepSeek-V4-Flash
  DeepSeek-direct DEEPSEEK_BASE_URL=https://api.deepseek.com
                  models like  deepseek-v4-flash

INSTALL:
  pip install streamlit openai msal requests python-docx pypdf sentence-transformers qdrant-client numpy
RUN:
  streamlit run tynor_sop_assistant.py

SECURITY: set ALL secrets as environment variables. This file has NO hardcoded keys on purpose.
"""

import os
import io
import json
import hashlib
import datetime
import numpy as np
import streamlit as st

# ============================================================
# >>> CONFIG  (all secrets come from environment variables) <<<
# ============================================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepinfra.com/v1/openai")

# Qdrant Cloud (vector store) - create a free cluster at cloud.qdrant.io
QDRANT_URL     = os.environ.get("QDRANT_URL", "https://3847198c-6464-4edb-aa66-760fab05f073.us-east-1-1.aws.cloud.qdrant.io")       # e.g. https://xxxx.cloud.qdrant.io:6333
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

GRAPH_TENANT_ID     = os.environ.get("GRAPH_TENANT_ID", "")     # from IT (app registration)
GRAPH_CLIENT_ID     = os.environ.get("GRAPH_CLIENT_ID", "")     # from IT
GRAPH_CLIENT_SECRET = os.environ.get("GRAPH_CLIENT_SECRET", "") # from IT

SP_HOSTNAME     = os.environ.get("SP_HOSTNAME", "tynorind.sharepoint.com")
SP_SITE_PATH    = os.environ.get("SP_SITE_PATH", "tynorkpi")
SP_LIBRARY_NAME = os.environ.get("SP_LIBRARY_NAME", "ISO")  # "" = default Documents lib
SP_FOLDER_PATH  = os.environ.get("SP_FOLDER_PATH", "SOP, Policies, WI")        # main SOPs folder (holds dept subfolders)

# Model strings differ by provider - set via env so you can switch hosts without editing code.
#  DeepInfra:      "deepseek-ai/DeepSeek-V4-Pro" / "deepseek-ai/DeepSeek-V4-Flash"
#                  (CONFIRM exact strings on the model pages at deepinfra.com - copy verbatim)
#  DeepSeek-direct "deepseek-v4-pro" / "deepseek-v4-flash"
MODEL_PRECISE = os.environ.get("MODEL_PRECISE", "deepseek-ai/DeepSeek-V4-Pro")
MODEL_QUICK   = os.environ.get("MODEL_QUICK",   "deepseek-ai/DeepSeek-V4-Flash")
# ============================================================

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6                       # per-question chunks (scope is narrowed by routing, so this stays low)
ROUTE_DEPTS = 2                 # how many top departments to route into (besides the always-include ones)
CHUNK_MAX_CHARS = 2500

# Folders always searched regardless of routing (cross-cutting content)
ALWAYS_INCLUDE = {"All Departments", "Interdepartmental"}

MAX_TOKENS = 2000
GRAPH = "https://graph.microsoft.com/v1.0"
STORE_DIR = "sop_vector_store"
COLLECTION = "tynor_sops"
HISTORY_FILE = "qa_history.jsonl"
LOGO_FILE = "tynor_logo.png"

# ---- Tynor SharePoint dark palette ----
BG        = "#112133"   # page background (dark navy)
BG_DEEP   = "#0C1826"   # deeper navy (gradient depth, code blocks)
SURFACE   = "#1A3047"   # cards / raised surfaces
BORDER    = "#2B4763"   # subtle borders on dark
PRIMARY   = "#14A3A5"   # teal accent (buttons, links)
ACCENT    = "#2ED8D3"   # bright cyan (highlights, hover)
TEXT      = "#FFFFFF"   # primary text on dark
TEXT_SOFT = "#D8E2EC"   # secondary text
MUTED     = "#8FA5B8"   # tertiary / muted text

DETAILS_MARKER = "===DETAILS==="

SYSTEM_INSTRUCTIONS = f"""You are the Tynor SOP Agent, an internal knowledge assistant for Tynor Orthotics.
Answer using ONLY the SOP excerpts provided. These are controlled company documents; accuracy is critical.

The excerpts contain Markdown tables. Many tables look similar (the same role names appear in different tables),
so READ TABLE TITLES CAREFULLY and use the row from the CORRECT table. Banded/conditional tables
(e.g. "if manpower cost is between 1.5% and 2.5%, then 7%") require matching the exact band.

CRITICAL - HOW TO REASON BEFORE ANSWERING:
Work out the complete answer in your head FIRST, before writing anything:
  1. Identify the correct table by its title.
  2. For banded/conditional values, carefully determine which band the value falls into.
     Pay special attention to boundary values (e.g. is 1.9% in "below 2%" or "2%-4%"?
     1.9% is below 2%, so it is the "below 2%" band). Double-check the band BEFORE you write.
  3. For eligibility, check EVERY required condition, not just the first one.
Only after you have fully resolved and verified the answer do you begin writing.

The SHORT answer you write must be your FINAL, verified conclusion. It must be correct on the
first try. NEVER write a preliminary guess and then correct it. NEVER write words like
"wait", "re-check", "actually", "correcting", "no —", or show yourself changing your mind.
If you catch a mistake while thinking, fix it BEFORE you start writing - the reader must only
ever see the single correct final answer, never the correction process.

OUTPUT FORMAT - follow exactly:
1. First: the SHORT answer only. 1-3 sentences maximum. Just the direct, final answer - no citations,
   no reasoning, no preamble, no self-correction.
2. Then on its own line write exactly: {DETAILS_MARKER}
3. After the marker: the full detail - which section/table you used, which band/condition you
   matched and why, verbatim supporting quotes, and the source citation
   (SOP file name + exact table title or section). Present this as a clean explanation of the
   correct answer - do NOT narrate a trial-and-error process here either.

Rules:
- Use ONLY information present in the excerpts. Never use outside knowledge.
- NEVER fabricate. If the answer is not clearly present, the short answer is:
  "I couldn't find this in the SOPs." (still include the marker and explain what you searched in details)
- Only put text in quotation marks if it appears verbatim in an excerpt.
- When a value comes from a conditional table, state in the details which band you matched.
"""


# ----------------------------- GRAPH -----------------------------
def get_graph_token():
    import msal
    app = msal.ConfidentialClientApplication(
        GRAPH_CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{GRAPH_TENANT_ID}",
        client_credential=GRAPH_CLIENT_SECRET,
    )
    result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
    if "access_token" not in result:
        raise RuntimeError(
            f"Graph auth failed: {result.get('error')} - {result.get('error_description')}. "
            "Most common cause: the app registration hasn't been granted admin consent."
        )
    return result["access_token"]


def graph_get(token, url):
    import requests
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r


def get_site_id(token):
    return graph_get(token, f"{GRAPH}/sites/{SP_HOSTNAME}:/sites/{SP_SITE_PATH}").json()["id"]


def get_drive_id(token, site_id):
    """Find the drive (document library) id by display name; fall back to default drive."""
    if not SP_LIBRARY_NAME:
        return graph_get(token, f"{GRAPH}/sites/{site_id}/drive").json()["id"]
    drives = graph_get(token, f"{GRAPH}/sites/{site_id}/drives").json().get("value", [])
    for d in drives:
        if d.get("name", "").strip().lower() == SP_LIBRARY_NAME.strip().lower():
            return d["id"]
    # fallback
    return graph_get(token, f"{GRAPH}/sites/{site_id}/drive").json()["id"]


def list_children(token, drive_id, folder_path):
    if folder_path:
        url = f"{GRAPH}/drives/{drive_id}/root:/{folder_path}:/children"
    else:
        url = f"{GRAPH}/drives/{drive_id}/root/children"
    return graph_get(token, url).json().get("value", [])


def walk_files(token, drive_id, folder_path, department=None):
    """Return (name, download_url, etag, department, web_url). Department = the first-level subfolder name."""
    files = []
    for item in list_children(token, drive_id, folder_path):
        name = item.get("name", "")
        if "folder" in item:
            sub = f"{folder_path}/{name}" if folder_path else name
            # department is the FIRST level under the main SOPs folder
            dept = name if department is None else department
            files.extend(walk_files(token, drive_id, sub, dept))
        elif "file" in item:
            dl = item.get("@microsoft.graph.downloadUrl")
            etag = item.get("eTag") or item.get("lastModifiedDateTime") or ""
            web_url = item.get("webUrl", "")  # link to open the file in SharePoint
            if dl and name.lower().endswith((".docx", ".pdf", ".txt", ".md")):
                files.append((name, dl, etag, department or "General", web_url))
    return files


def download_bytes(url):
    import requests
    r = requests.get(url)
    r.raise_for_status()
    return r.content


# ----------------------------- EXTRACTION -----------------------------
def table_to_markdown(rows):
    rows = [[(c if c is not None else "").strip().replace("\n", " ") for c in r] for r in rows]
    rows = [r for r in rows if any(cell for cell in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def extract_docx(content):
    import docx
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.text.paragraph import Paragraph
    from docx.table import Table
    doc = docx.Document(io.BytesIO(content))
    out, last_heading = [], "Table"
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            text = Paragraph(child, doc).text.strip()
            if text:
                out.append(text)
                if len(text) < 80:
                    last_heading = text
        elif isinstance(child, CT_Tbl):
            table = Table(child, doc)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            md = table_to_markdown(rows)
            if md:
                out.append(f"\n[TABLE: {last_heading}]\n{md}\n")
    return "\n".join(out)


def extract_pdf(content):
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(content)).pages)


def extract_text(name, content):
    n = name.lower()
    if n.endswith(".docx"):
        return extract_docx(content)
    if n.endswith(".pdf"):
        return extract_pdf(content)
    return content.decode("utf-8", errors="ignore")


def chunk_document(name, dept, text):
    chunks, current, cur_len = [], [], 0
    for line in text.split("\n"):
        is_table_line = line.strip().startswith("|") or line.strip().startswith("[TABLE")
        if cur_len + len(line) > CHUNK_MAX_CHARS and not is_table_line and current:
            chunks.append("\n".join(current))
            current, cur_len = [], 0
        current.append(line)
        cur_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return [f"[Department: {dept}] [Source: {name}]\n{c.strip()}" for c in chunks if c.strip()]


# ----------------------------- VECTOR STORE -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def get_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)


def ensure_collection():
    """Create the Qdrant collection (and payload indexes) once, sized to the embedder."""
    from qdrant_client.models import Distance, VectorParams
    client = get_client()
    if not client.collection_exists(COLLECTION):
        dim = get_embedder().get_sentence_embedding_dimension()
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        # keyword indexes on the fields we filter/delete by (routing + incremental sync)
        for field in ("department", "fkey", "fname"):
            client.create_payload_index(COLLECTION, field_name=field, field_schema="keyword")


def collection_count():
    client = get_client()
    if not client.collection_exists(COLLECTION):
        return 0
    return client.count(collection_name=COLLECTION, exact=True).count


def file_key(name, dept, etag):
    return hashlib.md5(f"{dept}::{name}::{etag}".encode()).hexdigest()


def sync_store(status_area):
    from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct, FilterSelector
    import uuid
    embedder = get_embedder()
    ensure_collection()
    client = get_client()
    token = get_graph_token()
    site_id = get_site_id(token)
    drive_id = get_drive_id(token, site_id)
    files = walk_files(token, drive_id, SP_FOLDER_PATH)

    # which files are already indexed? scroll all points (payload-only, no vectors)
    existing_keys = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, with_payload=["fkey"], with_vectors=False,
            limit=1000, offset=offset,
        )
        for p in points:
            k = (p.payload or {}).get("fkey")
            if k:
                existing_keys.add(k)
        if offset is None:
            break

    current_keys, new_files, depts = set(), 0, set()
    for name, dl, etag, dept, web_url in files:
        depts.add(dept)
        fkey = file_key(name, dept, etag)
        current_keys.add(fkey)
        if fkey in existing_keys:
            continue
        # remove any older version of this file (same name + department)
        client.delete(collection_name=COLLECTION, points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="fname", match=MatchValue(value=name)),
            FieldCondition(key="department", match=MatchValue(value=dept)),
        ])))
        try:
            text = extract_text(name, download_bytes(dl)).strip()
        except Exception:
            continue
        if not text:
            continue
        doc_chunks = chunk_document(name, dept, text)
        if not doc_chunks:
            continue
        embs = embedder.encode(doc_chunks, normalize_embeddings=True, show_progress_bar=False).tolist()
        pts = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{fkey}_{i}")),
                vector=embs[i],
                payload={"text": doc_chunks[i], "fname": name, "fkey": fkey,
                         "department": dept, "web_url": web_url or ""},
            )
            for i in range(len(doc_chunks))
        ]
        client.upsert(collection_name=COLLECTION, points=pts)
        new_files += 1
        if status_area:
            status_area.info(f"Indexed: {dept} / {name}")

    # delete files that vanished from SharePoint
    for rk in (existing_keys - current_keys):
        client.delete(collection_name=COLLECTION, points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="fkey", match=MatchValue(value=rk)),
        ])))

    return len(files), new_files, sorted(depts)


# ---- Auto department routing (uses embeddings, no extra AI call) ----
@st.cache_data(show_spinner=False)
def department_centroids():
    """Average embedding per department, computed from stored chunks. Cached."""
    client = get_client()
    if not client.collection_exists(COLLECTION):
        return {}
    by_dept = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, with_payload=["department"], with_vectors=True,
            limit=1000, offset=offset,
        )
        for p in points:
            d = (p.payload or {}).get("department", "General")
            if p.vector is not None:
                by_dept.setdefault(d, []).append(p.vector)
        if offset is None:
            break
    return {d: np.mean(np.array(v), axis=0) for d, v in by_dept.items() if v}


def route_departments(question):
    embedder = get_embedder()
    q = embedder.encode([question], normalize_embeddings=True)[0]
    cents = department_centroids()
    if not cents:
        return None
    scored = sorted(((float(np.dot(q, c)), d) for d, c in cents.items()), reverse=True)
    top = [d for _, d in scored[:ROUTE_DEPTS]]
    # weak-match fallback: if best score is low, don't restrict (search everything)
    if scored and scored[0][0] < 0.15:
        return None
    depts = set(top) | (ALWAYS_INCLUDE & set(cents.keys()))
    return list(depts)


def retrieve(question):
    from qdrant_client.models import Filter, FieldCondition, MatchAny
    client = get_client()
    embedder = get_embedder()
    if not client.collection_exists(COLLECTION):
        return [], None, {"weak": True, "best_sim": 0.0, "n_chunks": 0}, {}
    q = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    depts = route_departments(question)
    qfilter = Filter(must=[FieldCondition(key="department", match=MatchAny(any=depts))]) if depts else None

    hits = client.query_points(collection_name=COLLECTION, query=q, limit=TOP_K,
                               query_filter=qfilter, with_payload=True).points
    # safety fallback: if the scoped search returned nothing, search everything
    if not hits and qfilter is not None:
        hits = client.query_points(collection_name=COLLECTION, query=q, limit=TOP_K,
                                   with_payload=True).points

    docs = [(h.payload or {}).get("text", "") for h in hits]

    # --- Retrieval confidence signal (no extra API call) ---
    # Qdrant COSINE returns similarity directly (higher = better match); with normalized
    # vectors this sits ~0..1. We flag "weak" when nothing was retrieved, or the BEST match
    # is poor, meaning the right SOP section may not have been found and should be verified.
    scores = [h.score for h in hits]
    best_sim = max(scores) if scores else 0.0
    WEAK_THRESHOLD = 0.35  # below this, the top chunk isn't a strong match to the question
    weak = (not docs) or (best_sim < WEAK_THRESHOLD)
    conf = {"weak": weak, "best_sim": round(best_sim, 3), "n_chunks": len(docs)}

    # --- Source links: unique (filename -> SharePoint URL) for the retrieved chunks ---
    sources = {}
    for h in hits:
        md = h.payload or {}
        fn = md.get("fname", "")
        if fn and fn not in sources:
            sources[fn] = md.get("web_url", "")

    return docs, depts, conf, sources


# ----------------------------- HISTORY -----------------------------
def log_qa(question, short, details, mode, depts):
    try:
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                "mode": mode, "departments": depts, "question": question,
                "answer": short, "details": details,
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass


def read_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    out = []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def split_answer(raw):
    if raw is None:
        return "", ""
    if DETAILS_MARKER in raw:
        short, details = raw.split(DETAILS_MARKER, 1)
        return short.strip(), details.strip()
    return raw.strip(), ""


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Tynor SOP Agent", layout="centered", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* base / dark canvas */
html, body, [class*="css"] {{ font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; color:{TEXT_SOFT}; -webkit-font-smoothing:antialiased; }}
.stApp {{ background:radial-gradient(1100px 520px at 85% -12%, rgba(20,163,165,0.14) 0%, transparent 60%), radial-gradient(900px 460px at -10% 110%, rgba(46,216,211,0.08) 0%, transparent 55%), {BG}; color:{TEXT_SOFT}; }}

/* remove top header bar / any reserved image space */
header[data-testid="stHeader"] {{ height:0 !important; min-height:0 !important; background:transparent !important; }}
[data-testid="stToolbar"] {{ display:none !important; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:1.1rem !important; }}

/* text */
h1, h2, h3, h4, h5, h6, p, li, label {{ color:{TEXT_SOFT}; }}
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {{ color:{TEXT_SOFT}; }}
strong, b {{ color:{TEXT}; }}

/* sidebar */
section[data-testid="stSidebar"] {{ visibility:visible !important; transform:none !important; min-width:270px !important; background:linear-gradient(180deg,{SURFACE} 0%,{BG} 100%); border-right:1px solid {BORDER}; }}
section[data-testid="stSidebar"] * {{ color:{TEXT_SOFT}; }}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ font-weight:700; letter-spacing:0.02em; text-transform:uppercase; font-size:11px; color:{MUTED}; }}
[data-testid="stSidebarCollapseButton"] {{ display:none !important; }}

/* header block (text only, no image column) */
.tynor-header {{ background:linear-gradient(135deg,{PRIMARY} 0%, #0F6E70 55%, {SURFACE} 100%); padding:20px 26px; border-radius:16px; margin:0 0 20px 0; border:1px solid {BORDER}; box-shadow:0 14px 34px rgba(0,0,0,0.38), inset 0 1px 0 rgba(255,255,255,0.08); }}
.tynor-header h1 {{ color:{TEXT} !important; margin:0; font-size:24px; font-weight:800; letter-spacing:-0.02em; }}
.tynor-header p {{ color:rgba(255,255,255,0.82) !important; margin:6px 0 0 0; font-size:13px; font-weight:500; }}

/* chat bubbles */
[data-testid="stChatMessage"] {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:14px; padding:6px 12px; margin-bottom:10px; box-shadow:0 1px 3px rgba(0,0,0,0.35); transition:border-color .2s ease, box-shadow .2s ease, transform .2s ease; }}
[data-testid="stChatMessage"]:hover {{ border-color:{PRIMARY}; box-shadow:0 8px 24px rgba(20,163,165,0.16); transform:translateY(-1px); }}
[data-testid="stChatMessage"] * {{ color:{TEXT_SOFT}; }}

/* expanders */
[data-testid="stExpander"] {{ border:1px solid {BORDER} !important; border-radius:10px !important; background:{SURFACE} !important; box-shadow:0 1px 2px rgba(0,0,0,0.30); transition:border-color .2s ease, box-shadow .2s ease; }}
[data-testid="stExpander"]:hover {{ border-color:{PRIMARY} !important; box-shadow:0 4px 14px rgba(20,163,165,0.16); }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] p, [data-testid="stExpander"] span {{ color:{TEXT_SOFT} !important; }}
[data-testid="stExpander"] svg {{ fill:{MUTED} !important; }}

/* chat input + bottom bar (kill the white band and the grey/red input pill) */
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"], .stChatFloatingInputContainer, [data-testid="stChatFloatingInputContainer"] {{ background:{BG} !important; }}
[data-testid="stChatInput"] {{ background:{SURFACE} !important; border:1px solid {BORDER} !important; border-radius:12px !important; box-shadow:none !important; }}
[data-testid="stChatInput"]:focus-within {{ border-color:{PRIMARY} !important; box-shadow:0 0 0 3px rgba(20,163,165,0.22) !important; }}
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] [data-baseweb="textarea"], [data-testid="stChatInput"] [data-baseweb="base-input"] {{ background:transparent !important; border:none !important; box-shadow:none !important; }}
.stChatInput textarea, [data-testid="stChatInput"] textarea {{ font-family:'Inter',sans-serif !important; background:transparent !important; color:{TEXT} !important; border:none !important; box-shadow:none !important; }}
.stChatInput textarea::placeholder {{ color:{MUTED} !important; }}
[data-testid="stChatInputSubmitButton"] {{ background:transparent !important; color:{ACCENT} !important; box-shadow:none !important; }}
[data-testid="stChatInputSubmitButton"]:hover {{ background:rgba(20,163,165,0.15) !important; }}
[data-testid="stChatInputSubmitButton"] svg {{ fill:{ACCENT} !important; }}
[data-testid="stChatInput"] button {{ background:transparent !important; color:{ACCENT} !important; }}

/* tabs */
button[data-baseweb="tab"] {{ font-weight:600 !important; color:{MUTED} !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{ACCENT} !important; }}
div[data-baseweb="tab-highlight"] {{ background-color:{PRIMARY} !important; }}
div[data-baseweb="tab-border"] {{ background-color:{BORDER} !important; }}

/* links */
a {{ color:{ACCENT} !important; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}

/* buttons */
.stButton > button {{ background:linear-gradient(135deg,{PRIMARY},#0F6E70); color:{TEXT}; border:none; border-radius:10px; font-weight:600; box-shadow:0 4px 12px rgba(20,163,165,0.28); transition:transform .15s ease, box-shadow .15s ease; }}
.stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 8px 20px rgba(20,163,165,0.4); color:{TEXT}; }}

/* code blocks (Copy response) */
.stCode, pre, code {{ background:{BG_DEEP} !important; border-radius:10px !important; }}
.stCode *, pre, pre code {{ color:{TEXT_SOFT} !important; }}
[data-testid="stCode"] {{ border:1px solid {BORDER} !important; }}

/* alerts */
[data-testid="stAlert"], .stAlert {{ background:{SURFACE} !important; border:1px solid {BORDER} !important; border-left:3px solid {PRIMARY} !important; border-radius:10px !important; color:{TEXT_SOFT} !important; }}
[data-testid="stAlert"] *, .stAlert * {{ color:{TEXT_SOFT} !important; }}

/* captions */
[data-testid="stCaptionContainer"], .stCaption, small {{ color:{MUTED} !important; }}
/* hide Streamlit running indicator + dark-style any residual spinner */
[data-testid="stStatusWidget"] {{ display:none !important; }}
[data-testid="stSpinner"] {{ background:transparent !important; }}
[data-testid="stSpinner"] *, [data-testid="stSpinner"] p {{ color:{TEXT_SOFT} !important; }}
/* Chat/History view toggle (main-area radio, styled as tabs) */
[data-testid="stMain"] div[role="radiogroup"], section.main div[role="radiogroup"] {{ gap:16px !important; border-bottom:1px solid {BORDER}; padding-bottom:8px; margin-bottom:14px; }}
[data-testid="stMain"] div[role="radiogroup"] input[type="radio"], section.main div[role="radiogroup"] input[type="radio"] {{ accent-color:{PRIMARY}; }}
[data-testid="stMain"] div[role="radiogroup"] label p, section.main div[role="radiogroup"] label p {{ color:{MUTED} !important; font-weight:600 !important; font-size:15px !important; }}
[data-testid="stMain"] div[role="radiogroup"] label:hover p, section.main div[role="radiogroup"] label:hover p {{ color:{TEXT_SOFT} !important; }}
[data-testid="stMain"] div[role="radiogroup"] label:has(input:checked) p, section.main div[role="radiogroup"] label:has(input:checked) p {{ color:{ACCENT} !important; }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width=120)
    st.markdown("### Settings")
    mode = st.radio("Answer mode", ["Precise", "Quick"], index=0,
        help=("Precise (DeepSeek) - recommended. Reliable on increment tables, eligibility rules "
              "and cross-document questions.\n\nQuick (DeepSeek) - faster path for simple lookups."))
    active_model = MODEL_PRECISE if mode == "Precise" else MODEL_QUICK
    st.caption("Precise is recommended for increment, eligibility and policy calculations.")
    st.markdown("---")
    if st.button("Sync SOPs from SharePoint"):
        st.session_state._force_sync = True
    if st.button("Rebuild index (wipe & re-embed)"):
        try:
            get_client().delete_collection(COLLECTION)
        except Exception:
            pass
        st.cache_resource.clear()
        st.cache_data.clear()
        st.session_state.pop("synced", None)
        st.session_state._force_sync = True
        st.rerun()
    health_slot = st.empty()

st.markdown(f"""
<div class="tynor-header">
    <h1>Tynor SOP Agent</h1>
    <p>Ask a question about Tynor's SOPs &mdash; answers come only from the documents, with citations.</p>
</div>
""", unsafe_allow_html=True)

_missing = [k for k, v in {"DEEPSEEK_API_KEY": DEEPSEEK_API_KEY, "QDRANT_URL": QDRANT_URL,
    "QDRANT_API_KEY": QDRANT_API_KEY, "GRAPH_TENANT_ID": GRAPH_TENANT_ID,
    "GRAPH_CLIENT_ID": GRAPH_CLIENT_ID, "GRAPH_CLIENT_SECRET": GRAPH_CLIENT_SECRET}.items() if not v]
if _missing:
    st.error("Configuration incomplete. Set these as environment variables: " + ", ".join(_missing))
    st.stop()

try:
    _existing = collection_count()
except Exception:
    _existing = 0

# Only contact SharePoint if the user clicked a button, OR there is no index yet.
# If an index already exists on disk, load it and skip SharePoint (avoids the
# startup hang when SharePoint/Graph is slow or unreachable).
if st.session_state.get("_force_sync") or _existing == 0:
    box = st.empty()
    box.info("Checking SharePoint and updating the SOP index (only new/changed files are processed)...")
    try:
        total, new, depts = sync_store(box)
        st.cache_data.clear()
        st.session_state.synced = True
        st.session_state._force_sync = False
        st.session_state.total_files = total
        st.session_state.depts = depts
        box.empty()
    except Exception as e:
        st.session_state._force_sync = False
        if _existing > 0:
            st.session_state.synced = True
            box.warning(f"Couldn't reach SharePoint ({e}). Using the existing on-disk index.")
        else:
            box.error(f"Couldn't sync SOPs from SharePoint and no local index exists: {e}")
            st.stop()
else:
    st.session_state.synced = True

try:
    _chunks = collection_count()
except Exception:
    _chunks = 0
if _chunks == 0:
    health_slot.warning("No SOP chunks are indexed. Click **Sync SOPs from SharePoint** in the sidebar (or **Rebuild index** to wipe & re-embed).")
else:
    health_slot.caption(f"Index health: {_chunks} chunks")

debug = st.query_params.get("debug") == "1"
if "messages" not in st.session_state:
    st.session_state.messages = []

view = st.radio("view", ["Chat", "History"], horizontal=True,
                label_visibility="collapsed", key="view_toggle")

if view == "History":
    hist = read_history()
    st.caption(f"{len(hist)} questions logged")
    for h in reversed(hist):
        with st.expander(f"{h['time']}  ·  {h['question'][:90]}"):
            st.markdown(f"**Question**  \n{h['question']}")
            st.markdown(f"**Answer**  \n{h['answer']}")
            if h.get("departments"):
                st.caption("Searched in: " + ", ".join(h["departments"]))
            if h.get("details"):
                st.markdown("**Logic and citations**")
                st.markdown(h["details"])
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                if m.get("weak"):
                    st.warning("⚠️ Weak source match — the most relevant SOP section may not have "
                               "been found for this question. Please verify against the source "
                               "document before relying on this answer.")
                if m.get("details"):
                    with st.expander("View logic and citations"):
                        st.markdown(m["details"])
                with st.expander("Copy response"):
                    st.code(m["content"] + ("\n\n" + m["details"] if m.get("details") else ""), language=None)
                _sources = m.get("sources") or {}
                if _sources:
                    with st.expander("Open source file in SharePoint"):
                        for _fn, _url in _sources.items():
                            if _url:
                                st.markdown(f"- [{_fn}]({_url})")
                            else:
                                st.markdown(f"- {_fn}  \n  _(link unavailable — re-sync SOPs to capture it)_")

question = st.chat_input("Ask about the SOPs...") if view == "Chat" else None
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    # Show the user's message IMMEDIATELY (before the slow API call) so it doesn't
    # appear blank until the answer is ready.
    with st.chat_message("user"):
        st.markdown(question)

    used_depts = None
    conf = {"weak": False, "best_sim": 0.0, "n_chunks": 0}
    sources = {}
    with st.chat_message("assistant"):
        with st.spinner("Searching the SOPs..."):
            try:
                from openai import OpenAI
                retrieved, used_depts, conf, sources = retrieve(question)
                context = "\n\n---\n\n".join(retrieved)
                client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

                # Reasoning/thinking mode:
                #  - DeepSeek-direct: enabled via extra_body on the Precise model.
                #  - DeepInfra / other hosts: reasoning is a SEPARATE model string, not a param,
                #    so we only send this param when actually talking to DeepSeek-direct
                #    (prevents "unknown parameter" errors that would blank out the response).
                extra = {}
                if mode == "Precise" and "api.deepseek.com" in DEEPSEEK_BASE_URL:
                    extra = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}

                resp = client.chat.completions.create(
                    model=active_model,
                    max_tokens=MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                        {"role": "user", "content": f"SOP EXCERPTS:\n\n{context}\n\nQUESTION: {question}"},
                    ],
                    extra_body=extra,
                )
                raw = resp.choices[0].message.content
                if not raw or not raw.strip():
                    raw = ("I couldn't get a response from the model (empty reply). "
                           "This usually means the model name is wrong for this provider, or the "
                           "request was rejected. Check MODEL_QUICK / MODEL_PRECISE match the provider "
                           "in DEEPSEEK_BASE_URL.")
            except Exception as e:
                raw = f"Sorry, something went wrong while answering: {e}"
    short, details = split_answer(raw)
    st.session_state.messages.append({"role": "assistant", "content": short, "details": details,
                                      "weak": conf.get("weak", False),
                                      "best_sim": conf.get("best_sim", 0.0),
                                      "sources": sources})
    log_qa(question, short, details, mode, used_depts or ["All"])
    st.rerun()
