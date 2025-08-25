import json
import logging
import difflib
from pathlib import Path
from typing import List, Set, Dict, Tuple, Iterable

import spacy

# --------------------------- logging ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("character_analysis.log"), logging.StreamHandler()],
)

# --------------------------- config toggles ---------------------------
PREFER_GPU = True
MODEL_CANDIDATES = ["en_core_web_trf", "en_core_web_sm"]

WINDOW_SENTENCES = 2
MIN_MENTIONS = 2
SIMILARITY_THRESHOLD = 0.93

CHUNK_TARGET_CHARS = 200_000
NLP_MAX_LENGTH_SAFETY = 2_000_000
BATCH_SIZE = 4

HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord", "madam", "madame", "prof", "professor"}

NICKNAMES = {
    "liz": "elizabeth", "beth": "elizabeth", "bess": "elizabeth", "eliza": "elizabeth", "betsy": "elizabeth",
    "bill": "william", "will": "william", "billy": "william", "liam": "william",
    "bob": "robert", "bobby": "robert", "rob": "robert", "robbie": "robert", "bert": "robert",
    "kate": "katherine", "kathy": "katherine", "katy": "katherine", "kitty": "katherine",
    "jack": "john", "johnny": "john",
    "jim": "james", "jimmy": "james",
    "tom": "thomas", "tommy": "thomas",
    "harry": "henry",
}

# --------------------------- spaCy load ---------------------------
def load_nlp():
    if PREFER_GPU:
        try:
            spacy.require_gpu()
            logging.info("Using GPU for spaCy.")
        except Exception:
            logging.info("GPU not available; using CPU.")

    last_err = None
    for name in MODEL_CANDIDATES:
        try:
            nlp = spacy.load(name)
            if "parser" in nlp.pipe_names:
                nlp.disable_pipes("parser")
                logging.info("Disabled parser to reduce memory usage.")
            if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
                logging.info("Added sentencizer.")
            nlp.max_length = max(nlp.max_length, NLP_MAX_LENGTH_SAFETY)
            logging.info(f"Loaded spaCy model: {name}")
            return nlp
        except Exception as e:
            last_err = e
            logging.warning(f"Could not load {name}: {e}")
    raise RuntimeError(f"No suitable spaCy model found. Last error: {last_err}")

nlp = load_nlp()

# --------------------------- IO helpers ---------------------------
def read_text_file(file_path: Path) -> str:
    if not file_path.is_file():
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        text = file_path.read_text(encoding="utf-8")
        logging.info(f"Successfully read file: {file_path}")
        return text
    except Exception as e:
        logging.exception(f"Error reading file {file_path}: {e}")
        raise

def get_txt_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is not a directory: {input_dir}")
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if not files:
        logging.warning(f"No .txt files found in {input_dir}")
    else:
        logging.info(f"Found {len(files)} .txt file(s) in {input_dir}")
    return files

# --------------------------- chunking ---------------------------
def chunk_text(text: str, target_chars: int = CHUNK_TARGET_CHARS) -> List[str]:
    paras = text.split("\n\n")
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0

    for p in paras:
        plen = len(p)
        if plen > target_chars:
            flush()
            start = 0
            while start < plen:
                end = min(start + target_chars, plen)
                chunks.append(p[start:end])
                start = end
            continue

        if buf_len + plen + (2 if buf else 0) <= target_chars:
            buf.append(p)
            buf_len += plen + (2 if buf else 0)
        else:
            flush()
            buf.append(p)
            buf_len = plen

    flush()
    return chunks

# --------------------------- name normalization ---------------------------
def strip_possessive(name: str) -> str:
    return name.rstrip().removesuffix("'s").removesuffix("’s").strip()

def normalize_tokens(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        t0 = "".join([ch for ch in t if ch.isalpha() or ch in "-’'"])
        t0 = t0.strip("'’").lower()
        if not t0:
            continue
        if t0 in HONORIFICS:
            continue
        t0 = NICKNAMES.get(t0, t0)
        out.append(t0)
    return out

def normalize_name(raw: str) -> str:
    raw = strip_possessive(raw)
    toks = normalize_tokens(raw.split())
    return " ".join(t.title() for t in toks)

# --------------------------- extraction pass ---------------------------
def extract_person_mentions(doc) -> List[Tuple[int, str]]:
    mentions = []
    sent_starts = [s.start for s in doc.sents]

    def sent_index_of_token(tok_i: int) -> int:
        lo, hi = 0, len(sent_starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if sent_starts[mid] <= tok_i:
                lo = mid + 1
            else:
                hi = mid - 1
        return hi

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            norm = normalize_name(ent.text)
            if norm:
                sidx = sent_index_of_token(ent.start)
                mentions.append((sidx, norm))
    return mentions

# --------------------------- alias clustering ---------------------------
def last_name(name: str) -> str:
    parts = name.split()
    return parts[-1] if parts else ""

def share_last_name(a: str, b: str) -> bool:
    return last_name(a) and last_name(a) == last_name(b)

def is_subname(a: str, b: str) -> bool:
    aw = a.split()
    bw = b.split()
    return set(aw).issubset(set(bw)) or set(bw).issubset(set(aw))

def build_alias_map(mention_counts: Dict[str, int]) -> Dict[str, str]:
    names = list(mention_counts.keys())
    names.sort(key=lambda n: (len(n), mention_counts[n]), reverse=True)

    alias_to_canon: Dict[str, str] = {}
    canonicals: List[str] = []

    for name in names:
        if name in alias_to_canon:
            continue
        canonical = name
        alias_to_canon[name] = canonical
        canonicals.append(canonical)

        for other in names:
            if other in alias_to_canon:
                continue
            similar = (
                is_subname(other, canonical)
                or (share_last_name(other, canonical) and other[0] == canonical[0])
                or difflib.SequenceMatcher(None, other, canonical).ratio() >= SIMILARITY_THRESHOLD
            )
            if similar:
                alias_to_canon[other] = canonical

    return alias_to_canon

# --------------------------- co-occurrence (windowed) ---------------------------
def sliding_windows(items: List[int], k: int) -> Iterable[Tuple[int, int]]:
    if k <= 1:
        for s in items:
            yield (s, s)
        return
    for i in range(len(items) - k + 1):
        yield (items[i], items[i + k - 1])

def build_interaction_matrix_from_mentions(
    sentence_mentions: Dict[int, Set[str]], characters: List[str]
) -> List[List[int]]:
    if not characters:
        logging.warning("No characters found, skipping interaction matrix generation.")
        return []
    index = {c: i for i, c in enumerate(characters)}
    n = len(characters)
    M = [[0] * n for _ in range(n)]

    sidxs = sorted(sentence_mentions.keys())
    for start, end in sliding_windows(sidxs, WINDOW_SENTENCES):
        window_chars: Set[str] = set()
        for s in range(start, end + 1):
            window_chars.update(sentence_mentions.get(s, set()))
        if len(window_chars) < 2:
            continue
        wc = list(window_chars)
        for i in range(len(wc)):
            for j in range(i + 1, len(wc)):
                a, b = wc[i], wc[j]
                ai, bi = index[a], index[b]
                M[ai][bi] += 1
                M[bi][ai] += 1
    logging.info(f"Built interaction matrix of size {n}x{n}.")
    return M

# --------------------------- streaming a file in waves ---------------------------
def process_file_in_chunks(path: Path):
    """
    Yields (doc, last_sent_idx) for each chunk of the file, processed with nlp.pipe.
    """
    text = read_text_file(path)
    chunks = chunk_text(text, CHUNK_TARGET_CHARS)
    logging.info(f"{path.name}: split into {len(chunks)} chunk(s).")

    for doc in nlp.pipe(chunks, batch_size=BATCH_SIZE):
        if not doc.has_annotation("SENT_START"):
            with doc.retokenize():
                pass

        last_sent_idx = -1
        for i, _ in enumerate(doc.sents):
            last_sent_idx = i
        yield doc, last_sent_idx

# --------------------------- main corpus analysis ---------------------------
def analyze_corpus(input_dir: Path, output_file: Path) -> None:
    logging.info("Starting corpus analysis...")

    txt_files = get_txt_files(input_dir)
    if not txt_files:
        return

    all_mentions: List[Tuple[int, str]] = []
    mention_counts: Dict[str, int] = {}
    global_sent_offset = 0

    for p in txt_files:
        for doc, last_sent_idx in process_file_in_chunks(p):
            mentions = extract_person_mentions(doc)
            for sidx, name in mentions:
                sidx_g = sidx + global_sent_offset
                all_mentions.append((sidx_g, name))
                mention_counts[name] = mention_counts.get(name, 0) + 1
            global_sent_offset += (last_sent_idx + 1)

    if not all_mentions:
        logging.warning("No PERSON entities found in the corpus.")
        return

    mention_counts = {k: v for k, v in mention_counts.items() if v >= MIN_MENTIONS}
    if not mention_counts:
        logging.warning(f"All names filtered out by MIN_MENTIONS={MIN_MENTIONS}. Lower the threshold?")
        return

    alias_to_canon = build_alias_map(mention_counts)
    canonicals = sorted({alias_to_canon[a] for a in mention_counts.keys()})
    logging.info(f"Merged to {len(canonicals)} canonical character(s).")

    sentence_mentions: Dict[int, Set[str]] = {}
    for sidx_g, raw in all_mentions:
        if raw not in mention_counts:
            continue
        canon = alias_to_canon.get(raw, raw)
        sentence_mentions.setdefault(sidx_g, set()).add(canon)

    matrix = build_interaction_matrix_from_mentions(sentence_mentions, canonicals)

    data = {"characters": canonicals, "matrix": matrix}
    try:
        output_file.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
        logging.info(f"Interaction data successfully written to {output_file}")
    except Exception as e:
        logging.exception(f"Error writing JSON file {output_file}: {e}")
        raise

# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    input_dir = Path("input")
    output_path = Path("character_interactions.json")

    try:
        analyze_corpus(input_dir, output_path)
        logging.info("Process completed.")
    except Exception as e:
        logging.error(f"Failed to analyze corpus: {e}")
        raise
