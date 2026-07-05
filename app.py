import argparse
import importlib
import json
import logging
import difflib
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Set, Dict, Tuple

import spacy

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("character_analysis.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

PREFER_GPU = True
MODEL_CANDIDATES = ["en_core_web_trf", "en_core_web_sm"]

WINDOW_SENTENCES = 2
MIN_MENTIONS = 2
SIMILARITY_THRESHOLD = 0.97
CONTEXT_SIMILARITY_THRESHOLD = 0.55

CHUNK_TARGET_CHARS = 200_000
NLP_MAX_LENGTH_SAFETY = 2_000_000
BATCH_SIZE = 4

ENABLE_COREF = True
COREF_CANDIDATES = [
    "coreferee",          
    "experimental_coref",
]

ENABLE_DEPENDENCY_INTERACTIONS = True
ENABLE_DIALOGUE_INTERACTIONS = True
ENABLE_PARAGRAPH_EDGES = True
ENABLE_SCENE_EDGES = True

SCENE_BREAK_PARAGRAPHS = 3
SCENE_BREAK_BLANK_LINES = 2

EDGE_WEIGHTS = {
    "co_mention": 1.0,
    "same_paragraph": 1.75,
    "same_scene": 1.25,
    "dialogue": 3.5,
    "dependency": 4.0,
    "coref_linked_presence": 1.5,
}

SIGNAL_PRECISION = {
    "dependency": 0.9,
    "dialogue": 0.85,
    "coref_linked_presence": 0.7,
    "same_paragraph": 0.55,
    "co_mention": 0.45,
    "same_scene": 0.35,
}

HONORIFICS = {
    "mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord",
    "madam", "madame", "prof", "professor", "capt", "captain",
    "rev", "reverend"
}

RELATION_VERBS = {
    "say", "says", "said", "tell", "tells", "told", "ask", "asks", "asked",
    "reply", "replies", "replied", "answer", "answers", "answered",
    "speak", "speaks", "spoke", "meet", "meets", "met", "greet", "greets", "greeted",
    "see", "sees", "saw", "watch", "watches", "watched", "love", "loves", "loved",
    "hate", "hates", "hated", "admire", "admires", "admired", "help", "helps", "helped",
    "follow", "follows", "followed", "visit", "visits", "visited", "find", "finds", "found",
    "marry", "marries", "married", "kiss", "kisses", "kissed", "embrace", "embraces", "embraced",
    "hug", "hugs", "hugged", "attack", "attacks", "attacked", "save", "saves", "saved",
    "warn", "warns", "warned", "write", "writes", "wrote", "send", "sends", "sent",
    "call", "calls", "called", "invite", "invites", "invited",
}

NICKNAMES = {
    "liz": "elizabeth", "beth": "elizabeth", "bess": "elizabeth", "eliza": "elizabeth", "betsy": "elizabeth",
    "bill": "william", "will": "william", "billy": "william", "liam": "william",
    "bob": "robert", "bobby": "robert", "rob": "robert", "robbie": "robert", "bert": "robert",
    "kate": "katherine", "kathy": "katherine", "katy": "katherine", "kitty": "katherine",
    "jack": "john", "johnny": "john",
    "jim": "james", "jimmy": "james",
    "tom": "thomas", "tommy": "thomas",
    "harry": "henry",
    "peggy": "margaret", "maggie": "margaret",
    "dick": "richard", "rich": "richard", "ricky": "richard",
    "ted": "edward", "eddie": "edward", "ned": "edward",
    "alex": "alexander", "lex": "alexander",
}

BAD_PERSONS = {
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
    "Christian", "English", "French", "German",
}

PRONOUNS = {
    "he", "she", "him", "her", "his", "hers",
    "they", "them", "their", "theirs",
    "himself", "herself", "themselves",
}

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

            if not ENABLE_DEPENDENCY_INTERACTIONS and "parser" in nlp.pipe_names:
                nlp.disable_pipes("parser")
                logging.info("Disabled parser to reduce memory usage.")

            if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
                logging.info("Added sentencizer.")

            if ENABLE_COREF:
                maybe_add_coref(nlp)

            nlp.max_length = max(nlp.max_length, NLP_MAX_LENGTH_SAFETY)
            logging.info(f"Loaded spaCy model: {name}")
            logging.info(f"Active pipes: {nlp.pipe_names}")
            return nlp
        except Exception as e:
            last_err = e
            logging.warning(f"Could not load {name}: {e}")
    raise RuntimeError(f"No suitable spaCy model found. Last error: {last_err}")

def maybe_add_coref(nlp):
    for coref_name in COREF_CANDIDATES:
        try:
            if coref_name == "coreferee":
                importlib.import_module("coreferee")
                if "coreferee" not in nlp.pipe_names:
                    nlp.add_pipe("coreferee")
                    logging.info("Enabled coreferee coreference resolution.")
                return
            elif coref_name not in nlp.pipe_names:
                nlp.add_pipe(coref_name)
                logging.info(f"Enabled optional coreference component: {coref_name}")
                return
        except Exception as e:
            logging.info(f"Coreference component {coref_name} unavailable: {e}")
    logging.info("No coreference component enabled; proceeding without coreference.")

nlp = None

def get_nlp():
    global nlp
    if nlp is None:
        nlp = load_nlp()
    return nlp

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
        out.append(t0)
    return out

def normalize_name(raw: str) -> str:
    raw = strip_possessive(raw)
    toks = normalize_tokens(raw.split())
    return " ".join(t.title() for t in toks)

def is_likely_valid_person_name(name: str) -> bool:
    if not name:
        return False
    if name in BAD_PERSONS:
        return False
    if len(name) < 2:
        return False
    parts = name.split()
    if len(parts) == 1:
        p = parts[0]
        if len(p) < 3:
            return False
    if any(ch.isdigit() for ch in name):
        return False
    return True

def sent_index_of_token(tok_i: int, sent_starts: List[int]) -> int:
    lo, hi = 0, len(sent_starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sent_starts[mid] <= tok_i:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi

def paragraph_index_of_char(span_start_char: int, para_spans: List[Tuple[int, int]]) -> int:
    for i, (start, end) in enumerate(para_spans):
        if start <= span_start_char < end:
            return i
    return max(0, len(para_spans) - 1)

def scene_index_of_paragraph(para_idx: int, scene_map: Dict[int, int]) -> int:
    return scene_map.get(para_idx, 0)

def compute_paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for m in re.finditer(r"(.*?)(?:\n\s*\n|$)", text, flags=re.DOTALL):
        s, e = m.span(1)
        if s == e:
            continue
        spans.append((s, e))
    return spans

def compute_scene_map(text: str, para_spans: List[Tuple[int, int]]) -> Dict[int, int]:
    scene_map: Dict[int, int] = {}
    current_scene = 0
    last_para_end = None
    consecutive_short_breaks = 0

    for para_idx, (start, end) in enumerate(para_spans):
        if last_para_end is not None:
            gap = text[last_para_end:start]
            blank_groups = gap.count("\n\n")
            if blank_groups >= SCENE_BREAK_BLANK_LINES:
                current_scene += 1
                consecutive_short_breaks = 0
            else:
                consecutive_short_breaks += 1
                if consecutive_short_breaks >= SCENE_BREAK_PARAGRAPHS:
                    current_scene += 1
                    consecutive_short_breaks = 0
        scene_map[para_idx] = current_scene
        last_para_end = end

    return scene_map

def extract_person_mentions(doc, para_spans: List[Tuple[int, int]], scene_map: Dict[int, int]) -> List[Dict]:
    mentions = []
    sent_starts = [s.start for s in doc.sents]

    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        norm = normalize_name(ent.text)
        if not norm or not is_likely_valid_person_name(norm):
            continue

        sidx = sent_index_of_token(ent.start, sent_starts)
        pidx = paragraph_index_of_char(ent.start_char, para_spans)
        scene_idx = scene_index_of_paragraph(pidx, scene_map)
        mentions.append({
            "sentence": sidx,
            "paragraph": pidx,
            "scene": scene_idx,
            "name": norm,
            "raw": ent.text,
            "start": ent.start,
            "end": ent.end,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "is_coref": False,
        })
    return mentions

def build_local_entity_lookup(mentions: List[Dict]) -> Dict[Tuple[int, int], str]:
    lookup = {}
    for m in mentions:
        lookup[(m["start"], m["end"])] = m["name"]
    return lookup

def get_coref_mentions(doc, mentions: List[Dict], para_spans: List[Tuple[int, int]], scene_map: Dict[int, int]) -> List[Dict]:
    if not ENABLE_COREF:
        return []

    coref_mentions: List[Dict] = []
    sent_starts = [s.start for s in doc.sents]
    entity_lookup = build_local_entity_lookup(mentions)

    try:
        if hasattr(doc._, "coref_chains") and doc._.coref_chains:
            for chain in doc._.coref_chains:
                canonical_name = None
                for mention in chain:
                    try:
                        span = doc[mention[0]:mention[-1] + 1]
                    except Exception:
                        continue
                    key = (span.start, span.end)
                    if key in entity_lookup:
                        canonical_name = entity_lookup[key]
                        break

                if not canonical_name:
                    continue

                for mention in chain:
                    try:
                        span = doc[mention[0]:mention[-1] + 1]
                    except Exception:
                        continue
                    txt = span.text.strip()
                    if txt.lower() not in PRONOUNS:
                        continue

                    sidx = sent_index_of_token(span.start, sent_starts)
                    pidx = paragraph_index_of_char(span.start_char, para_spans)
                    scene_idx = scene_index_of_paragraph(pidx, scene_map)
                    coref_mentions.append({
                        "sentence": sidx,
                        "paragraph": pidx,
                        "scene": scene_idx,
                        "name": canonical_name,
                        "raw": txt,
                        "start": span.start,
                        "end": span.end,
                        "start_char": span.start_char,
                        "end_char": span.end_char,
                        "is_coref": True,
                    })
            return coref_mentions
    except Exception as e:
        logging.info(f"coreferee-style extraction skipped: {e}")

    return coref_mentions

def nickname_normalized_parts(name: str) -> List[str]:
    return [NICKNAMES.get(p.lower(), p.lower()) for p in name.split()]

def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def build_name_contexts(mentions: List[Dict]) -> Dict[str, Set[str]]:
    contexts: Dict[str, Set[str]] = defaultdict(set)
    by_paragraph: Dict[int, Set[str]] = defaultdict(set)

    for m in mentions:
        by_paragraph[m["paragraph"]].add(m["name"])

    for _, names in by_paragraph.items():
        for n in names:
            contexts[n].update(x for x in names if x != n)

    return contexts

def should_merge_name_pair(a: str, b: str, name_contexts: Dict[str, Set[str]]) -> bool:
    if a == b:
        return True

    a_nick = nickname_normalized_parts(a)
    b_nick = nickname_normalized_parts(b)
    if not a_nick or not b_nick:
        return False
    a_first_n = a_nick[0]
    b_first_n = b_nick[0]
    a_last = a_nick[-1]
    b_last = b_nick[-1]

    if len(a_nick) == 1 and a_nick[0] in b_nick:
        return True
    if len(b_nick) == 1 and b_nick[0] in a_nick:
        return True

    if a_last == b_last and a_first_n == b_first_n:
        return True

    if a_first_n == b_first_n and (len(a_nick) != len(b_nick)):
        if jaccard_similarity(name_contexts.get(a, set()), name_contexts.get(b, set())) >= CONTEXT_SIMILARITY_THRESHOLD:
            return True

    ratio = difflib.SequenceMatcher(None, " ".join(a_nick), " ".join(b_nick)).ratio()
    if ratio >= SIMILARITY_THRESHOLD:
        if a_last == b_last or set(a_nick) <= set(b_nick) or set(b_nick) <= set(a_nick):
            return True
        if a_first_n == b_first_n and jaccard_similarity(name_contexts.get(a, set()), name_contexts.get(b, set())) >= CONTEXT_SIMILARITY_THRESHOLD:
            return True

    return False

def choose_canonical_name(a: str, b: str, mention_counts: Dict[str, int]) -> str:
    ca = mention_counts.get(a, 0)
    cb = mention_counts.get(b, 0)
    a_words = len(a.split())
    b_words = len(b.split())

    if a_words != b_words:
        fuller, shorter = (a, b) if a_words > b_words else (b, a)
        c_fuller = mention_counts.get(fuller, 0)
        c_shorter = mention_counts.get(shorter, 0)
        if c_fuller >= max(MIN_MENTIONS, 0.1 * c_shorter):
            return fuller
        return shorter

    if ca != cb:
        return a if ca > cb else b
    if len(a) != len(b):
        return a if len(a) > len(b) else b
    return min(a, b)

def build_alias_map(mention_counts: Dict[str, int], name_contexts: Dict[str, Set[str]]) -> Dict[str, str]:
    names = sorted(mention_counts.keys(), key=lambda n: (-mention_counts[n], -len(n), n))
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        canonical = choose_canonical_name(ra, rb, mention_counts)
        other = rb if canonical == ra else ra
        parent[other] = canonical

    buckets: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        for part in set(nickname_normalized_parts(name)):
            buckets[part].append(name)

    compared: Set[Tuple[str, str]] = set()
    for bucket in buckets.values():
        for i, name in enumerate(bucket):
            for other in bucket[i + 1:]:
                pair = (name, other) if name < other else (other, name)
                if pair in compared:
                    continue
                compared.add(pair)
                if should_merge_name_pair(name, other, name_contexts):
                    union(name, other)

    alias_to_canon = {n: find(n) for n in names}
    return alias_to_canon

def add_pair_weight(edge_evidence: Dict[Tuple[str, str], Counter], a: str, b: str, key: str, amount: int = 1) -> None:
    if a == b:
        return
    pair = tuple(sorted((a, b)))
    edge_evidence[pair][key] += amount

def edge_confidence(evidence: Counter) -> float:
    miss = 1.0
    for key, amount in evidence.items():
        if amount > 0:
            miss *= 1.0 - SIGNAL_PRECISION.get(key, 0.4)
    return round(1.0 - miss, 3)

def build_interaction_evidence(
    sentence_mentions: Dict[int, Set[str]],
    paragraph_mentions: Dict[int, Set[str]],
    scene_mentions: Dict[int, Set[str]],
) -> Dict[Tuple[str, str], Counter]:
    edge_evidence: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    sidxs = sorted(sentence_mentions.keys())
    for idx, start in enumerate(sidxs):
        window_end = start + WINDOW_SENTENCES - 1
        window_chars: Set[str] = set()
        k = idx
        while k < len(sidxs) and sidxs[k] <= window_end:
            window_chars.update(sentence_mentions[sidxs[k]])
            k += 1
        if len(window_chars) < 2:
            continue
        wc = sorted(window_chars)
        for i in range(len(wc)):
            for j in range(i + 1, len(wc)):
                add_pair_weight(edge_evidence, wc[i], wc[j], "co_mention", 1)

    if ENABLE_PARAGRAPH_EDGES:
        for chars in paragraph_mentions.values():
            if len(chars) < 2:
                continue
            wc = sorted(chars)
            for i in range(len(wc)):
                for j in range(i + 1, len(wc)):
                    add_pair_weight(edge_evidence, wc[i], wc[j], "same_paragraph", 1)

    if ENABLE_SCENE_EDGES:
        for chars in scene_mentions.values():
            if len(chars) < 2:
                continue
            wc = sorted(chars)
            for i in range(len(wc)):
                for j in range(i + 1, len(wc)):
                    add_pair_weight(edge_evidence, wc[i], wc[j], "same_scene", 1)

    return edge_evidence

def collect_dialogue_sentence_names(doc) -> Counter:
    sentence_name_sets = Counter()
    if not ENABLE_DIALOGUE_INTERACTIONS:
        return sentence_name_sets

    for sent in doc.sents:
        sent_text = sent.text
        if '"' not in sent_text and "“" not in sent_text and "”" not in sent_text:
            continue

        names = set()
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                nm = normalize_name(ent.text)
                if nm:
                    names.add(nm)

        if len(names) >= 2:
            sentence_name_sets[tuple(sorted(names))] += 1

    return sentence_name_sets

def collect_dependency_raw_pairs(doc) -> Counter:
    raw_pairs = Counter()
    if not ENABLE_DEPENDENCY_INTERACTIONS or "parser" not in get_nlp().pipe_names:
        return raw_pairs

    token_name_map: Dict[int, str] = {}
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        nm = normalize_name(ent.text)
        if not nm:
            continue
        for tok in ent:
            token_name_map[tok.i] = nm

    for sent in doc.sents:
        for tok in sent:
            lemma = tok.lemma_.lower()
            if lemma not in RELATION_VERBS:
                continue

            subj_names = set()
            obj_names = set()

            for child in tok.children:
                dep = child.dep_.lower()
                if dep in {"nsubj", "nsubjpass", "csubj"}:
                    subj_names.update(collect_person_names_from_subtree(child, token_name_map))
                elif dep in {"dobj", "pobj", "iobj", "dative", "obj", "attr", "oprd"}:
                    obj_names.update(collect_person_names_from_subtree(child, token_name_map))
                elif dep == "prep":
                    for gc in child.children:
                        if gc.dep_.lower() == "pobj":
                            obj_names.update(collect_person_names_from_subtree(gc, token_name_map))

            if subj_names and obj_names:
                raw_pairs[(tuple(sorted(subj_names)), tuple(sorted(obj_names)))] += 1

    return raw_pairs

def collect_person_names_from_subtree(tok, token_name_map: Dict[int, str]) -> Set[str]:
    found = set()
    for t in tok.subtree:
        if t.i in token_name_map:
            found.add(token_name_map[t.i])
    return found

def compute_chunk_layout(
    text: str,
    para_spans: List[Tuple[int, int]],
    chunks: List[str],
) -> Tuple[List[List[Tuple[int, int]]], List[List[int]]]:
    chunk_para_slices: List[List[Tuple[int, int]]] = []
    chunk_para_indices: List[List[int]] = []
    current_char = 0
    para_cursor = 0

    for chunk in chunks:
        chunk_start = text.find(chunk, current_char)
        if chunk_start < 0:
            chunk_start = current_char
        chunk_end = chunk_start + len(chunk)
        current_char = chunk_end

        local_spans: List[Tuple[int, int]] = []
        local_indices: List[int] = []
        while para_cursor < len(para_spans):
            pstart, pend = para_spans[para_cursor]
            if pstart >= chunk_end:
                break
            if pend <= chunk_start:
                para_cursor += 1
                continue
            local_spans.append((max(0, pstart - chunk_start), min(len(chunk), pend - chunk_start)))
            local_indices.append(para_cursor)
            if pend > chunk_end:
                break
            para_cursor += 1
        chunk_para_slices.append(local_spans)
        chunk_para_indices.append(local_indices)

    return chunk_para_slices, chunk_para_indices

def analyze_corpus(input_dir: Path, output_file: Path) -> None:
    logging.info("Starting corpus analysis...")

    txt_files = get_txt_files(input_dir)
    if not txt_files:
        return

    all_mentions: List[Dict] = []
    mention_counts: Dict[str, int] = {}
    dialogue_sentence_sets: Counter = Counter()
    dependency_raw_pairs: Counter = Counter()
    global_sent_offset = 0
    global_para_offset = 0
    global_scene_offset = 0

    for p in txt_files:
        text = read_text_file(p)
        para_spans = compute_paragraph_spans(text)
        scene_map = compute_scene_map(text, para_spans)
        chunks = chunk_text(text, CHUNK_TARGET_CHARS)
        logging.info(f"{p.name}: split into {len(chunks)} chunk(s).")
        chunk_para_slices, chunk_para_indices = compute_chunk_layout(text, para_spans, chunks)

        docs = get_nlp().pipe(chunks, batch_size=BATCH_SIZE)
        for doc, local_para_spans, local_indices in zip(docs, chunk_para_slices, chunk_para_indices):
            para_index_map = [global_para_offset + g for g in local_indices]
            local_scene_map = {
                i: global_scene_offset + scene_map.get(g, 0) for i, g in enumerate(local_indices)
            }

            mentions = extract_person_mentions(doc, local_para_spans, local_scene_map)
            coref_mentions = get_coref_mentions(doc, mentions, local_para_spans, local_scene_map)

            for m in mentions + coref_mentions:
                m["sentence_global"] = m["sentence"] + global_sent_offset
                m["paragraph_global"] = (
                    para_index_map[m["paragraph"]] if m["paragraph"] < len(para_index_map) else global_para_offset
                )
                m["scene_global"] = m["scene"]
                all_mentions.append(m)

                if not m["is_coref"]:
                    mention_counts[m["name"]] = mention_counts.get(m["name"], 0) + 1

            dialogue_sentence_sets.update(collect_dialogue_sentence_names(doc))
            dependency_raw_pairs.update(collect_dependency_raw_pairs(doc))

            global_sent_offset += sum(1 for _ in doc.sents)

        global_para_offset += len(para_spans)
        global_scene_offset += (max(scene_map.values()) + 1) if scene_map else 1

    if not all_mentions:
        logging.warning("No PERSON entities found in the corpus.")
        return

    mention_counts = {
        k: v for k, v in mention_counts.items()
        if v >= MIN_MENTIONS or (len(k.split()) > 1 and v >= 1)
    }
    if not mention_counts:
        logging.warning(f"All names filtered out by MIN_MENTIONS={MIN_MENTIONS}. Lower the threshold?")
        return

    base_mentions_for_context = [m for m in all_mentions if m["name"] in mention_counts]
    name_contexts = build_name_contexts(base_mentions_for_context)
    alias_to_canon = build_alias_map(mention_counts, name_contexts)
    canonicals = sorted({alias_to_canon[a] for a in mention_counts.keys()})
    logging.info(f"Merged to {len(canonicals)} canonical character(s).")

    aliases: Dict[str, List[str]] = defaultdict(list)
    for name, canon in alias_to_canon.items():
        if name != canon:
            aliases[canon].append(name)
    for names in aliases.values():
        names.sort()

    sentence_mentions: Dict[int, Set[str]] = {}
    paragraph_mentions: Dict[int, Set[str]] = {}
    scene_mentions: Dict[int, Set[str]] = {}

    node_mentions = Counter()
    node_named_mentions = Counter()
    node_coref_mentions = Counter()
    node_paragraphs: Dict[str, Set[int]] = defaultdict(set)
    node_scenes: Dict[str, Set[int]] = defaultdict(set)

    for m in all_mentions:
        raw = m["name"]
        if raw not in mention_counts:
            continue
        canon = alias_to_canon.get(raw, raw)

        sentence_mentions.setdefault(m["sentence_global"], set()).add(canon)
        paragraph_mentions.setdefault(m["paragraph_global"], set()).add(canon)
        scene_mentions.setdefault(m["scene_global"], set()).add(canon)

        node_mentions[canon] += 1
        if m["is_coref"]:
            node_coref_mentions[canon] += 1
        else:
            node_named_mentions[canon] += 1
        node_paragraphs[canon].add(m["paragraph_global"])
        node_scenes[canon].add(m["scene_global"])

    edge_evidence = build_interaction_evidence(sentence_mentions, paragraph_mentions, scene_mentions)

    for names, count in dialogue_sentence_sets.items():
        canon = sorted({alias_to_canon[nm] for nm in names if nm in alias_to_canon})
        for i in range(len(canon)):
            for j in range(i + 1, len(canon)):
                add_pair_weight(edge_evidence, canon[i], canon[j], "dialogue", count)

    for (subj_raw, obj_raw), count in dependency_raw_pairs.items():
        subj = {alias_to_canon[nm] for nm in subj_raw if nm in alias_to_canon}
        obj = {alias_to_canon[nm] for nm in obj_raw if nm in alias_to_canon}
        for a in subj:
            for b in obj:
                if a != b:
                    add_pair_weight(edge_evidence, a, b, "dependency", count)

    for m in all_mentions:
        if not m["is_coref"]:
            continue
        canon = alias_to_canon.get(m["name"], m["name"])
        chars = sentence_mentions.get(m["sentence_global"], set())
        for other in chars:
            if other != canon:
                add_pair_weight(edge_evidence, canon, other, "coref_linked_presence", 1)

    edge_scores: Dict[Tuple[str, str], int] = {}
    for pair, evidence in edge_evidence.items():
        score = 0.0
        for key, amount in evidence.items():
            score += EDGE_WEIGHTS.get(key, 1.0) * amount
        edge_scores[pair] = max(1, round(score))

    nodes = []
    neighbor_map: Dict[str, Set[str]] = defaultdict(set)
    for (a, b), score in edge_scores.items():
        if score > 0:
            neighbor_map[a].add(b)
            neighbor_map[b].add(a)

    edges = []
    for (a, b), score in sorted(edge_scores.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
        evidence = edge_evidence[(a, b)]
        edges.append({
            "source": a,
            "target": b,
            "weight": score,
            "confidence": edge_confidence(evidence),
            "evidence": dict(evidence),
        })

    weighted_degree = Counter()
    for (a, b), score in edge_scores.items():
        weighted_degree[a] += score
        weighted_degree[b] += score

    for c in canonicals:
        nodes.append({
            "id": c,
            "mentions": node_mentions[c],
            "named_mentions": node_named_mentions[c],
            "coref_mentions": node_coref_mentions[c],
            "paragraphs": len(node_paragraphs[c]),
            "scenes": len(node_scenes[c]),
            "unique_neighbors": len(neighbor_map[c]),
            "weighted_degree": weighted_degree[c],
        })

    pipes = get_nlp().pipe_names
    data = {
        "characters": canonicals,
        "aliases": {c: aliases.get(c, []) for c in canonicals},

        "nodes": nodes,
        "edges": edges,

        "settings": {
            "window_sentences": WINDOW_SENTENCES,
            "min_mentions": MIN_MENTIONS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "context_similarity_threshold": CONTEXT_SIMILARITY_THRESHOLD,
            "edge_weights": EDGE_WEIGHTS,
            "coref_enabled": ENABLE_COREF,
            "dependency_enabled": ENABLE_DEPENDENCY_INTERACTIONS and ("parser" in pipes),
            "dialogue_enabled": ENABLE_DIALOGUE_INTERACTIONS,
            "paragraph_edges_enabled": ENABLE_PARAGRAPH_EDGES,
            "scene_edges_enabled": ENABLE_SCENE_EDGES,
            "spacy_model_pipes": pipes,
        },
    }

    try:
        output_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info(f"Interaction data successfully written to {output_file}")
    except Exception as e:
        logging.exception(f"Error writing JSON file {output_file}: {e}")
        raise

def main() -> None:
    global MIN_MENTIONS

    parser = argparse.ArgumentParser(
        description="Analyze character interactions in .txt files and write a JSON network for the D3 frontend."
    )
    parser.add_argument("--input", type=Path, default=Path("input"),
                        help="Directory containing .txt files to analyze (default: input)")
    parser.add_argument("--output", type=Path, default=Path("character_interactions.json"),
                        help="Path of the JSON file to write (default: character_interactions.json)")
    parser.add_argument("--min-mentions", type=int, default=MIN_MENTIONS,
                        help=f"Minimum mentions for a single-word name to be kept (default: {MIN_MENTIONS})")
    args = parser.parse_args()

    MIN_MENTIONS = args.min_mentions
    setup_logging()

    try:
        analyze_corpus(args.input, args.output)
        logging.info("Process completed.")
    except Exception as e:
        logging.error(f"Failed to analyze corpus: {e}")
        raise

if __name__ == "__main__":
    main()