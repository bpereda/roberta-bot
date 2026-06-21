import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import (
    RAG_ENABLED,
    RAG_KNOWLEDGE_DIR,
    RAG_LOG_ENABLED,
    RAG_LOG_PATH,
    RAG_MAX_CHUNKS,
    SENSOR_DATASET_PATH,
)


BASE_DIR = Path(__file__).parent
TOKEN_RE = re.compile(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9_]+")
STOPWORDS = {
    "a",
    "al",
    "como",
    "con",
    "cuando",
    "de",
    "del",
    "el",
    "en",
    "es",
    "esta",
    "este",
    "la",
    "las",
    "lo",
    "los",
    "me",
    "mi",
    "mis",
    "no",
    "o",
    "para",
    "por",
    "que",
    "se",
    "si",
    "su",
    "sus",
    "un",
    "una",
    "y",
}

DOMAIN_SYNONYMS = {
    "agua": {"humedad", "regar", "riego", "suelo", "sustrato"},
    "condicion": {"estado", "estres", "marchita", "salud", "saludable"},
    "cuidado": {"cuidados", "humedad", "luz", "riego", "temperatura"},
    "cuidados": {"cuidado", "humedad", "luz", "riego", "temperatura"},
    "estado": {"condicion", "estres", "marchita", "salud", "saludable"},
    "especie": {"aphelandra", "familia", "squarrosa", "taxonomia", "tipo"},
    "luz": {"iluminacion", "indirecta", "sol"},
    "marchita": {"agua", "condicion", "estado", "estres", "humedad", "riego"},
    "regar": {"agua", "humedad", "riego", "suelo", "sustrato"},
    "riego": {"agua", "humedad", "regar", "suelo", "sustrato"},
    "seca": {"agua", "marchita", "riego", "seco", "sequedad"},
    "seco": {"agua", "marchita", "riego", "seca", "sequedad"},
    "taxonomia": {"especie", "familia", "genero", "gbif", "ipni", "kew"},
    "temperatura": {"ambiente", "calor", "frio"},
    "tipo": {"aphelandra", "especie", "familia", "squarrosa", "taxonomia"},
}

SENSOR_INTENT_TOKENS = {
    "agua",
    "condicion",
    "estado",
    "humedad",
    "luz",
    "marchita",
    "regar",
    "riego",
    "salud",
    "sensores",
    "temperatura",
}

TAXONOMY_INTENT_TOKENS = {
    "aphelandra",
    "especie",
    "familia",
    "gbif",
    "genero",
    "ipni",
    "kew",
    "origen",
    "squarrosa",
    "taxonomia",
    "tipo",
}


@dataclass
class Chunk:
    source: str
    text: str
    tokens: set[str]
    token_counts: Counter
    kind: str
    recency_rank: int = 0


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float
    overlap_tokens: list[str]
    kind: str
    score_details: dict[str, float]


@dataclass
class RagResult:
    query: str
    query_tokens: list[str]
    total_chunks: int
    selected_chunks: list[RetrievedChunk]
    context: str
    enabled: bool
    intent: str


def get_configured_path(value: str) -> Path:
    configured_path = Path(value).expanduser()

    if not configured_path.is_absolute():
        configured_path = BASE_DIR / configured_path

    return configured_path


def tokenize_to_list(text: str) -> list[str]:
    words = TOKEN_RE.findall(text.lower())
    return [word for word in words if len(word) > 2 and word not in STOPWORDS]


def tokenize(text: str) -> set[str]:
    return set(tokenize_to_list(text))


def expand_query_tokens(tokens: set[str]) -> set[str]:
    expanded_tokens = set(tokens)

    for token in tokens:
        expanded_tokens.update(DOMAIN_SYNONYMS.get(token, set()))

    return expanded_tokens


def split_text(text: str, max_words: int = 110) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks = []

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) <= max_words:
            chunks.append(paragraph)
            continue

        for start in range(0, len(words), max_words):
            chunks.append(" ".join(words[start : start + max_words]))

    return chunks


def make_chunk(source: str, text: str, kind: str, recency_rank: int = 0) -> Chunk:
    tokens = tokenize(text)
    return Chunk(
        source=source,
        text=text,
        tokens=tokens,
        token_counts=Counter(tokenize_to_list(text)),
        kind=kind,
        recency_rank=recency_rank,
    )


def load_knowledge_chunks() -> list[Chunk]:
    knowledge_dir = get_configured_path(RAG_KNOWLEDGE_DIR)

    if not knowledge_dir.exists():
        return []

    chunks = []
    paths = sorted(knowledge_dir.glob("*.md")) + sorted(knowledge_dir.glob("*.txt"))
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue

        for index, chunk_text in enumerate(split_text(text), start=1):
            chunks.append(make_chunk(f"{path.name}#{index}", chunk_text, "knowledge"))

    return chunks


def load_dataset_chunks(max_rows: int = 80) -> list[Chunk]:
    dataset_path = get_configured_path(SENSOR_DATASET_PATH)

    if not dataset_path.exists():
        return []

    try:
        with dataset_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)
    except (OSError, csv.Error):
        return []

    if not rows:
        return []

    chunks = []
    recent_rows = rows[-max_rows:]
    first_row_number = max(1, len(rows) - len(recent_rows) + 1)
    for offset, row in enumerate(recent_rows):
        row_number = first_row_number + offset
        fields = []
        for key, value in row.items():
            if value not in ("", None):
                fields.append(f"{key}={value}")

        text = "Medición histórica de Roberta: " + ", ".join(fields)
        chunks.append(
            make_chunk(
                f"sensores.csv#fila-{row_number}",
                text,
                "sensor",
                recency_rank=offset + 1,
            )
        )

    return chunks


def build_inverse_document_frequency(chunks: list[Chunk]) -> dict[str, float]:
    document_count = len(chunks)
    document_frequency = Counter()

    for chunk in chunks:
        document_frequency.update(chunk.tokens)

    return {
        token: math.log((document_count + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }


def detect_intent(query_tokens: set[str]) -> str:
    if query_tokens & TAXONOMY_INTENT_TOKENS:
        return "taxonomy"

    if query_tokens & SENSOR_INTENT_TOKENS:
        return "sensor"

    return "general"


def score_chunk(
    query_tokens: set[str],
    chunk: Chunk,
    idf: dict[str, float],
    intent: str,
) -> tuple[float, dict[str, float]]:
    if not query_tokens or not chunk.tokens:
        return 0.0, {"tfidf": 0.0, "intent_boost": 1.0, "recency_boost": 1.0}

    overlap_tokens = query_tokens & chunk.tokens
    if not overlap_tokens:
        return 0.0, {"tfidf": 0.0, "intent_boost": 1.0, "recency_boost": 1.0}

    chunk_length = sum(chunk.token_counts.values()) or 1
    tfidf_score = 0.0
    for token in overlap_tokens:
        term_frequency = chunk.token_counts[token] / chunk_length
        tfidf_score += term_frequency * idf.get(token, 1.0)

    tfidf_score *= len(overlap_tokens)

    intent_boost = 1.0
    if intent == "sensor" and chunk.kind == "sensor":
        intent_boost = 1.25
    elif intent == "taxonomy" and "taxonomia" in chunk.source:
        intent_boost = 1.35
    elif intent == "taxonomy" and chunk.kind == "sensor":
        intent_boost = 0.55

    recency_boost = 1.0
    if chunk.kind == "sensor":
        recency_boost = 1.0 + min(chunk.recency_rank, 80) / 1000

    return tfidf_score * intent_boost * recency_boost, {
        "tfidf": tfidf_score,
        "intent_boost": intent_boost,
        "recency_boost": recency_boost,
    }


def should_include_fallback_chunk(query_tokens: set[str], chunk: Chunk, intent: str) -> bool:
    if intent == "taxonomy":
        return chunk.kind == "knowledge" and "taxonomia" in chunk.source

    if intent == "sensor":
        return chunk.kind == "sensor" and bool(query_tokens & SENSOR_INTENT_TOKENS)

    return False


def select_ranked_chunks(
    ranked_chunks: list[tuple[tuple[float, dict[str, float]], list[str], Chunk]],
    query_tokens: set[str],
    intent: str,
    max_chunks: int,
) -> list[RetrievedChunk]:
    candidates = [
        RetrievedChunk(
            source=chunk.source,
            text=chunk.text,
            score=score,
            overlap_tokens=overlap_tokens,
            kind=chunk.kind,
            score_details=score_details,
        )
        for (score, score_details), overlap_tokens, chunk in ranked_chunks
        if score > 0 or should_include_fallback_chunk(query_tokens, chunk, intent)
    ]

    selected_chunks = candidates[:max_chunks]
    selected_sources = {chunk.source for chunk in selected_chunks}

    if intent == "sensor":
        best_sensor_chunk = next(
            (chunk for chunk in candidates if chunk.kind == "sensor"),
            None,
        )
        has_sensor_near_top = any(chunk.kind == "sensor" for chunk in selected_chunks[:3])
        if best_sensor_chunk and not has_sensor_near_top:
            insert_at = min(2, len(selected_chunks))
            selected_chunks.insert(insert_at, best_sensor_chunk)

    if intent == "taxonomy":
        best_taxonomy_chunk = next(
            (chunk for chunk in candidates if "taxonomia" in chunk.source),
            None,
        )
        has_taxonomy_near_top = any(
            "taxonomia" in chunk.source for chunk in selected_chunks[:3]
        )
        if best_taxonomy_chunk and not has_taxonomy_near_top:
            selected_chunks.insert(0, best_taxonomy_chunk)

    deduped_chunks = []
    seen_sources = set()
    for chunk in selected_chunks:
        if chunk.source in seen_sources:
            continue

        deduped_chunks.append(chunk)
        seen_sources.add(chunk.source)

        if len(deduped_chunks) >= max_chunks:
            break

    if len(deduped_chunks) < max_chunks:
        for chunk in candidates:
            if chunk.source in seen_sources:
                continue

            deduped_chunks.append(chunk)
            seen_sources.add(chunk.source)

            if len(deduped_chunks) >= max_chunks:
                break

    return deduped_chunks


def build_rag_result(user_text: str, max_chunks: Optional[int] = None) -> RagResult:
    raw_query_tokens = tokenize(user_text)
    query_tokens = expand_query_tokens(raw_query_tokens)
    intent = detect_intent(query_tokens)

    if not RAG_ENABLED:
        return RagResult(
            query=user_text,
            query_tokens=sorted(query_tokens),
            total_chunks=0,
            selected_chunks=[],
            context="RAG desactivado por configuración.",
            enabled=False,
            intent=intent,
        )

    if max_chunks is None:
        max_chunks = RAG_MAX_CHUNKS

    chunks = load_knowledge_chunks() + load_dataset_chunks()
    idf = build_inverse_document_frequency(chunks)
    ranked_chunks = sorted(
        (
            (
                score_chunk(query_tokens, chunk, idf, intent),
                sorted(query_tokens & chunk.tokens),
                chunk,
            )
            for chunk in chunks
        ),
        key=lambda item: item[0][0],
        reverse=True,
    )

    selected_chunks = select_ranked_chunks(ranked_chunks, query_tokens, intent, max_chunks)

    if not selected_chunks:
        context = "No se recuperaron fragmentos relevantes para esta pregunta."
    else:
        context = "\n\n".join(
            f"[Fuente: {chunk.source} | tipo={chunk.kind} | score={chunk.score:.3f}]\n{chunk.text}"
            for chunk in selected_chunks
        )

    return RagResult(
        query=user_text,
        query_tokens=sorted(query_tokens),
        total_chunks=len(chunks),
        selected_chunks=selected_chunks,
        context=context,
        enabled=True,
        intent=intent,
    )


def get_rag_log_path() -> Path:
    log_path = get_configured_path(RAG_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def write_rag_log(result: RagResult) -> None:
    if not RAG_LOG_ENABLED:
        return

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": result.query,
        "query_tokens": result.query_tokens,
        "intent": result.intent,
        "enabled": result.enabled,
        "total_chunks": result.total_chunks,
        "selected_chunks": [
            {
                "source": chunk.source,
                "kind": chunk.kind,
                "score": round(chunk.score, 4),
                "score_details": {
                    key: round(value, 4)
                    for key, value in chunk.score_details.items()
                },
                "overlap_tokens": chunk.overlap_tokens,
                "preview": chunk.text[:220],
            }
            for chunk in result.selected_chunks
        ],
    }

    try:
        with get_rag_log_path().open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        return


def format_rag_debug(result: RagResult) -> str:
    if not result.enabled:
        return result.context

    lines = [
        f"Consulta: {result.query}",
        f"Intención detectada: {result.intent}",
        f"Tokens expandidos: {', '.join(result.query_tokens) or 'sin tokens'}",
        f"Chunks evaluados: {result.total_chunks}",
        f"Chunks seleccionados: {len(result.selected_chunks)}",
    ]

    for index, chunk in enumerate(result.selected_chunks, start=1):
        preview = chunk.text.replace("\n", " ")[:280]
        details = ", ".join(
            f"{key}={value:.3f}" for key, value in chunk.score_details.items()
        )
        lines.append(
            f"\n{index}. {chunk.source} | tipo={chunk.kind} | score={chunk.score:.3f} | {details} | match={', '.join(chunk.overlap_tokens)}\n{preview}"
        )

    if not result.selected_chunks:
        lines.append("\nNo se recuperaron fragmentos relevantes.")

    return "\n".join(lines)


def retrieve_rag_context(user_text: str, max_chunks: Optional[int] = None) -> str:
    result = build_rag_result(user_text, max_chunks=max_chunks)
    write_rag_log(result)
    return result.context
