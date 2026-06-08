import re
from functools import lru_cache
from typing import Iterable

from kiwipiepy import Kiwi


BM25_ALLOWED_POS_PREFIXES = ("N",)
BM25_ALLOWED_POS_TAGS = {"SL", "SN", "XR"}
BM25_STOPWORDS = {
    "것",
    "수",
    "등",
    "다음",
    "아래",
    "보기",
    "알맞",
    "작성",
    "확인",
    "문제",
}
CODE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


@lru_cache(maxsize=1)
def get_kiwi() -> Kiwi:
    return Kiwi()


def tokenize_for_bm25(text: str) -> list[str]:
    """BM25 검색에 사용할 토큰을 만듭니다.

    Kiwi로 한국어 조사/어미를 제거하고, 코드/SQL 식별자는 정규식으로 한 번 더 보강합니다.
    """
    if not text:
        return []

    tokens = []
    for token in _kiwi_tokens(text):
        tokens.append(token)

    for token in _code_identifier_tokens(text):
        tokens.append(token)

    tokens.extend(_code_phrase_tokens(text))

    return tokens


def build_bm25_document_text(
    *,
    title: str,
    chapter: str = "",
    content: str = "",
    title_weight: int = 3,
) -> str:
    weighted_title = " ".join([title] * max(title_weight, 1))
    return " ".join(part for part in [weighted_title, chapter, content] if part)


def _kiwi_tokens(text: str) -> Iterable[str]:
    for token in get_kiwi().tokenize(text):
        if not _is_allowed_pos(token.tag):
            continue

        normalized = _normalize_token(token.form)
        if _is_meaningful_token(normalized):
            yield normalized


def _code_identifier_tokens(text: str) -> Iterable[str]:
    for match in CODE_IDENTIFIER_PATTERN.finditer(text):
        normalized = _normalize_token(match.group(0))
        if _is_meaningful_token(normalized):
            yield normalized


def _code_phrase_tokens(text: str) -> list[str]:
    phrase_tokens = []
    matches = list(CODE_IDENTIFIER_PATTERN.finditer(text))
    for left, right in zip(matches, matches[1:]):
        between = text[left.end() : right.start()]
        if not _is_phrase_gap(between):
            continue

        left_token = _normalize_token(left.group(0))
        right_token = _normalize_token(right.group(0))
        if _is_phrase_part(left_token) and _is_phrase_part(right_token):
            phrase_tokens.append(f"{left_token}_{right_token}")

    return phrase_tokens


def _is_allowed_pos(tag: str) -> bool:
    return tag.startswith(BM25_ALLOWED_POS_PREFIXES) or tag in BM25_ALLOWED_POS_TAGS


def _normalize_token(token: str) -> str:
    return token.strip().lower()


def _is_meaningful_token(token: str) -> bool:
    if not token or token in BM25_STOPWORDS:
        return False

    if len(token) == 1 and not token.isalnum():
        return False

    return True


def _is_phrase_part(token: str) -> bool:
    return bool(re.fullmatch(r"[a-z_][a-z0-9_]*", token))


def _is_phrase_gap(text: str) -> bool:
    return bool(re.fullmatch(r"[\s.,;:()\[\]{}<>+\-*/=|&!?'\"]*", text))
