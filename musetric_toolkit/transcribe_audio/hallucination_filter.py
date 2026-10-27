import re

# Caption boilerplate Whisper writes on near-silent or instrumental regions, such
# as subtitle credits; a segment whose text matches any pattern is dropped.

_HALLUCINATION_PATTERNS = [
    # Russian caption credits, e.g. "Субтитры создавал DimaTorzok",
    # "Продолжение следует", or editor credits like
    # "Редактор субтитров А.Семкин Корректор А.Егорова".  # noqa: RUF003
    r"субтитр",
    r"дима\s*торзок",
    r"dimatorzok",
    r"редактор\s+субтитров",
    r"корректор",
    r"продолжение\s+следует",
    r"подписывайтесь",
    r"спасибо\s+за\s+просмотр",
    # English caption credits.
    r"subtitles?\s+by",
    r"thanks?\s+for\s+watching",
    r"please\s+subscribe",
    r"subscribe\s+to",
    # Bare "Thank you" (one or more repeats, any casing/punctuation) emitted as a
    # standalone segment on a silent/fading outro — a classic Whisper caption
    # artifact. Anchored to the WHOLE segment so genuine lyrics that merely
    # contain "thank you" mid-line are left intact; the repeat group also catches
    # "Thank you. Thank you." chains.
    r"^(?:thank\s*you[\s.,!]*)+$",
    # Cross-language caption-platform signatures.
    r"amara\.org",
    r"transcri(?:bed|ption)\s+by",
    r"www\.",
    r"\.com\b",
    # The copyright-transcript signature and a credit of instrumental or silent
    # outros.
    r"©\s*transcript",
    r"emily\s+beynon",
]

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in _HALLUCINATION_PATTERNS]


def _is_hallucination(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in _COMPILED)


def filter_hallucinated_segments(segments: list[dict]) -> list[dict]:
    return [
        segment
        for segment in segments
        if not _is_hallucination(segment.get("text", ""))
    ]


# A whole line that is only a sign-off ("Vielen Dank.", "Gracias.") is a
# caption artifact as well; lyrics that merely contain the word stay.
_SIGN_OFF = re.compile(
    r"^(?:vielen dank|danke(?: schön)?|gracias|muchas gracias|merci(?: beaucoup)?"
    r"|спасибо|thank you|thanks|bye(?: bye)?|tschüss|adiós|au revoir|до свидания)"
    r"[\s.,!]*$",
    re.IGNORECASE,
)


def _is_sign_off(text: str) -> bool:
    return bool(_SIGN_OFF.match(text.strip()))


def filter_sign_off_segments(segments: list[dict]) -> list[dict]:
    return [
        segment for segment in segments if not _is_sign_off(segment.get("text", ""))
    ]


# Caption credits Whisper writes into the same window as the lyrics. They are
# cut out of the text as whole phrases, so the sung words around them stay;
# a segment left with no text is dropped. The browser runtime cuts the same
# phrases out of its words (packages/ai/src/transcription/hallucinationFilter.ts).
_NOT_AFTER_LETTER = r"(?<![^\W\d_])"
_CREDIT_PHRASES = [
    re.compile(_NOT_AFTER_LETTER + pattern, re.IGNORECASE)
    for pattern in (
        r"sous-titr[^\W\d_]*(?: (?:société radio-canada|st' ?501|fr"
        r"|par (?:jérémy diaz|juanfrance)"
        r"|réalisés par la communauté d'amara\.org))?",
        r"société radio-canada",
        r"untertitel(?:ung)? (?:des zdf|im auftrag des zdf(?: für funk)?"
        r"|der amara\.org-community)(?:,? \d{4})?",
        r"subt[ií]tulos? (?:realizados )?por la comunidad de amara\.org",
        r"suscr[ií]bete(?: al canal)?",
        r"gracias por ver el v[ií]deo",
        r"legenda adriana zanotto",
        r"merci d'avoir regardé(?: cette vidéo)?",
        r"(?:la )?iglesia (?:de jesucristo )?(?:de los santos )?(?:de los )?"
        r"últimos días",
    )
]


def _word_end(text: str, index: int) -> int:
    while index < len(text) and not text[index].isspace():
        index += 1
    return index


def strip_credits(text: str) -> str:
    spans = sorted(
        (match.start(), _word_end(text, match.end()))
        for pattern in _CREDIT_PHRASES
        for match in pattern.finditer(text)
    )
    if not spans:
        return text
    kept = []
    position = 0
    for start, end in spans:
        if start > position:
            kept.append(text[position:start])
        position = max(position, end)
    kept.append(text[position:])
    return " ".join("".join(kept).split())


def strip_credit_segments(segments: list[dict]) -> list[dict]:
    stripped = []
    for segment in segments:
        text = strip_credits(segment.get("text", ""))
        if text:
            stripped.append({**segment, "text": text})
    return stripped
