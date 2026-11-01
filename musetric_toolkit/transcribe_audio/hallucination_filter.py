import re

_HALLUCINATION_PATTERNS = [
    r"субтитр",
    r"дима\s*торзок",
    r"dimatorzok",
    r"редактор\s+субтитров",
    r"корректор",
    r"продолжение\s+следует",
    r"подписывайтесь",
    r"спасибо\s+за\s+просмотр",
    r"subtitles?\s+by",
    r"thanks?\s+for\s+watching",
    r"please\s+subscribe",
    r"subscribe\s+to",
    r"^(?:thank\s*you[\s.,!]*)+$",
    r"amara\.org",
    r"transcri(?:bed|ption)\s+by",
    r"www\.",
    r"\.com\b",
    r"©\s*transcript",
    r"emily\s+beynon",
]

_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in _HALLUCINATION_PATTERNS]


def is_hallucination(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in _COMPILED)


def filter_hallucinated_segments(segments: list[dict]) -> list[dict]:
    return [segment for segment in segments if not is_hallucination(segment["text"])]


_SIGN_OFF = re.compile(
    r"^(?:vielen dank|danke(?: schön)?|gracias|muchas gracias|merci(?: beaucoup)?"
    r"|спасибо|thank you|thanks|bye(?: bye)?|tschüss|adiós|au revoir|до свидания)"
    r"[\s.,!]*$",
    re.IGNORECASE,
)


def _is_sign_off(text: str) -> bool:
    return bool(_SIGN_OFF.match(text.strip()))


def filter_sign_off_segments(segments: list[dict]) -> list[dict]:
    return [segment for segment in segments if not _is_sign_off(segment["text"])]


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


def drop_credit_words(words: list[dict]) -> list[dict]:
    starts = []
    text = ""
    for word in words:
        text += " " if text else ""
        starts.append(len(text))
        text += word["text"]
    ranges = [
        match.span() for pattern in _CREDIT_PHRASES for match in pattern.finditer(text)
    ]
    if not ranges:
        return words
    return [
        word
        for word, start in zip(words, starts, strict=True)
        if not any(low <= start < high for low, high in ranges)
    ]
