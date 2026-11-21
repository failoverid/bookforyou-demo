"""Community-tag curation (4 stages, same order as the paper) and text cleaning."""
import html
import re

STATUS_TAGS = {"to-read", "currently-reading", "read", "kindle", "favorites", "owned", "default"}
# Extra status words for raw Goodreads shelves: any shelf with one of these as a hyphen-separated word is dropped.
STATUS_WORDS = re.compile(
    r"(^|-)(read|reading|reads|tbr|own|owned|kindle|ebook|ebooks|audiobook|audiobooks|audio|library|"
    r"favorite|favorites|favourite|favourites|favs|wishlist|wish|dnf|arc|netgalley|buy|purchased|bought|"
    r"borrowed|paperback|hardcover|nook|audible|shelf|book|books|default|unfinished|abandoned|have|maybe|reviewed|finish|finished|did|not|series|standalone|stand|alone|novel|novels|first|part|next|unread|rated|rating|stars|star|wanna|want|bookshelf|collection|kobo|goodreads|publisher|paper|print|calibre|freebie|freebies|free|freeby|review|amazon|signed|giveaway|overdrive|meh|lendable|edelweiss|hold|listened|fave|lost|interest|recommended|need|english|cover|female|author|anthology|novella|sequel|trilogy|short|youth|kid|kids|children|childhood|juvenile|classic|literature|cry|opinion|loved|liked|disliked|hated|bad|good|great|ya-lit|galley|netgalley|net|epub|ibook|mine|get|downloaded|gave|release|released|reread|giveaway|ebooks|pdf|mobi|kindle|nook|tbr|wtr|ar|sub)(-|$)")
GENRE_WORDS = {"mystery", "thriller", "romance", "fantasy", "horror", "suspense", "crime", "paranormal",
               "contemporary", "urban", "adventure", "comedy", "humor", "drama", "action", "dystopian"}
KEEP_COMPOUNDS = {"sci-fi"}  # compounds that must never be split
KEEP_PLURAL = {"enemies to lovers", "friends to lovers"}  # fixed phrases, not singularized
# Synonym merge applied after stage 4 (an addition to the paper's 4 stages: without it "ya", "young adult"
# and "teen" count as separate shared tags and inflate relevance).
ALIASES = {
    "ya": "young adult", "youngadult": "young adult", "ya fiction": "young adult", "young adult fiction": "young adult",
    "teen": "young adult", "teen fiction": "young adult", "ya lit": "young adult", "ya contemporary": "young adult contemporary",
    "sci fi": "science fiction", "scifi": "science fiction", "sf": "science fiction", "speculative fiction": "science fiction",
    "sci fi fantasy": "science fiction fantasy", "scifi fantasy": "science fiction fantasy", "fantasy sci fi": "science fiction fantasy",
    "fantasy scifi": "science fiction fantasy", "sff": "science fiction fantasy",
    "mm": "mm romance", "m m": "mm romance", "m m romance": "mm romance",
    "lgbtq": "lgbt", "glbt": "lgbt", "gay": "lgbt",
    "pnr": "paranormal romance", "humour": "humor", "funny": "humor", "comedy": "humor",
    "dystopian": "dystopia", "vamp": "vampire", "shapeshifter": "shifter", "contemporary fiction": "contemporary",
    "adult fiction": "adult", "general fiction": "fiction", "crime fiction": "crime", "erotic": "erotica",
    "erotic romance": "erotica", "ya romance": "romance", "ya fantasy": "fantasy", "ya paranormal": "paranormal",
    "young adult fantasy": "fantasy", "cozy mystery": "cozy", "murder mystery": "murder", "mystery thriller": "mystery",
}
NO_SINGULAR = {"series", "news", "politics", "species", "mathematics", "physics", "economics"}


def clean_text(text):
    """Description: strip HTML, decode entities, collapse whitespace (case is kept for the models)."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def clean_synopsis(text):
    """Demo preprocessing: strip HTML/symbols and lowercase."""
    text = re.sub(r"[^a-z0-9\s'-]", " ", clean_text(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def stage1_status(tags, status=STATUS_TAGS):
    out = []
    for t in tags:
        t = t.lower()
        if t in status or STATUS_WORDS.search(t) or STATUS_WORDS.search(re.sub(r"s(?=-|$)", "", t)) or re.search(r"\d", t):
            continue
        out.append(t)
    return out


def singular_word(w):
    if w in NO_SINGULAR:
        return w
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ves"):
        return w[:-3] + "f"
    if re.search(r"(ss|us|is)$", w):
        return w
    if re.search(r"(ch|sh|x|z)es$", w):
        return w[:-2]
    return w[:-1] if w.endswith("s") else w


def stage3_singular(tags, keep_plural=KEEP_PLURAL):
    out = []
    for t in tags:
        keep = re.sub(r"[^a-z]+", " ", t).strip() in keep_plural
        out.append(t if keep else re.sub(r"[a-z]+$", lambda m: singular_word(m.group()), t))
    return out


def stage4_chars(tags):
    return [re.sub(r"[^a-z0-9]+", " ", t).strip() for t in tags]


def stage2_split(tags):
    """Split compounds made only of genre words (mystery-thriller). Anything else is a legitimate
    compound and stays (science-fiction, young-adult, enemies-to-lovers, sci-fi)."""
    out = []
    for t in tags:
        parts = t.split("-")
        split = len(parts) > 1 and t not in KEEP_COMPOUNDS and all(p in GENRE_WORDS for p in parts)
        out += parts if split else [t]
    return out


def curate(raw, vocab=None, status=STATUS_TAGS, keep_plural=KEEP_PLURAL):
    """Return (after stage 1, 2, 3, 4, final). final is filtered to vocab when one is given."""
    s1 = stage1_status(raw, status)
    s2 = stage2_split(s1)
    s3 = stage3_singular(s2, keep_plural)
    s4 = [ALIASES.get(t, t) for t in stage4_chars(s3) if t]
    final = sorted({t for t in s4 if vocab is None or t in vocab})
    return s1, s2, s3, s4, final
