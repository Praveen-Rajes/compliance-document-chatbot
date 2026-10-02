"""
Query repair: spell correction, quality gating, and follow-up expansion.

Everything here rewrites the string used for *retrieval only*. The user's own
wording is what reaches the model and what is shown back to them -- silently
"correcting" a compliance professional's question in the transcript would be
worse than the typo.

Three problems, all seen in production testing:

1. "give me a brief knhowlge what you know" retrieved nothing useful, because
   the embedding was computed on the misspelling. A vocabulary built from the
   indexed corpus fixes this without any dictionary download, which matters on
   an air-gapped server.

2. "tell me more about it" embedded the literal phrase, which is near-orthogonal
   to every document. Conversation history reached the model but never reached
   the retriever, so the model was handed irrelevant sources and correctly said
   it did not know.

3. "s", "ssss" and "sasass" matched real documents at 56-70% and produced
   confident, cited answers. In a bank that is worse than refusing.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter

# Words carrying no retrieval signal. Deliberately short: dropping domain words
# would hurt more than the noise they remove.
_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "do", "does", "did", "what", "which", "who",
    "whom", "how", "when", "where", "why", "can", "could", "would", "should",
    "shall", "will", "may", "might", "must", "i", "me", "my", "we", "our", "you",
    "your", "it", "its", "this", "that", "these", "those", "there", "here",
    "please", "tell", "give", "show", "about", "more", "some", "any", "all",
    "with", "from", "by", "as", "at", "into", "over", "under", "between",
    "brief", "briefly", "summary", "know", "knowledge", "explain", "describe",
}

# Anaphora: a short question containing one of these depends on the previous
# turn to mean anything.
_ANAPHORA = re.compile(
    r"\b(it|its|this|that|these|those|them|they|their|there|same|above"
    r"|more|else|further|other|another|elaborate|expand|continue)\b", re.I)

# Explicit references to the conversation itself. These are unambiguous
# follow-up markers regardless of how long the sentence is -- "that answer is
# too long, please give a more concise answer for the above question I just
# asked" is 18 words and could not be more clearly a follow-up.
_META_REF = re.compile(
    r"\b(the\s+)?(above|previous|last|earlier|former|prior)\s+"
    r"(question|answer|response|reply|point|message|thing|one)"
    r"|\b(that|this|your)\s+(answer|response|reply|summary|explanation)"
    r"|\bi\s+(just\s+)?asked\b"
    r"|\b(shorten|rephrase|rewrite|redo|repeat|condense|simplify)\b"
    r"|\bsame\s+(question|thing|topic)\b"
    r"|\bwhat\s+you\s+(just\s+)?(said|wrote|gave)\b"
    # "I didn't understand", "explain it more clearly", "in simpler terms".
    # These say nothing about a subject, so they can only be about the previous
    # turn -- but they are wordy enough that counting subject words alone
    # misjudged them.
    r"|\bi\s+(did\s*n[o\u2019']?t|didn?t|don'?t|do\s+not|cannot|can'?t)\s+"
    r"(really\s+|quite\s+|fully\s+)?(understand|understood|get|follow|see)\b"
    r"|\bexplain\s+(it\s+|this\s+|that\s+)?(more\s+|again\s+)?"
    r"(clearly|simply|better|properly|further|again)\b"
    r"|\bin\s+(simpler|simple|plain|easier|layman'?s?)\s+(terms|words|english|language)\b"
    r"|\bclarify\b|\bnot\s+(very\s+)?clear\b|\bconfus(ed|ing)\b"
    r"|\bbreak\s+(it|this|that)\s+down\b"
    r"|\bmore\s+(detail|details|clearly|simply)\b",
    re.I,
)

# Words about the conversation rather than about compliance. They are excluded
# when judging whether a question introduces a NEW subject, because "concise",
# "answer" and "points" say nothing about which document is wanted.
#: Ways of asking for the same answer with less of it. Matching one of these is
#: necessary but not sufficient -- see wants_brevity, which also requires that
#: the sentence names no new subject.
_BREVITY_RE = re.compile(
    r"\b(shorter|shorten|condense[sd]?|concise|briefer|summaris\w*|summariz\w*"
    r"|summary|tl\s*[;:]?\s*dr|key\s+points?|main\s+points?|bullet\s+points?"
    r"|in\s+(one|a)\s+(line|sentence|paragraph)|one[\s-]?liner"
    r"|short(er)?\s+version|brief(ly)?|in\s+(short|brief)|cut\s+(it\s+)?down"
    r"|too\s+long|less\s+detail|not\s+so\s+long|make\s+it\s+small\w*)\b", re.I)

#: Words that only ever describe the shape of the answer, never its subject.
_BREVITY_WORDS = frozenset("""
    short shorter shorten shortest brief briefly briefest summary summarise
    summarize summarised summarized concise condense condensed key main points
    point line lines sentence sentences version bullet bullets tldr less detail
    details long longer cut down small smaller quick quickly gist
""".split())

_META_WORDS = {
    "answer", "answers", "question", "questions", "response", "reply", "asked",
    "said", "told", "wrote", "gave", "above", "previous", "last", "earlier",
    "concise", "shorter", "longer", "short", "long", "detail", "details",
    "detailed", "deeply", "deep", "point", "points", "form", "bullet",
    "bullets", "summarise", "summarize", "elaborate", "expand", "rephrase",
    "rewrite", "shorten", "condense", "simplify", "again", "too", "much",
    "need", "want", "keep", "three", "four", "five", "few", "one", "make",
    "thing", "things", "way", "bit", "little", "lot", "just", "only", "clear",
    "clearly", "specific", "properly", "nicely", "better",
    # Words people use when the previous answer did not land. None of them names
    # a subject, so leaving them out was making wordy requests for clarification
    # look like fresh questions.
    "understand", "understood", "understanding", "didnt", "didn", "dont", "don",
    "cant", "couldnt", "wasnt", "isnt", "very", "well", "really", "quite",
    "sorry", "confused", "confusing", "unclear", "simple", "simpler", "simply",
    "easier", "easy", "kindly", "okay", "sure", "still", "actually", "mean",
    "meant", "saying", "told", "follow", "see", "know", "help", "give", "get",
}

_GREETING = re.compile(
    r"^\s*(hi|hii+|hey+|hello+|yo|sup|greetings"
    r"|good\s*(morning|afternoon|evening|day)"
    r"|thanks?|thank\s*you|thx|ty|cheers|bye|goodbye|see\s*you|ok(ay)?|cool|nice)"
    r"[\s!,.?]*(bro|bruh|man|mate|sir|madam|there|buddy)?[\s!,.?]*$", re.I)

# Ordinary English that must never be "corrected". A compliance corpus is not a
# dictionary: words like "tell", "about" and "more" may be rare in it, and the
# nearest corpus word to "tell" is something like "cell" or "till". Rewriting a
# question word silently changes what the user asked and can flip its routing --
# "tell me about the deposit insurance scheme" became unroutable this way.
# Correction is for domain vocabulary, so plain English is held back from it.
_PROTECTED = {
    "tell", "give", "show", "list", "find", "help", "make", "take", "know",
    "about", "above", "below", "under", "over", "with", "from", "into", "onto",
    "what", "when", "where", "which", "whom", "whose", "why", "how", "who",
    "this", "that", "these", "those", "there", "their", "them", "they", "then",
    "have", "has", "had", "does", "did", "done", "will", "would", "shall",
    "should", "could", "must", "may", "might", "can", "need", "want", "like",
    "more", "most", "much", "many", "some", "any", "all", "each", "every",
    "also", "just", "only", "very", "such", "same", "other", "another", "both",
    "please", "thanks", "thank", "hello", "hey", "good", "morning", "evening",
    "afternoon", "your", "yours", "yourself", "mine", "ours", "here", "been",
    "being", "were", "was", "are", "is", "am", "be", "brief", "briefly",
    "summary", "summarise", "summarize", "overview", "explain", "describe",
    "question", "answer", "information", "detail", "details", "example",
}

_TOKEN = re.compile(r"[a-z0-9]+")
_REPEAT = re.compile(r"(.)\1{3,}")          # 4+ of the same character in a row

GREETING_REPLY = (
    "Hello. I'm Amana Bank's Compliance Knowledge Assistant. Ask me about a "
    "regulation, direction, circular or policy and I'll answer from the bank's "
    "document library with citations."
)

TOO_SHORT_REPLY = (
    "I couldn't make out a question there. Could you write it out in full? "
    "For example: \"What is the maximum loan to value ratio for motor vehicles?\""
)



# ── Offensive language ────────────────────────────────────────────────────────
#
# Detected here rather than left to the model. Asking the model to refuse works
# most of the time, which is the problem: "most of the time" is not a policy. A
# server-side check fires on every request, costs no GPU, and can be shown to an
# auditor as a rule rather than a tendency.
#
# Matching is on whole words only, never substrings. A compliance corpus is full
# of words that contain rude ones -- assessment, assets, class, pass, analysis,
# Scunthorpe -- and a substring match would refuse perfectly ordinary questions.
# The patterns below are checked against the whole corpus vocabulary in
# selftest.py, and any hit there is a false positive by definition.
#
# Obfuscation is covered because people type around filters: f*ck, sh1t, b!tch.
# Digits and symbols commonly substituted for letters are folded first.
_LEET = str.maketrans({"@": "a", "4": "a", "3": "e", "1": "i", "!": "i",
                       "0": "o", "$": "s", "5": "s", "7": "t", "+": "t"})

# Strong terms. Always refused.
_PROFANITY_STRONG = [
    r"f+\s*[u\*]+\s*c+\s*k+", r"\bf+u+k+\b", r"\bph[u\*]+ck",
    r"\bs+h+[i\*]+t+\b", r"\bbull\s*sh[i\*]+t",
    r"\bb[i\*]+t+c+h", r"\bc+u+n+t+\b",
    r"\ba+s+s+h+o+l+e", r"\bar+se+\s*hole", r"\ba+s+s+w+i+p+e",
    r"\bd[i\*]+c+k+\s*head", r"\bp[u\*]+ss+y\b",
    r"\bb[a\*]+st[a\*]+rd", r"\bw[a\*]+nk", r"\bbollock",
    r"\bmother\s*f", r"\bmotherf", r"\bslut\b", r"\bwhore\b",
    r"\btwat\b", r"\bprick\b", r"\bshag\b", r"\bcock\s*sucker",
]

# Mild terms. Refused too, because the policy chosen is to refuse and ask for a
# rephrasing, and "what the hell is the LTV ratio" is the exact case that
# prompted it. Set PROFANITY_MILD=false to allow these through.
# Note the word boundaries: \bhell\b does not match "shell bank" or "hello",
# both of which appear constantly in this corpus.
_PROFANITY_MILD = [
    r"\bhell\b", r"\bdamn\b", r"\bgod\s*damn", r"\bbloody\s+hell",
    r"\bcrap\b", r"\bpiss(ed|ing)?\b", r"\bbugger\b", r"\bfrigging\b",
    r"\bfreaking\b", r"\bsucks\b", r"\bidiot\b", r"\bstupid\b",
    r"\bmoron\b", r"\bdumb\b",
]

PROFANITY_REPLY = (
    "I can't respond to a message containing that language. Please rephrase your "
    "question professionally and I'll answer it."
)


def _compile_profanity(include_mild: bool):
    pats = list(_PROFANITY_STRONG) + (list(_PROFANITY_MILD) if include_mild else [])
    return re.compile("|".join(pats), re.I)


_PROFANITY_RE = _compile_profanity(True)


def set_profanity_level(include_mild: bool):
    """Rebuild the matcher. Called once at startup from config."""
    global _PROFANITY_RE
    _PROFANITY_RE = _compile_profanity(include_mild)


def is_offensive(question: str) -> bool:
    """
    True if the message contains offensive language.

    Leet substitutions are folded first so f*ck and sh1t are caught, and
    separator characters between letters are collapsed so f.u.c.k is too.
    """
    q = question.lower().translate(_LEET)
    if _PROFANITY_RE.search(q):
        return True
    # collapse punctuation used to space out letters: f.u.c.k, s-h-i-t
    squeezed = re.sub(r"[^a-z]+", "", q)
    return bool(re.search(r"fuck|shit|bitch|cunt|asshole|bastard", squeezed))


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class QueryFixer:
    """Corpus-derived spelling and query quality checks."""

    #: Words shorter than this are never corrected. Acronyms (LCR, AML, KYC,
    #: SLFRS) live here and must survive untouched.
    MIN_CORRECT_LEN = 4
    #: difflib similarity a candidate must reach to replace a word.
    CUTOFF = 0.82
    #: Cap on candidates compared per token, keeping correction ~1 ms.
    BUCKET_CAP = 900

    #: Word-level memo. A difflib scan over a 900-word bucket costs a couple of
    #: milliseconds, and it is paid for every word that is not in the vocabulary
    #: -- even when nothing gets corrected. Across 50 concurrent users the same
    #: handful of words recur constantly, so remembering the verdict per word
    #: takes correction off the hot path entirely.
    MEMO_MAX = 50_000

    def __init__(self):
        self.vocab: set[str] = set()
        self._buckets: dict[str, list[str]] = {}
        self._memo: dict[str, str | None] = {}
        self.ready = False

    def build(self, meta: list[dict], max_chunks: int = 6000):
        """
        Vocabulary from the indexed text plus every filename.

        Sampled rather than exhaustive: 6000 chunks already contain essentially
        the whole domain vocabulary, and it keeps startup under a second.
        """
        counts: Counter[str] = Counter()
        step = max(1, len(meta) // max_chunks)
        for m in meta[::step]:
            counts.update(tokens(m.get("text", "")))
        for m in meta:
            counts.update(tokens(m.get("filename", "")))

        # A word seen once is usually itself a typo or OCR noise; requiring two
        # sightings stops the corrector "fixing" one misspelling into another.
        self.vocab = {w for w, n in counts.items() if len(w) >= 3 and n >= 2}

        # Bucket by first two characters. Typos almost never change the opening
        # letters, and this turns a 40k-word scan into a few hundred comparisons.
        buckets: dict[str, list[tuple[int, str]]] = {}
        for w in self.vocab:
            buckets.setdefault(w[:2], []).append((counts[w], w))
        self._buckets = {
            k: [w for _, w in sorted(v, reverse=True)[: self.BUCKET_CAP]]
            for k, v in buckets.items()
        }
        self._memo.clear()
        self.ready = True
        return len(self.vocab)

    # -- spelling --
    def correct(self, question: str) -> tuple[str, list[tuple[str, str]]]:
        """Returns (rewritten question, [(original, replacement), ...])."""
        if not self.ready:
            return question, []

        changes: list[tuple[str, str]] = []

        def fix(match: re.Match) -> str:
            word = match.group(0)
            low = word.lower()
            if (len(low) < self.MIN_CORRECT_LEN or low in self.vocab
                    or low in _PROTECTED or not low.isalpha()):
                return word
            if low in self._memo:
                hit = self._memo[low]
                if hit is None:
                    return word
                changes.append((word, hit))
                return hit

            bucket = self._buckets.get(low[:2])
            near = (difflib.get_close_matches(low, bucket, n=1, cutoff=self.CUTOFF)
                    if bucket else None)
            best = near[0] if near and near[0] != low else None
            if len(self._memo) < self.MEMO_MAX:
                self._memo[low] = best
            if best is None:
                return word
            changes.append((word, best))
            return best

        return re.sub(r"[A-Za-z]+", fix, question), changes

    # -- quality --
    def quality(self, question: str, base_min_score: float) -> dict:
        """
        Decide whether a question is worth retrieving for, and how confident a
        match has to be before we answer it.

        Short queries are not rejected on length -- "LCR", "AML" and "KYC" are
        real questions. They are held to a higher similarity bar instead, which
        real acronyms clear (they appear verbatim in the corpus) and noise does
        not.
        """
        q = question.strip()
        if len(q) < 2:
            return {"ok": False, "reason": "empty", "min_score": base_min_score}

        toks = tokens(q)
        if not toks:
            return {"ok": False, "reason": "no_words", "min_score": base_min_score}

        # "ssss", "aaaaaa", "saaaaaaaaaaaaaaaa"
        if all(len(set(t)) == 1 for t in toks) or _REPEAT.search(q.lower()):
            return {"ok": False, "reason": "mashed_keys", "min_score": base_min_score}

        content = [t for t in toks if t not in _STOP and len(t) > 1]
        known = [t for t in content if t in self.vocab] if self.ready else content

        # A real question almost always contains at least one word that appears
        # in the corpus. Short input that contains none is the dangerous case:
        # it still retrieves something, at a plausible-looking score.
        if len(content) <= 2 and not known:
            return {"ok": True, "reason": "short_unknown", "min_score": max(base_min_score, 0.75)}
        if len(content) <= 2:
            return {"ok": True, "reason": "short_known", "min_score": max(base_min_score, 0.55)}
        if not known:
            return {"ok": True, "reason": "no_known_terms", "min_score": max(base_min_score, 0.62)}
        return {"ok": True, "reason": "ok", "min_score": base_min_score}

    # -- verbose questions --
    #
    # There is deliberately no keyword-extraction step here. Stripping filler
    # from a long question and searching with the remainder was tried and
    # measured against this corpus, and it made retrieval worse: on
    # "I was wondering if you could tell me what the rules are around how much a
    # bank is allowed to lend to someone buying a motor vehicle", the full
    # question found the Loan to Value direction at 75.1% while the stripped
    # version found an unrelated Foreign Exchange Act at 75.3% -- a nonsense
    # 0.2-point "win" on a worse document. On the two other verbose questions
    # tried, the full question won by 3.5 and 2.1 points.
    #
    # bge-small is trained on natural sentences and handles the filler itself.
    # Leave the question alone.

    # -- follow-ups --
    def wants_brevity(self, question: str, history: list[dict]) -> bool:
        """
        True when the message asks for the previous answer in fewer words.

        Two conditions, and both are needed. There has to be a request for
        brevity, and there has to be no new subject -- because "make it
        shorter" means condense what is on screen, while "summarise the Mobile
        Payments Guidelines No. 2 of 2011" is a fresh request for a summary of a
        named document and must not be answered out of the last turn.

        The subject bar is tighter than is_followup's. A single stray word is
        allowed for phrasing, but two or more mean the sentence is about
        something, and something is not the previous answer.
        """
        if not history or not _BREVITY_RE.search(question):
            return False
        subject = [t for t in tokens(question)
                   if t not in _STOP and t not in _META_WORDS
                   and t not in _BREVITY_WORDS and len(t) > 2]
        return len(subject) <= 1

    def is_followup(self, question: str, history: list[dict]) -> bool:
        """
        True when the question continues the previous turn rather than opening a
        new subject.

        Length is not the test. An earlier version treated anything over 12
        words as a fresh question, which meant "that answer is too long, please
        give a more concise answer for the above question I just asked" was
        searched literally -- and matched an unrelated Customer Charter at 71%,
        comfortably above the confidence bar, so nothing looked wrong. What
        matters is whether the sentence names a new subject, not how many words
        it spends not naming one.
        """
        if not history:
            return False
        if _META_REF.search(question):
            return True
        if not _ANAPHORA.search(question):
            return False
        subject = [t for t in tokens(question)
                   if t not in _STOP and t not in _META_WORDS and len(t) > 2]
        return len(subject) <= 3

    def expand(self, question: str, history: list[dict]) -> tuple[str, bool]:
        """
        Resolve a follow-up into something the retriever can act on.

        "tell me more about it" carries no topic, so its embedding lands nowhere
        near the document it refers to. Content words from the previous turn are
        appended to the retrieval query -- not to what the model or the user
        sees, only to the vector we search with.

        A model rewrite would be more accurate but costs a whole extra
        generation on the critical path, which is not worth it here.
        """
        if not self.is_followup(question, history):
            return question, False
        toks = tokens(question)

        prev = history[-1]
        carry: list[str] = []
        seen = set(toks)
        for source in (prev.get("question", ""), prev.get("answer", "")[:300]):
            for t in tokens(source):
                if t in _STOP or len(t) < 3 or t in seen:
                    continue
                seen.add(t)
                carry.append(t)
            if len(carry) >= 12:
                break
        if not carry:
            return question, False
        return f"{question} {' '.join(carry[:12])}", True


fixer = QueryFixer()


def is_greeting(question: str) -> bool:
    return bool(_GREETING.match(question.strip()))
