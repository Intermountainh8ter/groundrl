"""Reward functions v0, v1 and v2, plus TRL-compatible wrappers.

Every reward is a pure function ``(completion, example) -> (score, breakdown)`` with the score in
[0, 1]. The versions exist because each earlier one has a documented hole (see
``docs/METHODOLOGY.md`` and the audit in ``scripts/reward_audit.py``):

* **v0** binary: 1.0 if the verifier accepts the answer. No partial credit, and an honest
  abstention scores 0 (it is an uncited sentence).
* **v1** graded: per-sentence partial credit for citation validity, the fraction of claim tokens
  supported and lexical coverage. Dense, but still blind to whether the answer is the *right*
  answer.
* **v2** v1 plus gold-aware terms, each of which can be switched off for ablation:

  ``abstention``   abstain scores 1 only when the gold label says the context cannot answer;
                   answering an unanswerable question scores 0
  ``correctness``  recall x precision of the answer's claim tokens against the gold tokens
                   (tokens echoed from the question are neutral)
  ``citation``     precision x recall of the cited chunks against the gold support groups
  ``relevance``    share of the question's claim tokens that the answer repeats
  ``length``       penalty for answers longer than 1.5x the gold answer plus a small allowance
  ``redundancy``   penalty for repeated sentences and repeated citation markers
  ``consistency``  penalty when two near-identical sentences assign different numbers (hedging)

  ``score = consistency * length * redundancy * grounding * (floor + (1 - floor) * correctness
  * citation * (0.7 + 0.3 * relevance))`` where ``grounding`` is the v1 score and ``floor`` keeps
  a small gradient for well-formed, grounded answers that are not yet correct. ``correctness`` is
  ``recall * precision ** 2``: a wrong extra claim costs more than a missing one.

  v2 has two revisions. Revision 1 (``V2Config.rev1()``) had no consistency term and used
  ``recall * precision``; the audit showed it could be exploited by answering twice with
  contradicting values, so revision 2 (the default) added both changes.

Training wrappers (:func:`make_trl_reward`) have the signature ``fn(prompts, completions,
**kwargs) -> list[float]`` expected by TRL and need no TRL import.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from statistics import fmean

from .example import Example
from .protocol import CITATION, is_abstention
from .tokens import (
    claim_tokens,
    content_words,
    number_variants,
    sentences,
    strip_citations,
    word_count,
)
from .verifier import VerificationResult, verify_answer

Breakdown = dict[str, float | int | bool | str]
RewardFn = Callable[..., tuple[float, Breakdown]]

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_SECTION_HINT = re.compile(r'\(section "[^"]*"\)')


def strip_think(text: str) -> str:
    """Drop ``<think>...</think>`` blocks; an unterminated block discards the rest."""
    text = _THINK.sub("", text)
    cut = text.find("<think>")
    return text if cut < 0 else text[:cut]


def completion_text(completion: object) -> str:
    """The assistant text of a TRL completion (plain string or chat-message list)."""
    if isinstance(completion, str):
        return strip_think(completion)
    if isinstance(completion, Mapping):
        return strip_think(str(completion.get("content", "")))
    if isinstance(completion, list | tuple):
        parts = [str(m.get("content", "")) for m in completion
                 if isinstance(m, Mapping) and m.get("role", "assistant") == "assistant"]
        return strip_think("".join(parts))
    return strip_think(str(completion))


# ---------------------------------------------------------------------------------------------
# v0, v1


def _verify(completion: str, ex: Example) -> VerificationResult:
    return verify_answer(completion, ex.context, ex.context_ids)


def reward_v0(completion: str, ex: Example) -> tuple[float, Breakdown]:
    """Binary: 1.0 if the verifier accepts the answer, else 0.0."""
    res = _verify(completion, ex)
    return (1.0 if res.ok else 0.0), {
        "version": "v0", "verified": res.ok, "issues": len(res.issues),
        "n_sentences": len(res.sentences),
    }


def graded_grounding(res: VerificationResult) -> float:
    """Mean per-sentence credit in [0, 1]; the v1 score.

    A sentence earns nothing without a valid citation. Otherwise it earns
    ``(1 - 0.5 * share_of_invalid_citations) * (0.7 * support_ratio + 0.3 * coverage)``.
    An answer containing invisible or look-alike characters scores 0 outright.
    """
    if res.suspicious or not res.sentences:
        return 0.0
    credits: list[float] = []
    for s in res.sentences:
        if not s.has_content or not s.valid:
            credits.append(0.0)
            continue
        total = len(s.valid) + len(s.invalid)
        cite_factor = 1.0 - 0.5 * len(s.invalid) / total
        credits.append(cite_factor * (0.7 * s.support_ratio + 0.3 * s.coverage))
    return fmean(credits)


def reward_v1(completion: str, ex: Example) -> tuple[float, Breakdown]:
    res = _verify(completion, ex)
    g = graded_grounding(res)
    return g, {
        "version": "v1", "verified": res.ok, "grounded": g, "support_ratio": res.support_ratio,
        "n_sentences": len(res.sentences), "hallucinated_citations": len(res.invalid_citations),
        "issues": len(res.issues),
    }


# ---------------------------------------------------------------------------------------------
# v2


@dataclass(frozen=True)
class V2Config:
    abstention: bool = True
    correctness: bool = True
    citation: bool = True
    relevance: bool = True
    length: bool = True
    redundancy: bool = True
    consistency: bool = True
    floor: float = 0.10
    length_slack: float = 1.5
    length_pad: int = 15
    relevance_weight: float = 0.3
    precision_power: float = 2.0
    consistency_factor: float = 0.25

    @classmethod
    def rev1(cls) -> V2Config:
        """The first v2 design, kept so the audit can show what revision 2 changed."""
        return cls(consistency=False, precision_power=1.0)

    def without(self, term: str) -> V2Config:
        if term not in V2_TERMS:
            raise ValueError(f"unknown reward term {term!r}; expected one of {V2_TERMS}")
        return replace(self, **{term: False})


V2_TERMS = ("abstention", "correctness", "citation", "relevance", "length", "redundancy",
            "consistency")


def token_equiv(a: str, b: str) -> bool:
    """Equal tokens, or numerically equivalent spellings (0x40 = 64 = 2**6)."""
    if a == b:
        return True
    return a[:1].isdigit() and b[:1].isdigit() and b in number_variants(a)


def _in(tok: str, bag: set[str]) -> bool:
    return any(token_equiv(tok, other) for other in bag)


def correctness(
    answer: str, ex: Example, precision_power: float = 1.0,
) -> tuple[float, float, float]:
    """``(recall x precision**power, recall, precision)`` of claim tokens vs the gold tokens.

    Tokens that also occur in the question, and the numbers of the EIPs involved, are neutral:
    repeating them neither earns credit nor counts as an unsupported extra.
    """
    gold = set(ex.task.gold_tokens)
    if not gold:
        return 1.0, 1.0, 1.0
    answer_tokens = claim_tokens(strip_citations(answer))
    neutral = claim_tokens(ex.task.question) | {e.split("-", 1)[1] for e in ex.task.eips}
    matched = {a for a in answer_tokens if _in(a, gold)}
    extras = {a for a in answer_tokens if a not in matched and not _in(a, neutral)}
    recall = sum(_in(g, answer_tokens) for g in gold) / len(gold)
    denom = len(matched) + len(extras)
    precision = len(matched) / denom if denom else 0.0
    return recall * precision**precision_power, recall, precision


def citation_quality(cited: list[str], ex: Example) -> tuple[float, float, float]:
    """``(precision x recall, precision, recall)`` of cited chunks against the gold groups."""
    groups = [set(g) for g in ex.task.gold_groups if g]
    if not groups:
        return 1.0, 1.0, 1.0
    union = set().union(*groups)
    recall = sum(bool(set(cited) & g) for g in groups) / len(groups)
    precision = len(set(cited) & union) / len(set(cited)) if cited else 0.0
    return precision * recall, precision, recall


def relevance(answer: str, ex: Example) -> float:
    """Share of the question's claim tokens that the answer repeats (1.0 if it has none).

    The EIP numbers and the quoted section title are excluded: they say where to look, not what
    is being asked.
    """
    body = _SECTION_HINT.sub("", ex.task.question)
    q = claim_tokens(body) - {e.split("-", 1)[1] for e in ex.task.eips}
    if not q:
        return 1.0
    a = claim_tokens(strip_citations(answer))
    return sum(_in(t, a) for t in q) / len(q)


def length_factor(answer: str, ex: Example, cfg: V2Config) -> tuple[float, int, float]:
    """``(factor, words, budget)``: 1 to the budget, then linear down to 0 at twice the budget."""
    words = word_count(answer)
    budget = cfg.length_slack * word_count(ex.task.gold_answer) + cfg.length_pad
    if words <= budget:
        return 1.0, words, budget
    return max(0.0, 1.0 - (words - budget) / budget), words, budget


def _sentence_key(s: str) -> str:
    return re.sub(r"[\W_]+", " ", strip_citations(s)).strip().lower()


def redundancy_factor(answer: str) -> tuple[float, float, float]:
    """``(factor, repeated_sentence_share, repeated_citation_share)``."""
    sents = sentences(answer)
    if not sents:
        return 1.0, 0.0, 0.0
    keys = [_sentence_key(s) for s in sents]
    dup_sentences = 1.0 - len(set(keys)) / len(keys)
    markers = repeats = 0
    for s in sents:
        cites = CITATION.findall(s)
        markers += len(cites)
        repeats += len(cites) - len(set(cites))
    dup_cites = repeats / markers if markers else 0.0
    return (1.0 - dup_sentences) * (1.0 - dup_cites), dup_sentences, dup_cites


def _canon(tok: str) -> str:
    """Decimal spelling of a numeric token when one exists (0x40 -> 64), else the token."""
    dec = [v for v in number_variants(tok) if v.isdigit()]
    return dec[0] if dec else tok


def contradictions(answer: str, ex: Example) -> int:
    """Pairs of near-identical sentences that give different numbers (answering twice).

    Two sentences are near-identical if their non-numeric content words overlap by at least 80%
    (Jaccard). They contradict if both carry numeric claims and the sets differ. EIP numbers are
    ignored. Legitimate multi-fact answers differ in their words, so they do not trigger this.
    """
    eips = {e.split("-", 1)[1] for e in ex.task.eips}
    rows: list[tuple[set[str], set[str]]] = []
    for s in sentences(answer):
        body = strip_citations(s)
        words = {w for w in content_words(body) if not w[:1].isdigit()}
        nums = {_canon(t) for t in claim_tokens(body) if t[:1].isdigit()} - eips
        rows.append((words, nums))
    count = 0
    for i, (wi, ni) in enumerate(rows):
        for wj, nj in rows[i + 1:]:
            union = wi | wj
            if not union or not ni or not nj or ni == nj:
                continue
            if len(wi & wj) / len(union) >= 0.8:
                count += 1
    return count


def reward_v2(completion: str, ex: Example, cfg: V2Config | None = None) -> tuple[float, Breakdown]:
    cfg = cfg or V2Config()
    task = ex.task
    abstained = is_abstention(completion)
    bd: Breakdown = {"version": "v2", "answerable": task.answerable, "abstained": abstained}
    if cfg.abstention:
        if abstained:
            bd["score_reason"] = "abstained"
            score = 0.0 if task.answerable else 1.0
            bd["verified"] = False
            return score, bd
        if not task.answerable:
            bd["score_reason"] = "answered_unanswerable"
            bd["verified"] = False
            return 0.0, bd
    res = _verify(completion, ex)
    g = graded_grounding(res)
    c = rc = pc = k = kp = kr = 1.0
    rel = 1.0
    if task.answerable or not cfg.abstention:
        if cfg.correctness and task.answerable:
            c, rc, pc = correctness(completion, ex, cfg.precision_power)
        if cfg.citation and task.answerable:
            k, kp, kr = citation_quality(res.cited_chunks, ex)
    if cfg.relevance:
        rel = relevance(completion, ex)
    rf = (1.0 - cfg.relevance_weight) + cfg.relevance_weight * rel if cfg.relevance else 1.0
    lf, words, budget = length_factor(completion, ex, cfg) if cfg.length else (1.0, 0, 0.0)
    df, dup_s, dup_c = redundancy_factor(completion) if cfg.redundancy else (1.0, 0.0, 0.0)
    conflicts = contradictions(completion, ex) if cfg.consistency else 0
    cf = cfg.consistency_factor if conflicts else 1.0
    inner = cfg.floor + (1.0 - cfg.floor) * c * k * rf
    score = max(0.0, min(1.0, cf * lf * df * g * inner))
    bd.update({
        "verified": res.ok, "grounded": g, "correctness": c, "recall": rc, "precision": pc,
        "citation_quality": k, "cite_precision": kp, "cite_recall": kr, "relevance": rel,
        "length_factor": lf, "words": words, "word_budget": budget, "redundancy_factor": df,
        "dup_sentences": dup_s, "dup_citations": dup_c, "contradictions": conflicts,
        "consistency_factor": cf,
        "hallucinated_citations": len(res.invalid_citations), "support_ratio": res.support_ratio,
        "n_sentences": len(res.sentences),
    })
    return score, bd


REWARD_VERSIONS: dict[str, RewardFn] = {"v0": reward_v0, "v1": reward_v1, "v2": reward_v2}


def score_completion(version: str, completion: str, ex: Example,
                     cfg: V2Config | None = None) -> tuple[float, Breakdown]:
    if version not in REWARD_VERSIONS:
        raise ValueError(f"unknown reward version {version!r}; expected one of "
                         f"{sorted(REWARD_VERSIONS)}")
    if version == "v2":
        return reward_v2(completion, ex, cfg)
    return REWARD_VERSIONS[version](completion, ex)


# ---------------------------------------------------------------------------------------------
# TRL wrappers


def make_trl_reward(
    version: str = "v2", cfg: V2Config | None = None, column: str = "example_json",
) -> Callable[..., list[float]]:
    """Build ``fn(prompts, completions, **kwargs) -> list[float]`` for TRL's ``GRPOTrainer``.

    The dataset must carry the per-row :class:`~groundrl.example.Example` as JSON in ``column``
    (``train/data.py`` writes it). TRL forwards extra dataset columns as keyword arguments, one
    list entry per completion. If the trainer passes ``log_metric`` the wrapper reports
    diagnostics (verified rate, abstain rate, hallucinated-citation rate, mean words) so reward
    hacking shows up in the training logs before it shows up in the reward.
    """
    if version not in REWARD_VERSIONS:
        raise ValueError(f"unknown reward version {version!r}")

    def reward_fn(prompts=None, completions=None, **kwargs) -> list[float]:
        blobs = kwargs.get(column)
        if blobs is None:
            raise KeyError(f"dataset column {column!r} missing; the reward needs the task context")
        if completions is None or len(blobs) != len(completions):
            raise ValueError("one example per completion is required")
        scores: list[float] = []
        rows: list[Breakdown] = []
        for comp, blob in zip(completions, blobs, strict=True):
            s, bd = score_completion(version, completion_text(comp), Example.from_json(blob), cfg)
            scores.append(s)
            rows.append(bd)
        log_metric = kwargs.get("log_metric")
        if callable(log_metric) and rows:
            _log_diagnostics(log_metric, rows)
        return scores

    reward_fn.__name__ = f"groundrl_reward_{version}"
    return reward_fn


def _log_diagnostics(log_metric: Callable[[str, float], None], rows: list[Breakdown]) -> None:
    n = len(rows)

    def rate(key: str) -> float:
        return sum(bool(r.get(key, False)) for r in rows) / n

    log_metric("groundrl/verified_rate", rate("verified"))
    log_metric("groundrl/abstain_rate", rate("abstained"))
    log_metric("groundrl/hallucinated_citation_rate",
               sum(float(r.get("hallucinated_citations", 0)) > 0 for r in rows) / n)
    if any("words" in r for r in rows):
        log_metric("groundrl/mean_words", fmean(float(r.get("words", 0)) for r in rows))
    if any("correctness" in r for r in rows):
        log_metric("groundrl/correctness", fmean(float(r.get("correctness", 0)) for r in rows))
