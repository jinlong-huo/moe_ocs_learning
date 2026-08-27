"""
suite.py — Categorised workload suite for MoE routing / expert-affinity study.

Why this file exists
────────────────────
The previous experiment compared exactly two prompts ("why MoE needs routing"
vs "how gradient descent works").  Two prompts cannot distinguish the three
hypotheses that matter:

  H_sem   routing is driven by SEMANTIC domain (coding vs history ...)
  H_lex   routing is driven by SURFACE FORM (shared tokens, syntax, length)
  H_none  routing is essentially token-identity driven and domain-invariant

A two-prompt comparison confounds all three, because two prompts from
different domains also differ lexically and in length.  This suite is built
as a factorial design so the three can be separated:

  * ``CATEGORIES``  — 12 semantic domains x 6 prompts.  Between-category
    contrasts vary BOTH semantics and lexis.

  * ``PARAPHRASE_SETS`` — groups of prompts with the SAME semantics and
    deliberately MINIMAL lexical overlap.  High routing similarity inside a
    paraphrase set supports H_sem; low similarity supports H_lex.

  * ``LEXICAL_CONTROLS`` — pairs with HIGH lexical overlap but DIFFERENT
    semantic domain (same template, swapped subject).  High routing
    similarity here supports H_lex; low similarity supports H_sem.

  * ``LENGTH_LADDER`` — the same request at 3 lengths, to check that measured
    "affinity" is not just a sequence-length artifact.

Statistical note: category labels are never used to *fit* anything that is
then evaluated on the same label.  ``split_by_category`` and
``split_within_category`` produce the disjoint fit/eval sets used by
``src/eval/`` so that every affinity/placement number reported is
out-of-sample.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════
# 1. Semantic categories — 12 domains x 6 prompts
#    Within each category prompts deliberately vary in: wording, length,
#    difficulty, subject and syntactic form (question / imperative /
#    declarative-completion), so that a category is NOT a single template.
# ═══════════════════════════════════════════════════════════════════════

CATEGORIES: dict[str, list[str]] = {
    # ── programming: generation ────────────────────────────────────────
    "code_generation": [
        "Write a Python function that merges two sorted linked lists into one sorted list.",
        "Implement a thread-safe LRU cache in Java with O(1) get and put.",
        "Given an array of integers, return all triplets that sum to zero. Provide C++ code.",
        "Produce a SQL query returning, per department, the employee with the third highest salary.",
        "Write a Rust iterator adaptor that yields overlapping windows of size n over a slice.",
        "Create a bash one-liner that finds every file larger than 100MB modified in the last week.",
    ],
    # ── programming: debugging / reading (adjacent sub-domain) ─────────
    "code_debugging": [
        "This loop sometimes skips the last element: for i in range(len(a)-1): print(a[i]). Why?",
        "My Python dict comprehension raises 'unhashable type: list'. Explain the cause and the fix.",
        "A Go program deadlocks when two goroutines send on unbuffered channels. Diagnose it.",
        "Explain why comparing floating point numbers with == can fail and what to do instead.",
        "A React component re-renders infinitely after I call setState inside useEffect. Why?",
        "Segmentation fault occurs after free(p); p->next = NULL; identify the bug class.",
    ],
    # ── mathematics ───────────────────────────────────────────────────
    "math_reasoning": [
        "A train leaves at 09:00 at 60 km/h; another at 10:30 at 90 km/h. When does the second catch up?",
        "Prove that the square root of 3 is irrational.",
        "Compute the derivative of f(x) = x^2 * ln(x) and state where it is increasing.",
        "How many distinct ways can 8 identical balls be placed into 3 labelled boxes?",
        "Find all real solutions of the equation 2^x + 2^(-x) = 5/2.",
        "A fair coin is tossed 10 times. What is the probability of exactly 3 heads?",
    ],
    # ── natural science ───────────────────────────────────────────────
    "science_qa": [
        "Why is the sky blue during the day but red near sunset?",
        "Describe how CRISPR-Cas9 introduces a targeted double-strand break in DNA.",
        "What distinguishes a type Ia supernova from a core-collapse supernova?",
        "Explain the role of the sodium-potassium pump in maintaining a neuron's resting potential.",
        "How does the greenhouse effect change the Earth's radiative balance?",
        "Why do superconductors expel magnetic fields below the critical temperature?",
    ],
    # ── history ───────────────────────────────────────────────────────
    "history_qa": [
        "What were the main economic causes of the French Revolution?",
        "Describe how the printing press altered the spread of ideas in 16th century Europe.",
        "Compare the administrative structures of the Roman Republic and the Roman Empire.",
        "Why did the Bronze Age collapse affect so many Mediterranean civilisations at once?",
        "Explain the significance of the Treaty of Westphalia for the modern state system.",
        "How did the Meiji Restoration change Japan's relationship with Western powers?",
    ],
    # ── short factual recall (low reasoning depth) ────────────────────
    "factual_recall": [
        "What is the capital of Australia?",
        "Who wrote the novel One Hundred Years of Solitude?",
        "In which year did the Berlin Wall fall?",
        "What is the chemical symbol for tungsten?",
        "Which planet has the shortest day in the solar system?",
        "What is the longest river in South America?",
    ],
    # ── instruction following with hard format constraints ────────────
    "instruction_following": [
        "List exactly four uses of a paperclip. Number them 1 to 4. Do not add any other text.",
        "Rewrite the following in exactly ten words: 'The meeting has been postponed until further notice.'",
        "Answer only with YES or NO: is 91 a prime number?",
        "Output a JSON object with keys name, age and city, filled with plausible example values.",
        "Reply using only lowercase letters and no punctuation: describe rain.",
        "Give three bullet points, each starting with a verb, on how to prepare for an interview.",
    ],
    # ── creative writing ──────────────────────────────────────────────
    "creative_writing": [
        "Write the opening paragraph of a detective story set in a flooded city.",
        "Compose a four-line poem about an abandoned lighthouse without using the word 'sea'.",
        "Invent a short myth explaining why cats sleep so much.",
        "Write dialogue between a retired astronaut and a child who wants to be one.",
        "Describe a marketplace at dawn using only smells and sounds.",
        "Write a product advert for an umbrella that refuses to open in the rain.",
    ],
    # ── summarisation ─────────────────────────────────────────────────
    "summarization": [
        "Summarise in one sentence: Photosynthesis converts light energy into chemical energy stored "
        "in glucose, consuming carbon dioxide and water and releasing oxygen as a by-product.",
        "Condense to a headline: Heavy rainfall overnight caused three rivers to burst their banks, "
        "forcing the evacuation of four hundred residents from low-lying districts.",
        "Give the key point in under fifteen words: The committee reviewed the proposal, requested "
        "additional cost estimates, and deferred its decision to the next quarterly session.",
        "Summarise the argument: Because remote work removes commuting time, employees reclaim hours, "
        "but reduced spontaneous contact can slow the transfer of tacit knowledge between teams.",
        "Reduce to three words: The experiment failed to reproduce the originally reported effect size.",
        "Write an abstract-style summary of a study finding that sleep deprivation impairs "
        "consolidation of motor skills but not of verbal recall.",
    ],
    # ── translation ───────────────────────────────────────────────────
    "translation": [
        "Translate into French: 'The library closes at six on weekdays.'",
        "Translate into Japanese: 'Please leave the package with the neighbour.'",
        "Render this into formal German: 'We regret to inform you that your application was unsuccessful.'",
        "Translate into Spanish and keep the imperative mood: 'Do not open the valve before venting.'",
        "Translate into Mandarin Chinese: 'The bridge was built in nineteen thirty-two.'",
        "Translate into Arabic: 'Water expands when it freezes.'",
    ],
    # ── technical explanation (systems / networking — near the paper's own domain) ──
    "technical_explanation": [
        "Explain how TCP congestion control reacts to packet loss and why that harms long fat pipes.",
        "Describe what a page fault is and the steps the kernel takes to service one.",
        "How does a copy-on-write filesystem snapshot avoid duplicating unchanged blocks?",
        "Explain the difference between a blocking and a non-blocking optical switch fabric.",
        "Why does false sharing between CPU cores degrade throughput on a multithreaded counter?",
        "Describe how consistent hashing limits key movement when a cache node is removed.",
    ],
    # ── open conversational ───────────────────────────────────────────
    "conversational": [
        "I have two hours free in an unfamiliar city. What should I do?",
        "My houseplant's leaves are turning yellow from the bottom up. Any ideas?",
        "Is it worth learning to cook if I live alone?",
        "I keep starting books and never finishing them. How do I fix that?",
        "What's a good gift for someone who says they want nothing?",
        "How do I politely leave a party early?",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 2. Paraphrase sets — SAME semantics, MINIMAL lexical overlap.
#    Tests H_sem: if routing follows meaning, members of a set should route
#    alike despite sharing almost no content words.
# ═══════════════════════════════════════════════════════════════════════

PARAPHRASE_SETS: dict[str, list[str]] = {
    "para_sort_algo": [
        "Write code that orders a list of numbers from smallest to largest.",
        "Implement an ascending arrangement routine for an integer sequence.",
        "Given unsorted values, produce a program returning them in increasing order.",
    ],
    "para_revolution_cause": [
        "What made the population of late eighteenth century France rise against its king?",
        "Identify the grievances that drove the 1789 overthrow of the Bourbon monarchy.",
        "Which social and fiscal pressures precipitated the fall of the ancien regime?",
    ],
    "para_photosynthesis": [
        "How do green plants turn sunlight into stored chemical energy?",
        "Describe the biological process converting radiant energy into carbohydrate.",
        "Explain how foliage manufactures sugar using illumination, gas and moisture.",
    ],
    "para_prob_coin": [
        "If I flip a balanced coin ten times, how likely am I to see heads three times?",
        "Determine the chance of obtaining exactly three successes in ten unbiased trials.",
        "Compute the binomial likelihood of three favourable outcomes among ten tosses.",
    ],
    "para_translate_greeting": [
        "Put this sentence into French: the shop shuts at eighteen hundred.",
        "Give the French rendering of: closing time for the store is 6 pm.",
        "Express in French that the establishment ceases trading at six in the evening.",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 3. Lexical controls — HIGH lexical overlap, DIFFERENT domain.
#    One template per group; only the subject noun-phrase changes domain.
#    Tests H_lex: if routing follows surface form, these should route alike
#    even though their domains differ.
# ═══════════════════════════════════════════════════════════════════════

LEXICAL_CONTROLS: dict[str, list[str]] = {
    "lex_causes_template": [
        "Explain the major causes and consequences of the French Revolution, "
        "including the social, economic and political factors involved.",
        "Explain the major causes and consequences of a memory leak in a long-running "
        "server process, including the allocation, retention and growth factors involved.",
        "Explain the major causes and consequences of coral bleaching events, "
        "including the thermal, chemical and biological factors involved.",
    ],
    "lex_stepbystep_template": [
        "Step by step, work out how many prime numbers lie below one hundred.",
        "Step by step, work out how a packet traverses a three tier datacenter network.",
        "Step by step, work out how a sourdough starter becomes active over five days.",
    ],
    "lex_compare_template": [
        "Compare and contrast the two approaches, listing three advantages of each: "
        "quicksort versus mergesort.",
        "Compare and contrast the two approaches, listing three advantages of each: "
        "the Roman Republic versus the Roman Empire.",
        "Compare and contrast the two approaches, listing three advantages of each: "
        "electrical packet switching versus optical circuit switching.",
    ],
    "lex_write_template": [
        "Write a short clear paragraph explaining recursion to a beginner.",
        "Write a short clear paragraph explaining the Marshall Plan to a beginner.",
        "Write a short clear paragraph explaining osmosis to a beginner.",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 4. Length ladder — same request, 3 lengths. Controls for the possibility
#    that measured affinity is a sequence-length artifact.
# ═══════════════════════════════════════════════════════════════════════

LENGTH_LADDER: dict[str, list[str]] = {
    "len_sorting": [
        "Sort a list in Python.",
        "Explain how to sort a list of integers in Python and mention the time complexity.",
        "Explain in detail how to sort a list of integers in Python. Cover the built-in "
        "sorted function, the key argument, stability guarantees, the underlying Timsort "
        "algorithm, its best and worst case time complexity, and when you would instead "
        "reach for a heap or a counting sort.",
    ],
    "len_history": [
        "Why did Rome fall?",
        "Explain the main reasons for the decline of the Western Roman Empire.",
        "Explain in detail the main reasons for the decline of the Western Roman Empire, "
        "covering fiscal strain, currency debasement, military overextension, the "
        "settlement of federate peoples, administrative division after Diocletian, and "
        "the historiographical debate over continuity versus collapse.",
    ],
    "len_physics": [
        "Why is the sky blue?",
        "Explain why the daytime sky appears blue and why sunsets appear red.",
        "Explain in detail why the daytime sky appears blue while sunsets appear red, "
        "covering Rayleigh scattering, its wavelength dependence, the role of atmospheric "
        "path length at low solar elevation, the contribution of aerosols and Mie "
        "scattering, and why the sky is not violet.",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# 5. Repeat set — identical prompt, repeated. Establishes the measurement
#    NOISE FLOOR (numerical nondeterminism of the quantised gate).  Every
#    similarity number elsewhere must be read against this floor.
# ═══════════════════════════════════════════════════════════════════════

REPEAT_PROMPT = (
    "Explain how a hash table resolves collisions and what happens when the "
    "load factor grows too large."
)
N_REPEATS_DEFAULT = 4


# ═══════════════════════════════════════════════════════════════════════
# Records + builders
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PromptSpec:
    """One prompt with the labels every downstream analysis needs."""

    uid: str
    prompt: str
    category: str            # semantic domain, or the control-group id
    group: str               # finer grouping (paraphrase set / template / ladder)
    role: str                # "category" | "paraphrase" | "lexical_control"
                             # | "length_ladder" | "repeat"
    variant: int = 0         # index within group (ladder rung, repeat index)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "uid": self.uid, "prompt": self.prompt, "category": self.category,
            "group": self.group, "role": self.role, "variant": self.variant,
            "meta": self.meta,
        }


def build_suite(
    *,
    include_categories: bool = True,
    include_paraphrase: bool = True,
    include_lexical: bool = True,
    include_length: bool = True,
    n_repeats: int = N_REPEATS_DEFAULT,
    per_category: int | None = None,
    seed: int = 0,
) -> list[PromptSpec]:
    """Assemble the full suite.

    ``per_category`` subsamples each semantic category (seeded) so a cheap
    smoke run and the full run use the *same* code path.
    """
    rng = random.Random(seed)
    out: list[PromptSpec] = []

    if include_categories:
        for cat, prompts in CATEGORIES.items():
            sel = list(prompts)
            if per_category is not None and per_category < len(sel):
                sel = rng.sample(sel, per_category)
                sel.sort(key=prompts.index)          # keep deterministic order
            for i, p in enumerate(sel):
                out.append(PromptSpec(
                    uid=f"cat.{cat}.{i:02d}", prompt=p, category=cat,
                    group=cat, role="category", variant=i,
                ))

    if include_paraphrase:
        for gid, prompts in PARAPHRASE_SETS.items():
            for i, p in enumerate(prompts):
                out.append(PromptSpec(
                    uid=f"para.{gid}.{i:02d}", prompt=p, category=gid,
                    group=gid, role="paraphrase", variant=i,
                ))

    if include_lexical:
        for gid, prompts in LEXICAL_CONTROLS.items():
            for i, p in enumerate(prompts):
                out.append(PromptSpec(
                    uid=f"lex.{gid}.{i:02d}", prompt=p, category=gid,
                    group=gid, role="lexical_control", variant=i,
                ))

    if include_length:
        for gid, prompts in LENGTH_LADDER.items():
            for i, p in enumerate(prompts):
                out.append(PromptSpec(
                    uid=f"len.{gid}.{i:02d}", prompt=p, category=gid,
                    group=gid, role="length_ladder", variant=i,
                    meta={"rung": ["short", "medium", "long"][min(i, 2)]},
                ))

    for i in range(max(0, n_repeats)):
        out.append(PromptSpec(
            uid=f"rep.noise.{i:02d}", prompt=REPEAT_PROMPT, category="repeat",
            group="repeat", role="repeat", variant=i,
        ))

    return out


# ═══════════════════════════════════════════════════════════════════════
# Fit / eval splits — every reported affinity or placement number must be
# produced by fitting on one side and scoring on the other.
# ═══════════════════════════════════════════════════════════════════════

def split_within_category(specs: list[PromptSpec], seed: int = 0
                          ) -> tuple[list[str], list[str]]:
    """Stratified half/half split *inside* each category.

    Fit and eval see the same domains but different prompts.  Answers:
    "does an affinity graph learned from some coding prompts transfer to
    other coding prompts?"
    """
    rng = random.Random(seed)
    by_cat: dict[str, list[PromptSpec]] = {}
    for s in specs:
        if s.role != "category":
            continue
        by_cat.setdefault(s.category, []).append(s)
    fit, ev = [], []
    for cat in sorted(by_cat):
        items = sorted(by_cat[cat], key=lambda s: s.uid)
        rng.shuffle(items)
        h = len(items) // 2
        fit += [s.uid for s in items[:h]]
        ev += [s.uid for s in items[h:]]
    return fit, ev


def split_by_category(specs: list[PromptSpec], holdout: list[str] | None = None,
                      seed: int = 0) -> tuple[list[str], list[str]]:
    """Leave-whole-categories-out split.

    Fit and eval see DISJOINT domains.  Answers the much harder question:
    "does an affinity graph learned on coding+math transfer to history?"
    A placement that helps here is workload-robust; one that only helps
    under ``split_within_category`` is domain-specific.
    """
    cats = sorted({s.category for s in specs if s.role == "category"})
    if holdout is None:
        rng = random.Random(seed)
        cats_shuf = list(cats)
        rng.shuffle(cats_shuf)
        holdout = sorted(cats_shuf[: max(1, len(cats) // 3)])
    hold = set(holdout)
    fit = [s.uid for s in specs if s.role == "category" and s.category not in hold]
    ev = [s.uid for s in specs if s.role == "category" and s.category in hold]
    return fit, ev


def category_of(specs: list[PromptSpec]) -> dict[str, str]:
    return {s.uid: s.category for s in specs}


def suite_summary(specs: list[PromptSpec]) -> dict:
    by_role: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for s in specs:
        by_role[s.role] = by_role.get(s.role, 0) + 1
        by_cat[s.category] = by_cat.get(s.category, 0) + 1
    return {"n_prompts": len(specs), "by_role": by_role, "by_category": by_cat}


if __name__ == "__main__":
    import json
    specs = build_suite()
    print(json.dumps(suite_summary(specs), indent=2))
