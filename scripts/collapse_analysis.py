"""
Collapse analysis for debate training episodes.
Checks: vocabulary richness, structural templates, answer entropy,
argument strategy evolution, copy-paste detection.
"""

import json
import re
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

EPISODES_PATH = Path("logs/thinking-experiment/rung1-no-think/episodes/episodes.jsonl")


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def split_quintiles(episodes: list[dict], n: int = 5) -> list[list[dict]]:
    """Split episodes into n equal quintiles by file order (= training order)."""
    size = len(episodes) // n
    return [episodes[i * size : (i + 1) * size] for i in range(n)]


def extract_turns(episodes: list[dict], phase: str | None = None, role: str | None = None) -> list[str]:
    """Extract text from transcript turns matching optional phase/role filters."""
    texts = []
    for ep in episodes:
        for turn in ep["transcript"]:
            if phase and turn.get("phase") != phase:
                continue
            if role and turn.get("role") != role:
                continue
            texts.append(turn["text"])
    return texts


def strip_answer_tag(text: str) -> str:
    """Remove <answer>...</answer> from text, return the prose part."""
    return re.sub(r"<answer>.*?</answer>\s*", "", text, flags=re.DOTALL).strip()


def extract_answer(text: str) -> str | None:
    """Extract answer from <answer> tags."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    probs = [c / total for c in counter.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


# ============================================================
# 1. VOCABULARY RICHNESS
# ============================================================
def analyze_vocabulary(quintiles: list[list[dict]]):
    print("=" * 70)
    print("1. VOCABULARY RICHNESS OVER TRAINING (propose phase only)")
    print("=" * 70)
    print(f"{'Q':>3} {'TTR':>8} {'Bigram-TR':>10} {'MeanSentLen':>12} {'UniqueW':>8} {'TotalW':>8} {'UniqBi':>8} {'TotalBi':>8}")
    print("-" * 70)

    for qi, q in enumerate(quintiles):
        propose_texts = [strip_answer_tag(t) for t in extract_turns(q, phase="propose")]
        all_tokens = []
        all_bigrams = []
        sent_lengths = []

        for text in propose_texts:
            toks = tokenize(text)
            all_tokens.extend(toks)
            all_bigrams.extend(bigrams(toks))
            for s in sentences(text):
                sent_lengths.append(len(tokenize(s)))

        unique_words = len(set(all_tokens))
        total_words = len(all_tokens)
        ttr = unique_words / total_words if total_words else 0

        unique_bi = len(set(all_bigrams))
        total_bi = len(all_bigrams)
        btr = unique_bi / total_bi if total_bi else 0

        mean_sent_len = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0

        print(f"Q{qi+1:>2} {ttr:>8.4f} {btr:>10.4f} {mean_sent_len:>12.1f} {unique_words:>8} {total_words:>8} {unique_bi:>8} {total_bi:>8}")

    print()


# ============================================================
# 2. STRUCTURAL TEMPLATES
# ============================================================
def analyze_structure(quintiles: list[list[dict]]):
    print("=" * 70)
    print("2. STRUCTURAL TEMPLATES (propose phase)")
    print("=" * 70)

    for qi, q in enumerate(quintiles):
        propose_texts = [strip_answer_tag(t) for t in extract_turns(q, phase="propose")]
        total = len(propose_texts)

        # Count numbered lists
        numbered_list_count = 0
        for text in propose_texts:
            if re.search(r"^\s*\d+\.\s", text, re.MULTILINE):
                numbered_list_count += 1

        # Count bold headers like **Header:**
        bold_header_count = 0
        for text in propose_texts:
            if re.search(r"\*\*[^*]+\*\*", text):
                bold_header_count += 1

        # Count direct-answer openers
        direct_answer_count = 0
        direct_patterns = [
            r"^the\s+(correct\s+)?answer\s+is",
            r"^the\s+best\s+(answer|explanation|option)",
            r"^the\s+most\s+(plausible|likely|accurate)",
            r"^i\s+(believe|argue|contend)\s+that",
        ]
        for text in propose_texts:
            prose = strip_answer_tag(text).strip()
            first_line = prose.split("\n")[0].strip() if prose else ""
            for pat in direct_patterns:
                if re.search(pat, first_line.lower()):
                    direct_answer_count += 1
                    break

        # First sentence clustering — extract first sentences and find most common prefixes
        first_sents = []
        for text in propose_texts:
            prose = strip_answer_tag(text).strip()
            if prose:
                s = sentences(prose)
                if s:
                    first_sents.append(s[0])

        # Get first 6 words as "opener template"
        opener_counter = Counter()
        for s in first_sents:
            words = tokenize(s)[:6]
            opener_counter[" ".join(words)] += 1

        top_openers = opener_counter.most_common(5)

        # Compute opener concentration: what fraction of responses use a top-10 opener?
        top10_count = sum(c for _, c in opener_counter.most_common(10))

        print(f"\nQ{qi+1} (n={total}):")
        print(f"  Numbered lists:      {numbered_list_count:>5} ({100*numbered_list_count/total:.1f}%)")
        print(f"  Bold headers:        {bold_header_count:>5} ({100*bold_header_count/total:.1f}%)")
        print(f"  Direct-answer opener: {direct_answer_count:>4} ({100*direct_answer_count/total:.1f}%)")
        print(f"  Top-10 opener conc:  {top10_count:>5} ({100*top10_count/total:.1f}%)")
        print(f"  Top 5 openers:")
        for opener, count in top_openers:
            print(f"    [{count:>3}x] \"{opener}...\"")

    print()


# ============================================================
# 3. ANSWER DISTRIBUTION ENTROPY
# ============================================================
def analyze_answer_entropy(quintiles: list[list[dict]]):
    print("=" * 70)
    print("3. ANSWER DISTRIBUTION ENTROPY")
    print("=" * 70)
    print(f"{'Q':>3} {'Entropy':>8} {'Unique':>7} {'Total':>7} {'Top1%':>7} {'Top3%':>7}")
    print("-" * 50)

    for qi, q in enumerate(quintiles):
        # Collect answers from all propose turns
        answers = []
        for ep in q:
            for turn in ep["transcript"]:
                if turn.get("phase") == "propose":
                    ans = extract_answer(turn["text"])
                    if ans:
                        # Normalize: lowercase, strip
                        answers.append(ans.lower().strip())

        counter = Counter(answers)
        total = len(answers)
        unique = len(counter)
        entropy = shannon_entropy(counter)

        top1_pct = counter.most_common(1)[0][1] / total * 100 if total else 0
        top3_count = sum(c for _, c in counter.most_common(3))
        top3_pct = top3_count / total * 100 if total else 0

        print(f"Q{qi+1:>2} {entropy:>8.3f} {unique:>7} {total:>7} {top1_pct:>6.1f}% {top3_pct:>6.1f}%")

    # Also show per-quintile top answers
    print("\nTop 5 answers per quintile:")
    for qi, q in enumerate(quintiles):
        answers = []
        for ep in q:
            for turn in ep["transcript"]:
                if turn.get("phase") == "propose":
                    ans = extract_answer(turn["text"])
                    if ans:
                        answers.append(ans.lower().strip())
        counter = Counter(answers)
        total = len(answers)
        print(f"\n  Q{qi+1}:")
        for ans, count in counter.most_common(5):
            truncated = ans[:80] + "..." if len(ans) > 80 else ans
            print(f"    [{count:>4}, {100*count/total:>5.1f}%] {truncated}")

    print()


# ============================================================
# 4. ARGUMENT STRATEGY EVOLUTION (qualitative)
# ============================================================
def analyze_strategies(quintiles: list[list[dict]]):
    print("=" * 70)
    print("4. ARGUMENT STRATEGY EVOLUTION (critique phase)")
    print("=" * 70)

    # Quantitative rhetorical move detection
    move_patterns = {
        "attack_flaw": r"(flaw|error|incorrect|wrong|mistake|fallac|mischaracter|misrepresent|mislead)",
        "defend_position": r"(my (argument|position|answer)|i maintain|i stand by|as i (stated|argued))",
        "concede": r"(i (concede|acknowledge|agree|accept)|fair point|valid (point|criticism|concern))",
        "redirect": r"(however|but (the )?real|the (key|central|crucial|important) (issue|point|question)|more importantly|instead)",
        "appeal_evidence": r"(evidence|data|stud(y|ies)|research|experiment|observation|documented|demonstrated|shown)",
        "appeal_authority": r"(expert|scientist|researcher|literature|published|peer.reviewed|journal|well.established|widely (accepted|recognized))",
        "enumerate_points": r"^\s*\d+\.\s+\*\*",
        "bold_structure": r"\*\*[^*]{5,}\*\*",
    }

    print(f"\n{'Move':<20}", end="")
    for qi in range(len(quintiles)):
        print(f" {'Q'+str(qi+1):>7}", end="")
    print("  (% of critique responses)")
    print("-" * 70)

    for move_name, pattern in move_patterns.items():
        print(f"{move_name:<20}", end="")
        for qi, q in enumerate(quintiles):
            critique_texts = extract_turns(q, phase="critique")
            total = len(critique_texts)
            if move_name in ("enumerate_points", "bold_structure"):
                hits = sum(1 for t in critique_texts if re.search(pattern, t, re.MULTILINE))
            else:
                hits = sum(1 for t in critique_texts if re.search(pattern, t, re.IGNORECASE))
            pct = 100 * hits / total if total else 0
            print(f" {pct:>6.1f}%", end="")
        print()

    # Print 10 sample critiques from Q1 and Q5 for qualitative comparison
    for qi_idx, qi_label in [(0, "Q1 (earliest)"), (-1, f"Q{len(quintiles)} (latest)")]:
        q = quintiles[qi_idx]
        critiques = extract_turns(q, phase="critique")
        print(f"\n{'='*60}")
        print(f"SAMPLE CRITIQUES — {qi_label} (10 samples)")
        print(f"{'='*60}")
        import random
        random.seed(42)
        samples = random.sample(critiques, min(10, len(critiques)))
        for si, sample in enumerate(samples):
            print(f"\n--- Sample {si+1} ---")
            # Print full text, no truncation
            print(sample)
            print()

    print()


# ============================================================
# 5. COPY-PASTE DETECTION (4-gram overlap between debaters)
# ============================================================
def analyze_copypaste(quintiles: list[list[dict]]):
    print("=" * 70)
    print("5. COPY-PASTE DETECTION (4-gram overlap in critique phase)")
    print("=" * 70)
    print("For each episode with critiques from both sides, compute fraction of")
    print("B's critique 4-grams that appear in A's critique.")
    print()
    print(f"{'Q':>3} {'MeanOverlap':>12} {'MedianOverlap':>14} {'P90':>8} {'P99':>8} {'N_episodes':>11}")
    print("-" * 60)

    for qi, q in enumerate(quintiles):
        overlaps = []
        for ep in q:
            a_critiques = []
            b_critiques = []
            for turn in ep["transcript"]:
                if turn.get("phase") == "critique":
                    if turn["role"] == "debater_a":
                        a_critiques.append(turn["text"])
                    elif turn["role"] == "debater_b":
                        b_critiques.append(turn["text"])

            if not a_critiques or not b_critiques:
                continue

            # Combine all A critique text, compute 4-grams
            a_text = " ".join(a_critiques)
            b_text = " ".join(b_critiques)

            a_tokens = tokenize(a_text)
            b_tokens = tokenize(b_text)

            a_4grams = set(ngrams(a_tokens, 4))
            b_4grams = ngrams(b_tokens, 4)

            if not b_4grams:
                continue

            b_4gram_set = set(b_4grams)
            overlap_count = sum(1 for g in b_4grams if g in a_4grams)
            overlap_frac = overlap_count / len(b_4grams)
            overlaps.append(overlap_frac)

        if not overlaps:
            print(f"Q{qi+1:>2} {'N/A':>12}")
            continue

        overlaps.sort()
        mean_ov = sum(overlaps) / len(overlaps)
        median_ov = overlaps[len(overlaps) // 2]
        p90 = overlaps[int(0.9 * len(overlaps))]
        p99 = overlaps[int(0.99 * len(overlaps))]

        print(f"Q{qi+1:>2} {mean_ov:>12.4f} {median_ov:>14.4f} {p90:>8.4f} {p99:>8.4f} {len(overlaps):>11}")

    # Also find the worst offenders
    print("\nTop 10 highest-overlap episodes (across all data):")
    all_episodes = []
    for ep_idx, ep in enumerate(sum(quintiles, [])):
        a_critiques = []
        b_critiques = []
        for turn in ep["transcript"]:
            if turn.get("phase") == "critique":
                if turn["role"] == "debater_a":
                    a_critiques.append(turn["text"])
                elif turn["role"] == "debater_b":
                    b_critiques.append(turn["text"])

        if not a_critiques or not b_critiques:
            continue

        a_text = " ".join(a_critiques)
        b_text = " ".join(b_critiques)
        a_tokens = tokenize(a_text)
        b_tokens = tokenize(b_text)
        a_4grams = set(ngrams(a_tokens, 4))
        b_4grams = ngrams(b_tokens, 4)

        if not b_4grams:
            continue

        overlap_count = sum(1 for g in b_4grams if g in a_4grams)
        overlap_frac = overlap_count / len(b_4grams)
        all_episodes.append((overlap_frac, ep_idx, ep.get("debate_id", "?")))

    all_episodes.sort(reverse=True)
    for frac, idx, did in all_episodes[:10]:
        print(f"  [{frac:.4f}] episode {idx}, debate_id={did}")

    print()


# ============================================================
# BONUS: Response length over time
# ============================================================
def analyze_lengths(quintiles: list[list[dict]]):
    print("=" * 70)
    print("BONUS: RESPONSE LENGTH OVER TRAINING")
    print("=" * 70)
    print(f"{'Q':>3} {'ProposeLen':>11} {'CritiqueLen':>12} {'ProposeStd':>11} {'CritiqueStd':>12}")
    print("-" * 55)

    for qi, q in enumerate(quintiles):
        propose_lens = [len(tokenize(strip_answer_tag(t))) for t in extract_turns(q, phase="propose")]
        critique_lens = [len(tokenize(strip_answer_tag(t))) for t in extract_turns(q, phase="critique")]

        p_mean = sum(propose_lens) / len(propose_lens) if propose_lens else 0
        c_mean = sum(critique_lens) / len(critique_lens) if critique_lens else 0

        p_std = (sum((x - p_mean) ** 2 for x in propose_lens) / len(propose_lens)) ** 0.5 if propose_lens else 0
        c_std = (sum((x - c_mean) ** 2 for x in critique_lens) / len(critique_lens)) ** 0.5 if critique_lens else 0

        print(f"Q{qi+1:>2} {p_mean:>11.1f} {c_mean:>12.1f} {p_std:>11.1f} {c_std:>12.1f}")

    print()


def main():
    print(f"Loading episodes from {EPISODES_PATH}...")
    episodes = load_episodes(EPISODES_PATH)
    print(f"Loaded {len(episodes)} episodes.\n")

    quintiles = split_quintiles(episodes, 5)
    print(f"Quintile sizes: {[len(q) for q in quintiles]}")
    print()

    analyze_vocabulary(quintiles)
    analyze_structure(quintiles)
    analyze_answer_entropy(quintiles)
    analyze_lengths(quintiles)
    analyze_copypaste(quintiles)
    analyze_strategies(quintiles)


if __name__ == "__main__":
    main()
