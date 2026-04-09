"""Extract key concepts from claims."""

import os
import json
import re


def extract_concepts(claims: list[dict], llm_client=None) -> list[str]:
    """Extract key concepts from verified claims.

    Args:
        claims: List of claim dicts with claim_text field.
        llm_client: Optional LLM client for extraction.

    Returns:
        List of concept strings.
    """
    if not claims:
        return []

    claim_texts = [c.get("claim_text", "") for c in claims]
    combined = "\n".join(claim_texts)

    # Try LLM extraction
    if llm_client:
        try:
            prompt = (
                "Extract the key concepts from these claims. "
                "Return ONLY a JSON array of short concept strings.\n\n"
                f"Claims:\n{combined}\n\nConcepts:"
            )
            response = llm_client.chat.completions.create(
                model=os.environ.get("LLM_MODEL", "llama3-8b-8192"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                concepts = json.loads(json_match.group())
                return [c for c in concepts if isinstance(c, str) and c.strip()]
        except Exception:
            pass

    # Rule-based fallback: extract significant words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "that", "this", "these", "those",
        "it", "its", "not", "no", "nor", "or", "and", "but", "if",
        "so", "very", "just", "also", "more", "most", "each", "every",
        "all", "any", "both", "few", "some", "such", "than", "too",
        "only", "own", "same", "when", "where", "which", "while",
    }

    concepts = set()
    for text in claim_texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        for word in words:
            if word not in stop_words:
                concepts.add(word)

    return sorted(concepts)[:20]
