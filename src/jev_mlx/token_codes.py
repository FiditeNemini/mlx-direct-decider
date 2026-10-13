"""Verify answer tokens in context, including SentencePiece boundary behavior."""

from dataclasses import replace


def continuation_choices(tokenizer, text, tokens, choices):
    """Retain codes, requiring exactly one round-trip token after the full prompt.

    SentencePiece may encode a standalone letter with a leading-space token but
    use its bare-letter token after an assistant delimiter (Mistral v0.3). Read
    the verified continuation ID, never the standalone ID in that case.
    """
    verified = []
    seen = set()
    for choice in choices:
        encoded = tokenizer.encode(text + choice.code, add_special_tokens=False)
        if len(encoded) != len(tokens) + 1 or encoded[:-1] != list(tokens):
            raise ValueError(f"Choice code {choice.code!r} is not a single-token continuation")
        token_id = encoded[-1]
        if token_id in seen or tokenizer.decode([token_id]) != choice.code:
            raise ValueError(f"Choice code {choice.code!r} does not round-trip in context")
        seen.add(token_id)
        verified.append(replace(choice, token_id=token_id))
    return tuple(verified)
