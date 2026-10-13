"""Text and image decision API, with optional MLX-VLM imports kept lazy."""

from .engine import MLXDecisionEngine


class _ImageBackend:
    def __init__(self, backend, images):
        self.backend, self.images = backend, images

    def score(self, request, *, cache_key=None, use_cache=True):
        return self.backend.score(
            request, images=self.images, cache_key=cache_key, use_cache=use_cache,
        )


class MLXVLMDecisionEngine(MLXDecisionEngine):
    """Decide with local images using the same typed result and rejection rules.

    Images belong to each call, never to mutable shared backend state. Supply
    local paths or PIL images; an empty sequence performs a text-only decision.
    Prefix reuse is disabled, including when ``use_cache=True`` is requested.
    """

    def __init__(self, model: str, *, margin_threshold=0.0, **backend_options):
        from .backends.mlx_vlm import MLXVLMBackend

        super().__init__(
            backend=MLXVLMBackend(model, **backend_options), margin_threshold=margin_threshold,
        )

    def decide(self, request, *, images=(), use_cache=True, cache_key=None):
        engine = MLXDecisionEngine(
            backend=_ImageBackend(self.backend, images), margin_threshold=self.margin_threshold,
        )
        return engine.decide(request, use_cache=use_cache, cache_key=cache_key)
