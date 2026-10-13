"""Command-line entry point; help does not load a model."""

from jev_mlx.cli import main as main

if __name__ == "__main__":
    raise SystemExit(main())
