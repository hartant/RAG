"""Entry point for `python -m src`."""

import fire  # type: ignore[import-untyped]

from .cli import Cli


def main() -> None:
    """Run the command line interface."""
    fire.Fire(Cli)


if __name__ == "__main__":
    main()
