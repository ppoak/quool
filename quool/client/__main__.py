import argparse
from .terminal import Terminal


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--api_key", type=str, default="ollama")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--file", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    terminal = Terminal(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        prompt=args.prompt,
        file=args.file,
    ).run()
