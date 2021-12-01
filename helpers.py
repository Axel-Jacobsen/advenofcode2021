from typing import Any, Callable


def quantify(d, pred=bool):
    return sum(pred(dd) for dd in d)


def process_inputs(day: str, convert: Callable[str, Any]):
    with open(f"inputs/d{day}.txt") as f:
        data = [convert(s) for s in f.read().strip().split("\n") if s != ""]

    return data
