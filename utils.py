import random
from collections import defaultdict
import pickle


def parenthetic_contents(string):
    """Generate parenthesized contents in string."""
    stack = []
    for i, c in enumerate(string):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            start = stack.pop()
            yield string[start : i + 1]


def outer_parenthetic_contents(string):
    """Generate outer parenthesized contents in string."""
    stack = []
    for i, c in enumerate(string):
        if c == "(":
            stack.append(i)
        elif c == ")" and stack:
            start = stack.pop()
            depth = len(stack)
            if depth == 0:
                yield string[start : i + 1]


def shuffle_tree(tree, p=0.5):
    children = tuple(outer_parenthetic_contents(tree[1:-1]))
    if len(children) < 2:
        return tree
    if random.random() > p:
        cl, cr = children
    else:
        cr, cl = children
    return "({} {} {})".format(
        tree[1], shuffle_tree(cl, p), shuffle_tree(cr, p)
    )


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def defaultdict2dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict2dict(v) for k, v in d.items()}
    return d


save_pickle = lambda fname, data: pickle.dump(
    data, open(str(RESULTS_DIR / fname), "wb")
)
load_pickle = lambda fname: pickle.load(open(str(RESULTS_DIR / fname), "rb"))
