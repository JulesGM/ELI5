""" Generic Utils
"""
import faiss
import nlp
import numpy as np
import transformers

LOGGER = logging.getLogger(__name__)

def get_terminal_width(default=80):
    try:
        _, columns = os.popen('stty size', 'r').read().split()
    except ValueError as err:
        columns = default
    return int(columns)


def wrap(text, joiner, padding=4):
    wrapped = [line.strip() for line in 
               textwrap.wrap(text, get_terminal_width() - padding)]
    return joiner.join(wrapped)


def print_wrapped(text, first_line="   "):
    print(first_line + wrap(text, "\n   ", 3))


def print_iterable(iterable: Iterable, extra_return: bool = False) -> None:
    """ Assumes that there is an end to the iterable, or will run forever.
    """
    for line in iterable:
            print_wrapped(line, " - ")
            print("")
    
    if extra_return:
        print("")

def print_numbered_iterable(iterable: Iterable, 
                            extra_return: bool = False) -> None:
    """ Assumes that there is an end to the iterable, or will run forever.
    """
    for i, line in enumerate(iterable):
            print_wrapped(line, f"{i}. ")
            print("")
    
    if extra_return:
        print("")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_equal(a: Any, b: Any)  -> None:
    check(a == b, f"Expected arguments to be equal, got \n{a} != {b}.")


def check_type(obj: Any, type_: type) -> None:
    check(isinstance(obj, type_), 
          f"Failed isinstance. Expected type `{type_}`, got `{type(obj)}`.")


class ExpRunningAverage:
    def __init__(self, new_contrib_rate: float):
        self.new_contrib_rate = new_contrib_rate
        self.running_average = None

    def step(self, val):
        if self.running_average is None:
            self.running_average = val
        else:
            self.running_average = (self.new_contrib_rate * val + 
                (1 - self.new_contrib_rate) * self.running_average)
    
        return self.running_average


def zip_safe(*args):
    """Returns an exception if not all iterators finish simultaneously.
    The `zip` builtin just ends if any of the iterators stop. We want to know
    if one of the iterators unexpectedly finishes early (or late).
    """
    iterators = [iter(iterable) for iterable in args]
    while True:
        results = []
        done = set()
        for i, iterator in enumerate(iterators):
            try:
                results.append(next(iterator))
            except StopIteration:
                done.add(i)

        # We stop the infinite loop if any of the items are done
        if done:
            if not all([it_idx in done for it_idx in range(len(iterators))]):
                # Return an exception with an helpful error message
                sorted_done_w_hashes = "#" + ", #".join(map(str, sorted(done)))
                possible_iterators_w_hashes = "#" + ", #".join(
                    map(str, range(len(iterators))))

                raise RuntimeError(
                    f"Not all iterators were done:"
                    f"Only iterator(s) {sorted_done_w_hashes} " + 
                    f"out of {possible_iterators_w_hashes}")
            return

        yield results


def get_logging_module(target_name):
    possible_characters = string.ascii_letters[:26] | {"_"}
    with_dot = possible_characters | {"."}
    assert all(c in possible_characters for c in name), name
    results = []
    all_loggers = [logging.getLogger(name) for name in 
        logging.root.manager.loggerDict]

    for logger in all_loggers:
        name = logger.name.strip()
        if all(c in with_dot for c in name):
            parent = name.split(".")[0]
            if parent == target_name and name != target_name:
                results.append(name)

    return results


class LoggingFormatter():
    def __init__(self, logger):
        self.logger = logger

    def log_loading(self, message, color="blue"):
        self.logger.info(click.style(message, fg=color))
