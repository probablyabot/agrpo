from datasets import load_dataset
from rewards import extract_boxed

SYSTEM_PROMPT = r"""You will be given a math problem.
Think through the problem step by step, enclosing your reasoning process in <think> tags.
Provide your final answer in <answer> tags, with only the number or expression enclosed in \boxed{}, as follows:
<think>
...
</think>
<answer>
\boxed{...}
</answer>
"""

COUNTDOWN_SYSTEM_PROMPT = r"""You will be given a target number and a list of numbers. Create a mathematical expression using each number exactly once and the operations +, -, *, / (with parentheses allowed) to reach the target.
Think through different combinations systematically, exploring which operations might work. Enclose your reasoning process in <think> tags.
Format your response as:
<think>
reasoning process here
</think>
<answer>
final expression = target
</answer>
For example, if the target is 67 and the numbers are [17, 18, 5], a valid answer is:
<answer>
17 * 5 - 18 = 67
</answer>
"""

# Few-shot Sudoku examples taken from https://github.com/facebookresearch/SPG/blob/main/spg/data_utils.py
SUDOKU_SYSTEM_PROMPT = r"""You will be given a 4x4 Sudoku puzzle, provided as a 16-character string from left to right, top to bottom, with 0s representing empty cells.
Fill in the empty cells with the correct digits 1-4 such that each row, column, and 2x2 subgrid contains each digit exactly once.
Think through the puzzle step by step, enclosing your reasoning process in <think> tags.
Your final answer should be a complete 16-character string with only the digits 1-4 representing the filled-in grid.
Format your response as:
<think>
reasoning process here
</think>
<answer>
[16-character string with no spaces]
</answer>

Puzzle: 3014002020004130

<think>
Interpret puzzle as 4 rows of 4:
R1: 3 0 1 4
R2: 0 0 2 0
R3: 2 0 0 0
R4: 4 1 3 0

Fill easy singles:
R1 missing 2 → R1C2=2.
R4 missing 2 → R4C4=2.
Box D (R3-4,C3-4) then needs {1,4}; column4 can only accept 1 → R3C4=1, R3C3=4.
R3 now missing 3 → R3C2=3.
Column1 missing 1 → R2C1=1.
Column2 missing 4 → R2C2=4.
Last cell R2C4=3.

Final grid:
R1: 3 2 1 4
R2: 1 4 2 3
R3: 2 3 4 1
R4: 4 1 3 2
</think>
<answer>
3214142323414132
</answer>

Puzzle: 0000100420013142

<think>
Interpret puzzle as 4 rows of 4:
R1: 0 0 0 0
R2: 1 0 0 4
R3: 2 0 0 1
R4: 3 1 4 2

Fill easy singles:
Col1 missing 4 → R1C1=4.
Col4 missing 3 → R1C4=3.
Box A (R1-2,C1-2) missing {2,3} and R1 now needs {1,2} → R1C2=2, R2C2=3.
R1C3=1.
R2 now missing 2 → R2C3=2.
Col2 missing 4 → R3C2=4, then R3C3=3.

Final grid:
R1: 4 2 1 3
R2: 1 3 2 4
R3: 2 4 3 1
R4: 3 1 4 2
</think>
<answer>
4213132424313142
</answer>

Puzzle: 2001403002001420

<think>
Interpret puzzle as 4 rows of 4:
R1: 2 0 0 1
R2: 4 0 3 0
R3: 0 2 0 0
R4: 1 4 2 0

Fill easy singles:
R1 missing {3,4}; Col2 can't be 1 so R1C2=3 → R1C3=4.
R4 missing 3 → R4C4=3.
Col4 missing {2,4}; R2 must take 2 → R2C4=2 → R2C2=1.
Col1 missing 3 → R3C1=3.
Col3 missing 1 → R3C3=1 → R3C4=4.

Final grid:
R1: 2 3 4 1
R2: 4 1 3 2
R3: 3 2 1 4
R4: 1 4 2 3
</think>
<answer>
2341413232141423
</answer>
"""

DATASETS = {}

def register_dataset(alias):
    def decorator(func):
        DATASETS[alias] = func
        return func
    return decorator

@register_dataset("gsm_train")
def load_gsm_train():
    ds = load_dataset("gsm8k", "main", split="train")
    return ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": x["answer"].split("####")[1].strip(),
    }, load_from_cache_file=False)

@register_dataset("gsm_test")
def load_gsm_test():
    ds = load_dataset("madrylab/gsm8k-platinum", split="test")
    return ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": x["answer"].split("####")[1].strip(),
    }, load_from_cache_file=False)

@register_dataset("math_train")
def load_math_train():
    ds = load_dataset("qwedsacf/competition_math", split="train")
    # filter out test split problems
    test_probs = set(load_dataset("HuggingFaceH4/MATH-500", split="test")["problem"])
    ds = ds.filter(lambda x: x["problem"] not in test_probs)
    # filter out level 5 problems (too hard for LLaDA)
    ds = ds.filter(lambda x: x["level"][-1].isdigit() and int(x["level"][-1]) <= 4)
    return ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["problem"]},
        ],
        "answer": extract_boxed(x["solution"]) or "",
    }, load_from_cache_file=False).filter(lambda x: x["answer"] != "")

@register_dataset("math_test")
def load_math_test():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["problem"]},
        ],
        "answer": x["answer"],
    }, load_from_cache_file=False)

@register_dataset("countdown_train")
def load_countdown_train():
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    ds = ds.select(range(20000))
    ds = ds.filter(lambda x: len(x["nums"]) == 3)
    # filter out test split problems
    test_ds = load_dataset("json", data_files="countdown_cd3_test.jsonl")["train"]
    test_probs = set(map(lambda x: tuple(map(int, x.split(","))), test_ds["input"]))
    ds = ds.filter(lambda x: tuple(x["nums"]) not in test_probs)
    return ds.map(lambda x: {
        "prompt": [
            {"role": "system", "content": COUNTDOWN_SYSTEM_PROMPT},
            {"role": "user", "content": f"Target: {x['target']}\nNumbers: {x['nums']}"},
        ],
        "target": x["target"],
        "nums": x["nums"],
    }, load_from_cache_file=False)

@register_dataset("countdown_test")
def load_countdown_test():
    ds = load_dataset("json", data_files="countdown_cd3_test.jsonl")["train"]
    return ds.map(lambda x: {
        "nums": (nums := [int(i) for i in x["input"].split(",")]),
        "prompt": [
            {"role": "system", "content": COUNTDOWN_SYSTEM_PROMPT},
            {"role": "user", "content": f"Target: {x['output']}\nNumbers: {nums}"},
        ],
        "target": int(x["output"]),
    }, load_from_cache_file=False)

@register_dataset("sudoku_train")
def load_sudoku_train():
    ds = load_dataset("csv", data_files="train_sudoku_split_new.csv")["train"]
    ds = ds.select(range(20000))
    return ds.map(lambda x: {
        "puzzle": (puzzle := str(x['Puzzle']).zfill(16)),
        "prompt": [
            {"role": "system", "content": SUDOKU_SYSTEM_PROMPT},
            {"role": "user", "content": f"Puzzle: {puzzle}"},
        ],
        "answer": str(x["Solution"]),
    }, load_from_cache_file=False)

@register_dataset("sudoku_test")
def load_sudoku_test():
    ds = load_dataset("csv", data_files="test_sudoku_split_new.csv")["train"]
    return ds.map(lambda x: {
        "puzzle": (puzzle := str(x['Puzzle']).zfill(16)),
        "prompt": [
            {"role": "system", "content": SUDOKU_SYSTEM_PROMPT},
            {"role": "user", "content": f"Puzzle: {puzzle}"},
        ],
        "answer": str(x["Solution"]),
    }, load_from_cache_file=False)
