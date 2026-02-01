import re
from typing import List
from math_verify import parse, verify


THINKING_TAG = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_boxed(text: str):
    if match := re.search(r"\\boxed\{", text):
        i = match.end()
        depth = 1
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i:j]
    return None


def thinking_format_reward(completions: List[str], **kwargs):
    rewards = []
    for comp in completions:
        rewards.append(0.5 * bool(THINKING_TAG.search(comp)))
    return rewards


def answer_format_reward(completions: List[str], **kwargs):
    rewards = []
    for comp in completions:
        rewards.append(0.5 * bool(ANSWER_TAG.search(comp)))
    return rewards


def parse_ans(comp: str):
    if extracted_ans := ANSWER_TAG.search(comp):
        comp = extracted_ans.group(1).strip()
    if boxed_ans := extract_boxed(comp):
        comp = boxed_ans
    return comp


def correctness_reward(completions: List[str], answer: List[str], **kwargs):
    rewards = []
    for comp, ans in zip(completions, answer):
        gold = ans
        r = 0.0
        comp = parse_ans(comp)
        parsed_comp = parse(f"${comp}$")
        parsed_gold = parse(f"${gold}$")
        r += 3.0 * verify(parsed_gold, parsed_comp)
        rewards.append(r)
    return rewards


def countdown_format_reward(completions: List[str], target: List[int], nums: List[List[int]], **kwargs):
    rewards = []
    for comp, target, nums in zip(completions, target, nums):
        r = 0.0
        if extracted_ans := ANSWER_TAG.search(comp):
            eq_text = extracted_ans.group(1).strip()
            if "=" in eq_text:
                s = eq_text.split("=")
                if len(s) > 2:
                    r -= 0.5
                lhs, rhs = s[0].strip(), s[-1].strip()
                try:
                    if re.match(r"^\d+$", rhs) and int(rhs) == target:
                        r += 1 / 3
                    if re.match(r"^[\d\s+\-*/()]+$", lhs):
                        r += 1 / 3
                    used_nums = [int(num) for num in re.findall(r"\d+", lhs)]
                    if sorted(used_nums) == sorted(nums):
                        r += 1 / 3
                except Exception as e:
                    print(f"Error parsing {eq_text}: {e}")
                    r = None
        rewards.append(r)
    return rewards


def verify_equation(equation: str, target: int, nums: List[int]):
    if "=" not in equation:
        return False
    lhs = equation.split("=", 1)[0]
    if sorted(int(num) for num in re.findall(r"\d+", lhs)) != sorted(nums):
        return False
    parsed_lhs = parse(lhs)
    parsed_target = parse(str(target))
    return verify(parsed_target, parsed_lhs)


def countdown_correctness_reward(completions: List[str], target: List[int], nums: List[List[int]], **kwargs):
    rewards = []
    for comp, target, nums in zip(completions, target, nums):
        r = 0.0
        if extracted_ans := ANSWER_TAG.search(comp):
            eq_text = extracted_ans.group(1).strip()
            if verify_equation(eq_text, target, nums):
                r += 3.0
        rewards.append(r)
    return rewards


def sudoku_format_reward(completions: List[str], puzzle: List[str], **kwargs):
    rewards = []
    for comp, p in zip(completions, puzzle):
        r = 0.0
        if extracted_ans := ANSWER_TAG.search(comp):
            grid_text = extracted_ans.group(1).strip()
            if len(grid_text) == 16:
                r += 0.25
                if "".join(sorted(grid_text)) == "1111222233334444":
                    r += 0.25
                for i in range(16):
                    if p[i] != "0" and p[i] != grid_text[i]:
                        break
                else:
                    r += 0.5
        rewards.append(r)
    return rewards


def sudoku_correctness_reward(completions: List[str], puzzle: List[str], answer: List[str], **kwargs):
    rewards = []
    for comp, p, ans in zip(completions, puzzle, answer):
        comp = parse_ans(comp)
        comp = comp.zfill(16)[:16]
        empty = match = 0
        for i in range(16):
            if p[i] == "0":
                empty += 1
                if ans[i] == comp[i]:
                    match += 1
        rewards.append(3.0 * match / empty)
    return rewards