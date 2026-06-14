"""Strict answer verifier for the 24-point game."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from game24_rl.solver import DEFAULT_TARGET, normalize_puzzle

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
VERIFIER_VERSION = "strict-ast-fraction-v1"


@dataclass(frozen=True)
class ExpressionEvaluation:
    """Result of strict arithmetic expression evaluation."""

    value: Fraction
    numbers: tuple[int, ...]


@dataclass(frozen=True)
class VerificationResult:
    """Verifier decision and diagnostic reason."""

    valid: bool
    reason: str
    expression: str | None = None
    value: Fraction | None = None
    numbers: tuple[int, ...] = ()


class UnsupportedExpressionError(ValueError):
    """Raised when an expression uses syntax outside the verifier contract."""


def extract_answer_expression(output: str) -> str:
    """Extracts the single expression inside ``<answer>...</answer>``.

    Args:
        output: Raw model output.

    Raises:
        ValueError: If the output violates the single-answer contract.
    """

    if output.count("<answer>") != 1 or output.count("</answer>") != 1:
        raise ValueError("expected exactly one <answer>...</answer> block")

    matches = list(ANSWER_PATTERN.finditer(output))
    if len(matches) != 1:
        raise ValueError("expected exactly one complete answer block")

    expression = matches[0].group(1).strip()
    if not expression:
        raise ValueError("answer expression is empty")
    if "<answer>" in expression or "</answer>" in expression:
        raise ValueError("nested answer tags are not allowed")
    return expression


def evaluate_arithmetic_expression(expression: str) -> ExpressionEvaluation:
    """Evaluates a strict integer arithmetic expression with rational math.

    Only binary ``+``, ``-``, ``*``, and ``/`` over integer literals are allowed.
    Names, calls, attributes, floats, unary operators, and every other Python
    syntax node are rejected.
    """

    tree = ast.parse(expression, mode="eval")
    evaluator = _ArithmeticEvaluator()
    value = evaluator.visit(tree)
    return ExpressionEvaluation(value=value, numbers=tuple(evaluator.numbers))


def verify_answer(
    output: str,
    puzzle: Sequence[int],
    target: int | Fraction = DEFAULT_TARGET,
) -> VerificationResult:
    """Verifies a model answer against one 24-point puzzle."""

    expected_numbers = normalize_puzzle(puzzle)
    try:
        expression = extract_answer_expression(output)
    except ValueError as exc:
        return VerificationResult(False, f"answer_contract:{exc}")

    try:
        evaluated = evaluate_arithmetic_expression(expression)
    except SyntaxError as exc:
        return VerificationResult(
            False, f"syntax_error:{exc.msg}", expression=expression
        )
    except ZeroDivisionError:
        return VerificationResult(False, "division_by_zero", expression=expression)
    except UnsupportedExpressionError as exc:
        return VerificationResult(
            False, f"unsupported_expression:{exc}", expression=expression
        )

    used_numbers = tuple(sorted(evaluated.numbers))
    if used_numbers != expected_numbers:
        return VerificationResult(
            False,
            "wrong_numbers",
            expression=expression,
            value=evaluated.value,
            numbers=used_numbers,
        )

    target_fraction = Fraction(target)
    if evaluated.value != target_fraction:
        return VerificationResult(
            False,
            "wrong_value",
            expression=expression,
            value=evaluated.value,
            numbers=used_numbers,
        )

    return VerificationResult(
        True,
        "ok",
        expression=expression,
        value=evaluated.value,
        numbers=used_numbers,
    )


class _ArithmeticEvaluator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.numbers: list[int] = []

    def visit_Expression(self, node: ast.Expression) -> Fraction:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> Fraction:
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError
            return left / right

        raise UnsupportedExpressionError(type(node.op).__name__)

    def visit_Constant(self, node: ast.Constant) -> Fraction:
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise UnsupportedExpressionError(f"constant {node.value!r}")
        self.numbers.append(node.value)
        return Fraction(node.value)

    def generic_visit(self, node: ast.AST) -> Fraction:
        raise UnsupportedExpressionError(type(node).__name__)
