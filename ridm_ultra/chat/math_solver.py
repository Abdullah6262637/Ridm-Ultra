"""Native Symbolic Arithmetic and Math Solver for RIDM Ultra."""
from __future__ import annotations

import ast
import logging
import math
import re
from typing import Optional

logger = logging.getLogger(__name__)

_DIGIT_PATTERN = re.compile(r'\d')
_OP_PATTERN = re.compile(r'(-?\d+(?:\.\d+)?)\s*([\x27a-zçğıöşü\s\+\-\*\/\:]+?)\s*(-?\d+(?:\.\d+)?)(?:\s*([\x27a-zçğıöşü\s]*))?')
_RAW_EXPR_CLEAN = re.compile(r'[^0-9\+\-\*\/\(\)\.\s\^]')

class SafeMathEvaluator(ast.NodeVisitor):
    """Safe AST evaluator for mathematical expressions."""

    ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load
    }

    ALLOWED_FUNCTIONS = {
        "abs": abs, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "pow": math.pow, "round": round
    }

    def visit(self, node):
        if type(node) not in self.ALLOWED_NODES:
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        return super().visit(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Non-numeric constant")

    def visit_Num(self, node):
        return node.n

    def visit_BinOp(self, node):
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
                raise ZeroDivisionError("Division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError("Division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")

    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in self.ALLOWED_FUNCTIONS:
            raise ValueError("Unsupported function call")
        func = self.ALLOWED_FUNCTIONS[node.func.id]
        args = [self.visit(arg) for arg in node.args]
        return func(*args)


def evaluate_math(text: str) -> Optional[str]:
    """Parse and evaluate math expressions in Turkish or standard notation."""
    if not text or not isinstance(text, str):
        return None

    norm_text = text.lower().strip()

    # Must contain numbers or explicit arithmetic keywords
    if not _DIGIT_PATTERN.search(norm_text) and not any(k in norm_text for k in ["artı", "eksi", "çarp", "böl", "topla", "farkı"]):
        return None

    # Match patterns like "3 ile 5 i çarparsan", "10 artı 5", "20 bölü 4", "15 eksi 7"
    m = _OP_PATTERN.search(norm_text)

    # Check for direct numeric expressions (e.g. "3 * 5", "25 + 14", "100 / 4", "2 ^ 10")
    raw_expr = _RAW_EXPR_CLEAN.sub('', norm_text).strip()
    if raw_expr and len(raw_expr) >= 3 and any(op in raw_expr for op in ['+', '-', '*', '/', '^', '**']):
        try:
            tree = ast.parse(raw_expr.replace('^', '**'), mode='eval')
            val = SafeMathEvaluator().visit(tree)
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            return f"Matematiksel İfade Sonucu: **{raw_expr} = {val}**"
        except Exception as e:
            logger.debug(f"Failed to evaluate explicit math expression '{raw_expr}': {e}")

    # Check Turkish word-based math queries
    if m:
        num1_str = m.group(1)
        middle_op = m.group(2).strip()
        num2_str = m.group(3)
        trailing = (m.group(4) or "").strip()
        verb = f"{middle_op} {trailing}".strip()

        try:
            n1 = float(num1_str) if '.' in num1_str else int(num1_str)
            n2 = float(num2_str) if '.' in num2_str else int(num2_str)

            if any(k in verb for k in ["çarp", "çarpımı", "çarpması", "çarpılırsa", "çarparsan", "*", "x"]):
                res = n1 * n2
                if isinstance(res, float) and res.is_integer():
                    res = int(res)
                return f"{n1} × {n2} = **{res}** eder."
            if any(k in verb for k in ["topla", "toplamı", "toplaması", "ekle", "artı", "+"]):
                res = n1 + n2
                if isinstance(res, float) and res.is_integer():
                    res = int(res)
                return f"{n1} + {n2} = **{res}** eder."
            if any(k in verb for k in ["çıkar", "farkı", "eksi", "çıkarırsan", "-"]):
                res = n1 - n2
                if isinstance(res, float) and res.is_integer():
                    res = int(res)
                return f"{n1} - {n2} = **{res}** eder."
            if any(k in verb for k in ["böl", "bölümü", "bölmesi", "bölünürse", "bölersen", "/", ":"]):
                if n2 == 0:
                    return "Sıfıra bölme hatası: Bir sayı 0'a bölünemez."
                res = n1 / n2
                if isinstance(res, float) and res.is_integer():
                    res = int(res)
                return f"{n1} ÷ {n2} = **{res}** eder."
        except Exception as e:
            logger.debug(f"Failed to evaluate word-based math expression '{verb}': {e}")

    return None
