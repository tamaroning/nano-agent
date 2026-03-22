//! Math expression evaluation via [`meval`](https://github.com/rekka/meval-rs) and I/O schemas for orchestration.

use genai::chat::Tool as GenaiTool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Tool input: a single expression string (see `meval` docs for supported syntax).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalculatorInput {
    /// Mathematical expression using `+`, `-`, `*`, `/`, `%`, `^`, functions like `sin`, `sqrt`, constants `pi`, `e`, etc.
    #[schemars(
        description = "Mathematical expression to evaluate (meval syntax: + - * / % ^, sin/cos/sqrt/ln/exp, pi, e, …)"
    )]
    pub expression: String,
}

/// Tool output: numeric result as `f64`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalculatorOutput {
    /// Evaluated value
    pub value: f64,
}

/// Errors from parsing or evaluating an expression.
#[derive(Debug, thiserror::Error)]
pub enum CalculatorError {
    #[error("expression must not be empty")]
    EmptyExpression,
    #[error("failed to evaluate expression: {0}")]
    Eval(#[from] meval::Error),
}

/// Stateless evaluator wrapping `meval::eval_str`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Calculator;

impl Calculator {
    pub fn new() -> Self {
        Self
    }

    /// Metadata for LLM function calling (e.g. `genai::ChatRequest::with_tools`).
    pub fn as_genai_tool() -> GenaiTool {
        let schema: Value = serde_json::to_value(schemars::schema_for!(CalculatorInput))
            .unwrap_or_else(|_| Value::Object(Default::default()));
        GenaiTool::new("calculator")
            .with_description(
                "Evaluate a mathematical expression to a number. Use for arithmetic, roots, trig, logs, powers, min/max, etc.",
            )
            .with_schema(schema)
    }

    /// Parses and evaluates `input.expression` with default `meval` context (built-in functions and constants).
    pub fn run(&self, input: CalculatorInput) -> Result<CalculatorOutput, CalculatorError> {
        let expr = input.expression.trim();
        if expr.is_empty() {
            return Err(CalculatorError::EmptyExpression);
        }
        let value = meval::eval_str(expr)?;
        Ok(CalculatorOutput { value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_arithmetic() {
        let c = Calculator::new();
        let out = c
            .run(CalculatorInput {
                expression: "1 + 2 * 3".into(),
            })
            .unwrap();
        assert!((out.value - 7.0).abs() < 1e-9);
    }

    #[test]
    fn builtins() {
        let c = Calculator::new();
        let out = c
            .run(CalculatorInput {
                expression: "sqrt(4) + sin(pi/2)".into(),
            })
            .unwrap();
        assert!((out.value - 3.0).abs() < 1e-9);
    }

    #[test]
    fn empty_rejected() {
        let c = Calculator::new();
        assert!(matches!(
            c.run(CalculatorInput {
                expression: "   ".into(),
            }),
            Err(CalculatorError::EmptyExpression)
        ));
    }
}
