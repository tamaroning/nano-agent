# Orchestration Agent

## Overview
An example orchestrator that analyzes user queries and returns structured output to choose either a search tool or a calculator tool.

## How to run
Set an API key and run from the repository root.

```bash
export OPENAI_API_KEY=your_key
cargo run -p orchestrator
```

`GEMINI_API_KEY` is also supported. This example processes two sample queries on startup.
