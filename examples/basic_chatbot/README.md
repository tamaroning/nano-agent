# Basic Chatbot

## Overview
A minimal interactive chat agent. It reads messages from standard input and returns model responses.

## How to run
Set an API key and run from the repository root.

```bash
export OPENAI_API_KEY=your_key
cargo run -p basic_chatbot
```

You can use `GEMINI_API_KEY` instead of `OPENAI_API_KEY`. Type `/exit` or `/quit` to leave the chat.
