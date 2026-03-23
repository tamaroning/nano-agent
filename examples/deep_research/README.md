# Deep Research

## Overview
An interactive research CLI that combines search, webpage scraping, and answer generation. It runs new web searches when needed and updates its context.

## How to run
Set an API key and run from the repository root.

```bash
export OPENAI_API_KEY=your_key
cargo run -p deep_research
```

Optionally set `SEARXNG_URL` to use SearXNG as the search backend. If it is not set, DuckDuckGo is used. Type `/exit` or `/quit` to end the session.
