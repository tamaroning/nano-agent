# Nano Agent

Nano Agent is a Rust library for building AI agents, inspired by [Atomic Agents](https://github.com/BrainBlend-AI/atomic-agents)—with the same focus on staying small, modular, and easy to maintain without sacrificing developer experience.

- **Modular pieces** — agents, tools, and context stay separate so you can swap or extend one without untangling the rest.
- **Structured I/O Types** — define what you send and what you expect back in Rust; the model is guided to answer in that shape so you handle fields and values instead of parsing free-form text.  
- **Custom tools** — run your own logic from the workflow, plus optional ready-made tools when you want them.
- **Native Rust** — lighter and faster than typical Python-based agent setups.

## Example

```rust
use nano_agent::{
    AgentConfig, BasicChatInputSchema, BasicNanoAgent, NanoAgent,
    context::{ChatHistory, SystemPromptGenerator},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
struct CustomOutputSchema {
    #[schemars(description = "The response to the user's question")]
    chat_message: String,
    #[schemars(description = "Suggested follow-up questions for the user")]
    suggested_questions: Vec<String>,
}

#[tokio::main]
async fn main() {
    let mut agent = BasicNanoAgent::<BasicChatInputSchema, CustomOutputSchema>::new(
        AgentConfig::new("gemini-2.5-flash-lite")
            .with_system_prompt_generator(
                SystemPromptGenerator::new()
                    .with_background(vec![
                        "This assistant is knowledgeable, helpful, and suggests follow-up questions."
                            .to_string(),
                    ])
                    .with_steps(vec![
                        "Analyze the user's input to understand the context and intent."
                            .to_string(),
                        "Formulate a relevant and informative response.".to_string(),
                        "Generate 3 suggested follow-up questions for the user.".to_string(),
                    ])
                    .with_output_instructions(vec![
                        "Provide clear and concise information in response to user queries."
                            .to_string(),
                        "Conclude each response with 3 relevant suggested questions for the user."
                            .to_string(),
                    ]),
            )
            .with_chat_history(ChatHistory::new()),
    );

    let response: CustomOutputSchema = match agent
        .run(BasicChatInputSchema {
            chat_message: "Tell me about the Nano Agent framework".to_string(),
        })
        .await
    {
        Ok(response) => response,
        Err(error) => {
            eprintln!("Agent error: {}", error);
            return;
        }
    };
    println!("Agent: {}", response.chat_message);
    println!("Suggested questions:");
    for question in response.suggested_questions {
        println!("- {}", question);
    }
}
```

To run this example, add the following dependencies to your `Cargo.toml`:

```toml
[dependencies]
nano-agent = "0.1"
schemars = "1.0"
serde = { version = "1", features = ["derive"] }
serde_json = { version = "1" }
tokio = { version = "1", features = ["full"] }
```

## Tools

Built-in tools are optional and enabled via Cargo features:

| Tool | Feature flag | What it does |
| --- | --- | --- |
| `Calculator` | `calculator` | Evaluates math expressions (e.g. arithmetic, trig, logs, roots). |
| `SearxngSearch` | `searxng` | Runs web search via a SearXNG instance (`SEARXNG_URL`). |
| `DuckDuckGoSearch` | `duckduckgo_search` | Runs web search via DuckDuckGo. |
| `WebpageScraper` | `webpage_scraper` | Fetches a webpage and returns readable Markdown + metadata. |

Example:

```toml
nano-agent = { version = "0.1", features = ["calculator", "duckduckgo_search"] }
```


## Status

- [x] Structured I/O types
- [x] Chat history
- [x] Custom tools
- [x] Context providers
- [x] JSON mode
- [ ] Native function call
- [ ] MCP

