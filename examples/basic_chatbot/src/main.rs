use std::io::{self, Write};

use console::Style;
use nano_agent::genai::chat::{ChatOptions, ReasoningEffort};
use nano_agent::{
    AgentConfig, BasicChatInputSchema, BasicChatOutputSchema, BasicNanoAgent, NanoAgent,
    context::ChatHistory,
};

#[tokio::main]
async fn main() {
    let has_api_key = std::env::var("OPENAI_API_KEY")
        .map(|k| !k.is_empty())
        .unwrap_or(false)
        || std::env::var("GEMINI_API_KEY")
            .map(|k| !k.is_empty())
            .unwrap_or(false);
    if !has_api_key {
        eprintln!("API key is not set. Set OPENAI_API_KEY or GEMINI_API_KEY (genai resolver).");
        std::process::exit(1);
    }

    let model = std::env::var("AGENT_MODEL").unwrap_or_else(|_| {
        if std::env::var("OPENAI_API_KEY")
            .map(|k| !k.is_empty())
            .unwrap_or(false)
        {
            "gpt-5-mini".to_string()
        } else {
            "gemini-2.5-flash-lite".to_string()
        }
    });

    let mut history = ChatHistory::new();
    let initial_message = BasicChatOutputSchema {
        chat_message: "Hello! How can I assist you today?".to_string(),
    };
    history
        .add_message_schema("assistant", &initial_message)
        .expect("seed history");

    let chat_options = if model.contains("gpt-5") || model.contains("o3") || model.contains("o4") {
        ChatOptions::default().with_reasoning_effort(ReasoningEffort::Low)
    } else {
        ChatOptions::default()
    };

    let mut agent: BasicNanoAgent = BasicNanoAgent::new(
        AgentConfig::new(&model)
            .with_chat_options(chat_options)
            .with_chat_history(history),
    );

    let agent_label = Style::new().green().bold();
    print!("{} ", agent_label.apply_to("Agent:"));
    println!("{}", agent_label.apply_to(&initial_message.chat_message));

    let you_style = Style::new().blue().bold();
    loop {
        print!("{}", you_style.apply_to("You: "));
        io::stdout().flush().ok();

        let mut user_input = String::new();
        if io::stdin().read_line(&mut user_input).is_err() {
            break;
        }
        let user_input = user_input.trim_end_matches(['\r', '\n']);

        match user_input.to_ascii_lowercase().as_str() {
            "/exit" | "/quit" => {
                println!("Exiting chat...");
                break;
            }
            _ => {}
        }

        let input_schema = BasicChatInputSchema {
            chat_message: user_input.to_string(),
        };

        let response = match agent.run(input_schema).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Agent error: {e}");
                continue;
            }
        };

        print!("{} ", agent_label.apply_to("Agent:"));
        println!("{}", agent_label.apply_to(&response.chat_message));
    }
}
