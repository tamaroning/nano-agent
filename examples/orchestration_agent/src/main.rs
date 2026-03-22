//! Orchestrator example: structured routing to search vs calculator.
use nano_agent::{AgentConfig, BasicNanoAgent, NanoAgent, context::SystemPromptGenerator};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Input for the web search tool (factual / lookup queries).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SearchToolInput {
    /// Search queries to execute
    #[schemars(description = "Search queries to execute")]
    queries: Vec<String>,
}

/// Input for the calculator tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct CalculatorToolInput {
    /// Mathematical expression to evaluate
    #[schemars(description = "Mathematical expression to evaluate")]
    expression: String,
}

/// Which tool to run and its parameters.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum Tool {
    Search(SearchToolInput),
    Calculator(CalculatorToolInput),
}

/// Orchestrator structured output: reasoning + one tool branch.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct OrchestratorOutput {
    /// Why this tool was selected
    #[schemars(description = "Why this tool was selected")]
    reasoning: String,
    /// Selected tool and its parameters
    #[schemars(description = "Selected tool and its parameters")]
    tool: Tool,
}

/// User message to the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct OrchestratorInput {
    /// User's question or request
    #[schemars(description = "User's question or request")]
    query: String,
}

async fn process_query(
    agent: &mut BasicNanoAgent<OrchestratorInput, OrchestratorOutput>,
    query: &str,
) -> Result<(), nano_agent::AgentError> {
    let result = agent
        .run(OrchestratorInput {
            query: query.to_string(),
        })
        .await?;

    println!("Reasoning: {}", result.reasoning);

    match result.tool {
        Tool::Search(p) => {
            println!("Using Search with queries: {:?}", p.queries);
            // let search_results = search_tool.run(p);
        }
        Tool::Calculator(p) => {
            println!("Using Calculator with: {}", p.expression);
            // let calc_result = calculator_tool.run(p);
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

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
            "gpt-4o-mini".to_string()
        } else {
            "gemini-2.5-flash-lite".to_string()
        }
    });

    let system_prompt = SystemPromptGenerator::new()
        .with_background(vec![
            "You are an orchestrator that routes queries to appropriate tools.".to_string(),
            "Use search for factual questions, current events, or lookups.".to_string(),
            "Use calculator for mathematical expressions and computations.".to_string(),
        ])
        .with_output_instructions(vec![
            "Analyze the query to determine the best tool.".to_string(),
            "Provide clear reasoning for your choice.".to_string(),
            "Format parameters correctly for the selected tool.".to_string(),
            "For tool_parameters, use exactly one key: \"search\" (object with queries) or \"calculator\" (object with expression)."
                .to_string(),
        ]);

    let mut orchestrator: BasicNanoAgent<OrchestratorInput, OrchestratorOutput> =
        BasicNanoAgent::new(AgentConfig::new(&model).with_system_prompt_generator(system_prompt));

    if let Err(e) = process_query(&mut orchestrator, "What is the capital of France?").await {
        eprintln!("orchestrator error: {e}");
    }

    if let Err(e) = process_query(&mut orchestrator, "Calculate 15% of 250").await {
        eprintln!("orchestrator error: {e}");
    }
}
