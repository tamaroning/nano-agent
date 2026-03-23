//! Deep Research CLI.

use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use chrono::Utc;
use nano_agent::{
    AgentConfig, AgentError, BasicNanoAgent, NanoAgent,
    context::{ChatHistory, ContextProvider, SystemPromptGenerator},
    tools::{
        SearxngSearch, SearxngSearchInput, SearxngSearchOutput, WebpageScraper, WebpageScraperInput,
    },
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const WELCOME_MESSAGE: &str = "Welcome to Deep Research - your AI-powered research assistant! I can help you explore and \
    understand any topic through detailed research and interactive discussion.";

const STARTER_QUESTIONS: &[&str] = &[
    "Can you help me research the latest AI news?",
    "Who won the Nobel Prize in Physics this year?",
    "Where can I learn more about quantum computing?",
];

const CHOICE_DECISION_PROMPT: &str = "Should we perform a new web search? TRUE if we need new or updated information, FALSE if existing \
    context is sufficient. Consider: 1) Is the context empty? 2) Is the existing information relevant? \
    3) Is the information recent enough?";

const MAX_SCRAPE_URLS: usize = 3;
const MAX_MARKDOWN_CHARS_PER_PAGE: usize = 12_000;

#[derive(Clone)]
struct ContentItem {
    url: String,
    content: String,
}

struct CurrentDateContextProvider;

impl ContextProvider for CurrentDateContextProvider {
    fn title(&self) -> &str {
        "Current Date"
    }

    fn get_info(&self) -> String {
        let now = Utc::now().format("%Y-%m-%d %H:%M UTC");
        format!("Current date/time (UTC): {now}")
    }
}

struct ScrapedContentContextProvider {
    items: Arc<Mutex<Vec<ContentItem>>>,
}

impl ScrapedContentContextProvider {
    fn new(items: Arc<Mutex<Vec<ContentItem>>>) -> Self {
        Self { items }
    }
}

impl ContextProvider for ScrapedContentContextProvider {
    fn title(&self) -> &str {
        "Scraped Content"
    }

    fn get_info(&self) -> String {
        let items = self.items.lock().unwrap_or_else(|e| e.into_inner());
        if items.is_empty() {
            return "No web content has been scraped yet for this session.".to_string();
        }
        items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let body = truncate_chars(&item.content, MAX_MARKDOWN_CHARS_PER_PAGE);
                format!("--- Source {}: {} ---\n{body}", i + 1, item.url)
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct QueryAgentInput {
    #[schemars(
        description = "A detailed instruction or request to generate search engine queries for."
    )]
    instruction: String,
    #[schemars(description = "The number of search queries to generate.")]
    num_queries: u32,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct QueryAgentOutput {
    #[schemars(description = "Search-engine-style queries (keywords/operators, not long prose).")]
    queries: Vec<String>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct ChoiceAgentInput {
    #[schemars(description = "The user's latest message or question")]
    user_message: String,
    #[schemars(description = "Explanation of the type of decision to make")]
    decision_type: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct ChoiceAgentOutput {
    #[schemars(description = "Detailed explanation of the decision-making process")]
    reasoning: String,
    #[schemars(
        description = "TRUE if a new web search is needed, FALSE if existing context is enough"
    )]
    decision: bool,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct QuestionAnsweringInput {
    #[schemars(description = "The question to answer.")]
    question: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct QuestionAnsweringOutput {
    #[schemars(description = "The answer to the question.")]
    answer: String,
    #[schemars(
        description = "Topic-specific follow-up questions to deepen research (not generic small talk)."
    )]
    follow_up_questions: Vec<String>,
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    format!("{}… [truncated]", s.chars().take(max).collect::<String>())
}

fn pick_top_results(
    output: SearxngSearchOutput,
    limit: usize,
) -> Vec<nano_agent::tools::SearxngResultItem> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for r in output.results {
        if seen.insert(r.url.clone()) {
            out.push(r);
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn query_agent_config(model: &str) -> AgentConfig {
    AgentConfig::new(model).with_system_prompt_generator(
        SystemPromptGenerator::new()
            .with_background(vec![
                "You are an expert search engine query generator with a deep understanding of which \
                 queries will maximize the number of relevant results."
                    .to_string(),
            ])
            .with_steps(vec![
                "Analyze the given instruction to identify key concepts and aspects that need to be researched"
                    .to_string(),
                "For each aspect, craft a search query using appropriate search operators and syntax".to_string(),
                "Ensure queries cover different angles of the topic (technical, practical, comparative, etc.)"
                    .to_string(),
            ])
            .with_output_instructions(vec![
                "Return exactly the requested number of queries (see field num_queries in the user JSON)."
                    .to_string(),
                "Format each query like a search engine query, not a natural language question".to_string(),
                "Each query should be a concise string of keywords and operators".to_string(),
            ]),
    )
}

fn choice_agent_config(model: &str) -> AgentConfig {
    AgentConfig::new(model).with_system_prompt_generator(
        SystemPromptGenerator::new()
            .with_background(vec![
                "You are a decision-making agent that determines whether a new web search is needed to answer the user's question."
                    .to_string(),
                "Your primary role is to analyze whether the existing context contains sufficient, up-to-date information to answer the question."
                    .to_string(),
                "You must output a clear TRUE/FALSE decision - TRUE if a new search is needed, FALSE if existing context is sufficient."
                    .to_string(),
            ])
            .with_steps(vec![
                "1. Analyze the user's question to determine whether or not an answer warrants a new search".to_string(),
                "2. Review the \"Scraped Content\" section under EXTRA INFORMATION AND CONTEXT in the system prompt"
                    .to_string(),
                "3. Determine if existing information is sufficient and relevant".to_string(),
                "4. Make a binary decision: TRUE for new search, FALSE for using existing context".to_string(),
            ])
            .with_output_instructions(vec![
                "Your reasoning must clearly state WHY you need or don't need new information".to_string(),
                "If the web search context is empty or irrelevant, always decide TRUE for new search".to_string(),
                "If the question is time-sensitive, check the current date to ensure context is recent".to_string(),
                "For ambiguous cases, prefer to gather fresh information".to_string(),
                "Your decision must match your reasoning - don't contradict yourself".to_string(),
            ]),
    )
}

fn qa_agent_config(model: &str, history: ChatHistory) -> AgentConfig {
    AgentConfig::new(model)
        .with_system_prompt_generator(
            SystemPromptGenerator::new()
                .with_background(vec![
                    "You are an expert question answering agent focused on providing factual information and encouraging deeper topic exploration."
                        .to_string(),
                    "For general greetings or non-research questions, provide relevant questions about the system's capabilities and research functions."
                        .to_string(),
                ])
                .with_steps(vec![
                    "Analyze the question and identify the core topic".to_string(),
                    "Answer the question using available information, including the \"Scraped Content\" section when present"
                        .to_string(),
                    "For topic-specific questions, generate follow-up questions that explore deeper aspects of the same topic"
                        .to_string(),
                    "For general queries about the system, suggest questions about research capabilities and functionality"
                        .to_string(),
                ])
                .with_output_instructions(vec![
                    "Answer in a direct, informative manner".to_string(),
                    "NEVER generate generic conversational follow-ups like 'How are you?' or 'What would you like to know?'"
                        .to_string(),
                    "For topic questions, follow-up questions MUST be about specific aspects of that topic".to_string(),
                    "For system queries, follow-up questions should be about specific research capabilities".to_string(),
                ]),
        )
        .with_chat_history(history)
}

async fn perform_search_and_update_context(
    user_message: &str,
    scraped: &Arc<Mutex<Vec<ContentItem>>>,
    query_agent: &mut BasicNanoAgent<QueryAgentInput, QueryAgentOutput>,
    searx: &SearxngSearch,
    scraper: &WebpageScraper,
) -> Result<(), AgentError> {
    println!("\n🤔 Analyzing your question to generate relevant search queries...");
    let query_out = query_agent
        .run(QueryAgentInput {
            instruction: user_message.to_string(),
            num_queries: 3,
        })
        .await?;

    println!("\n🔍 Generated search queries:");
    for (i, q) in query_out.queries.iter().enumerate() {
        println!("  {}. {}", i + 1, q);
    }

    println!("\n🌐 Searching the web via SearXNG...");
    let search_input = SearxngSearchInput {
        queries: query_out.queries.clone(),
        category: None,
    };
    let search_results = searx
        .run(search_input)
        .await
        .map_err(|e| AgentError::RequestFailed(format!("SearXNG: {e}")))?;

    let top = pick_top_results(search_results, MAX_SCRAPE_URLS);
    println!("\n📑 Found relevant web pages:");
    for (i, r) in top.iter().enumerate() {
        println!("  {}. {} — {}", i + 1, r.title, r.url);
    }

    println!("\n📥 Extracting content from web pages...");
    let mut new_items = Vec::with_capacity(top.len());
    for r in &top {
        let out = scraper
            .run(WebpageScraperInput {
                url: r.url.clone(),
                include_links: true,
            })
            .await;
        if let Some(err) = &out.error {
            eprintln!("  (warn) {} — {}", r.url, err);
        }
        new_items.push(ContentItem {
            url: r.url.clone(),
            content: out.content,
        });
    }
    *scraped.lock().unwrap_or_else(|e| e.into_inner()) = new_items;

    println!("\n🔄 Updating research context with new information...");
    Ok(())
}

fn print_welcome() {
    println!();
    println!("┌─ Deep Research Chat ─────────────────────────────────────────");
    println!("│ {WELCOME_MESSAGE}");
    println!("└──────────────────────────────────────────────────────────────");
    println!();
    println!("Example questions to get started:");
    for (i, q) in STARTER_QUESTIONS.iter().enumerate() {
        println!("  {}. {}", i + 1, q);
    }
    println!();
    println!("{}", "─".repeat(72));
    println!();
}

fn print_search_status(is_new_search: bool, reasoning: &str) {
    println!();
    if is_new_search {
        println!("┌─ Performing new search ─────────────────────────────────────");
    } else {
        println!("┌─ Using existing context ─────────────────────────────────────");
    }
    for line in reasoning.lines() {
        println!("│ {line}");
    }
    println!("└──────────────────────────────────────────────────────────────");
}

fn print_answer(answer: &str, follow_ups: &[String]) {
    println!();
    println!("┌─ Answer ─────────────────────────────────────────────────────");
    for line in answer.lines() {
        println!("│ {line}");
    }
    println!("└──────────────────────────────────────────────────────────────");
    if !follow_ups.is_empty() {
        println!();
        println!("Follow-up questions:");
        for (i, q) in follow_ups.iter().enumerate() {
            println!("  {}. {}", i + 1, q);
        }
    }
}

fn read_user_line() -> io::Result<String> {
    print!("\nYour question: ");
    io::stdout().flush()?;
    let mut buf = String::new();
    io::stdin().read_line(&mut buf)?;
    Ok(buf.trim().to_string())
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
        eprintln!("Set OPENAI_API_KEY or GEMINI_API_KEY for genai.");
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

    let searx_url =
        std::env::var("SEARXNG_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
    let searx = SearxngSearch::new(&searx_url).expect("SearXNG client");
    let scraper = WebpageScraper::new().expect("HTTP client for scraper");

    let scraped_store: Arc<Mutex<Vec<ContentItem>>> = Arc::new(Mutex::new(Vec::new()));

    let mut query_agent: BasicNanoAgent<QueryAgentInput, QueryAgentOutput> =
        BasicNanoAgent::new(query_agent_config(&model));
    query_agent.register_context_provider(CurrentDateContextProvider);
    query_agent.register_context_provider(ScrapedContentContextProvider::new(Arc::clone(
        &scraped_store,
    )));

    let mut choice_agent: BasicNanoAgent<ChoiceAgentInput, ChoiceAgentOutput> =
        BasicNanoAgent::new(choice_agent_config(&model));
    choice_agent.register_context_provider(CurrentDateContextProvider);
    choice_agent.register_context_provider(ScrapedContentContextProvider::new(Arc::clone(
        &scraped_store,
    )));

    let mut qa_history = ChatHistory::new();
    let welcome = QuestionAnsweringOutput {
        answer: WELCOME_MESSAGE.to_string(),
        follow_up_questions: STARTER_QUESTIONS.iter().map(|s| (*s).to_string()).collect(),
    };
    qa_history
        .add_message_schema("assistant", &welcome)
        .expect("seed assistant turn");

    let mut qa_agent: BasicNanoAgent<QuestionAnsweringInput, QuestionAnsweringOutput> =
        BasicNanoAgent::new(qa_agent_config(&model, qa_history));
    qa_agent.register_context_provider(CurrentDateContextProvider);
    qa_agent.register_context_provider(ScrapedContentContextProvider::new(Arc::clone(
        &scraped_store,
    )));

    println!("\n🚀 Initializing Deep Research System...");
    println!("✨ System initialized successfully!");
    print_welcome();

    loop {
        let user_message = match read_user_line() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("stdin: {e}");
                break;
            }
        };

        if user_message.eq_ignore_ascii_case("/exit") || user_message.eq_ignore_ascii_case("/quit")
        {
            println!("\n👋 Goodbye! Thanks for using Deep Research.");
            break;
        }
        if user_message.is_empty() {
            continue;
        }

        println!("\n🤖 Processing your question...");

        let choice_out = match choice_agent
            .run(ChoiceAgentInput {
                user_message: user_message.clone(),
                decision_type: CHOICE_DECISION_PROMPT.to_string(),
            })
            .await
        {
            Ok(o) => o,
            Err(e) => {
                eprintln!("choice agent: {e}");
                continue;
            }
        };

        print_search_status(choice_out.decision, &choice_out.reasoning);

        if choice_out.decision
            && let Err(e) = perform_search_and_update_context(
                &user_message,
                &scraped_store,
                &mut query_agent,
                &searx,
                &scraper,
            )
            .await
        {
            eprintln!("search pipeline: {e}");
        }

        println!("\n🎯 Generating comprehensive answer...");
        let qa_out = match qa_agent
            .run(QuestionAnsweringInput {
                question: user_message,
            })
            .await
        {
            Ok(o) => o,
            Err(e) => {
                eprintln!("QA agent: {e}");
                continue;
            }
        };

        print_answer(&qa_out.answer, &qa_out.follow_up_questions);
    }
}
