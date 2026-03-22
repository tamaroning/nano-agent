//! Structured LLM agents in Rust: JSON-schema input/output, system prompt building, optional chat
//! history, context providers, and optional tools. Backed by [`genai`](https://docs.rs/genai) (e.g.
//! Gemini with JSON response format).
//!
//! ## Features
//!
//! | Feature | Purpose |
//! |---------|---------|
//! | `searxng` | SearXNG search tool |
//! | `calculator` | Expression evaluation tool |
//! | `webpage_scraper` | Fetch and simplify web pages |
//!
//! ## Example
//!
//! ```no_run
//! use nano_agent::{
//!     AgentConfig, BasicChatInputSchema, BasicNanoAgent, NanoAgent,
//!     context::{ChatHistory, SystemPromptGenerator},
//! };
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, JsonSchema)]
//! struct Out {
//!     chat_message: String,
//! }
//!
//! # async fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let mut agent = BasicNanoAgent::<BasicChatInputSchema, Out>::new(
//!     AgentConfig::new("gemini-2.5-flash-lite")
//!         .with_system_prompt_generator(SystemPromptGenerator::new())
//!         .with_chat_history(ChatHistory::new()),
//! );
//! let _reply: Out = agent
//!     .run(BasicChatInputSchema {
//!         chat_message: "Hello".into(),
//!     })
//!     .await?;
//! # Ok(())
//! # }
//! ```

pub mod context;
pub mod tools;

// Re-exports for convenience.
pub use async_trait::async_trait;
pub use genai;

use std::marker::PhantomData;

use crate::context::{ChatHistory, ContextProvider, SystemPromptGenerator};
use genai::chat::{ChatMessage, ChatRequest, ChatResponseFormat, ChatRole, JsonSpec};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Basic chat input schema.
#[derive(Serialize, Deserialize, JsonSchema)]
pub struct BasicChatInputSchema {
    pub chat_message: String,
}

/// Basic chat output schema.
#[derive(Serialize, Deserialize, JsonSchema)]
pub struct BasicChatOutputSchema {
    pub chat_message: String,
}

/// Agent error.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    #[error("Request failed: {0}")]
    RequestFailed(String),
}

/// Inner agent state.
pub struct AgentInner {
    config: AgentConfig,
}

impl AgentInner {
    fn new(config: AgentConfig) -> Self {
        Self { config }
    }

    async fn run_turn<I, O>(&mut self, input: I) -> Result<O, AgentError>
    where
        I: Serialize + JsonSchema + Send,
        O: Serialize + for<'de> Deserialize<'de> + JsonSchema + Send,
    {
        self.config.run_turn(input).await
    }
}

/// Agent configuration.
pub struct AgentConfig {
    model: String,
    system_prompt_generator: SystemPromptGenerator,
    chat_history: Option<ChatHistory>,
    chat_options: genai::chat::ChatOptions,
    genai_client: genai::Client,
}

impl AgentConfig {
    /// Create a new Agent configuration.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system_prompt_generator: SystemPromptGenerator::new(),
            chat_history: None,
            chat_options: genai::chat::ChatOptions::default(),
            genai_client: genai::Client::builder().build(),
        }
    }

    /// Set the system prompt generator.
    pub fn with_system_prompt_generator(
        mut self,
        system_prompt_generator: SystemPromptGenerator,
    ) -> Self {
        self.system_prompt_generator = system_prompt_generator;
        self
    }

    /// Set the chat history.
    pub fn with_chat_history(mut self, chat_history: ChatHistory) -> Self {
        self.chat_history = Some(chat_history);
        self
    }

    /// Set default [`genai::chat::ChatOptions`] merged into each `run` (e.g. reasoning effort).
    pub fn with_chat_options(mut self, chat_options: genai::chat::ChatOptions) -> Self {
        self.chat_options = chat_options;
        self
    }

    /// Generate the system prompt with the output schema.
    fn system_prompt_with_output_schema<O: JsonSchema>(&self) -> String {
        let system_prompt = self.system_prompt_generator.generate();
        let schema_context = output_schema_instructions::<O>();
        format!("{}\n\n{}", system_prompt, schema_context)
    }

    /// Prepare the messages for the LLM API request.
    pub(crate) fn prepare_messages<I: Serialize + JsonSchema, O: JsonSchema>(
        &self,
        input: &I,
    ) -> Vec<ChatMessage> {
        let system_message = self.system_prompt_with_output_schema::<O>();
        let mut messages = vec![ChatMessage {
            role: ChatRole::System,
            content: system_message.into(),
            options: None,
        }];
        if let Some(chat_history) = &self.chat_history {
            messages.extend(chat_history.get_history().to_vec());
        }
        messages.push(ChatMessage {
            role: ChatRole::User,
            content: serde_json::to_string_pretty(input).unwrap().into(),
            options: None,
        });
        messages
    }

    /// One chat turn via `genai`: serialize `I` as the user message, parse the reply as `O`.
    async fn run_turn<I, O>(&mut self, input: I) -> Result<O, AgentError>
    where
        I: Serialize + JsonSchema + Send,
        O: Serialize + for<'de> Deserialize<'de> + JsonSchema + Send,
    {
        let messages = self.prepare_messages::<I, O>(&input);

        let options = self
            .chat_options
            .clone()
            .with_response_format(ChatResponseFormat::JsonSpec(JsonSpec {
                name: "Output schema".to_string(),
                description: None,
                schema: schemars::schema_for!(O).into(),
            }));

        tracing::debug!(
            "Sending messages to the model: {}",
            serde_json::to_string_pretty(&messages).unwrap()
        );

        let response = self
            .genai_client
            .exec_chat(&self.model, ChatRequest::new(messages), Some(&options))
            .await
            .map_err(|e| AgentError::RequestFailed(e.to_string()))?;

        let response_text = response.first_text().ok_or_else(|| {
            AgentError::InvalidResponse("LLM response did not contain text content".to_string())
        })?;
        let parsed: O = serde_json::from_str(response_text)
            .map_err(|e| AgentError::InvalidResponse(e.to_string()))?;

        tracing::debug!(
            "Received response: {}",
            serde_json::to_string_pretty(&parsed).unwrap()
        );

        if let Some(chat_history) = &mut self.chat_history {
            chat_history.add_message(
                ChatRole::User,
                serde_json::to_string_pretty(&input).unwrap().into(),
            );
            chat_history.add_message(ChatRole::Assistant, response_text.into());
        }

        Ok(parsed)
    }
}

/// Wraps [`AgentInner`] and implements [`NanoAgent`] with no extra state or hooks.
///
/// Type parameters default to [`BasicChatInputSchema`] / [`BasicChatOutputSchema`]. For custom
/// structured I/O, set `I` and `O` explicitly, e.g.
/// `BasicNanoAgent<BasicChatInputSchema, MyOutput>::new(config)`.
pub struct BasicNanoAgent<
    I: Serialize + JsonSchema + Send + 'static = BasicChatInputSchema,
    O: for<'de> Deserialize<'de> + JsonSchema + Send + 'static = BasicChatOutputSchema,
> {
    inner: AgentInner,
    _io: PhantomData<fn() -> (I, O)>,
}

impl<I, O> BasicNanoAgent<I, O>
where
    I: Serialize + JsonSchema + Send + 'static,
    O: for<'de> Deserialize<'de> + JsonSchema + Send + 'static,
{
    pub fn new(config: AgentConfig) -> Self {
        Self {
            inner: AgentInner::new(config),
            _io: PhantomData,
        }
    }
}

/// Schema-driven chat agent: system prompt, optional history, structured I/O.
///
/// Type parameters default to [`BasicChatInputSchema`] / [`BasicChatOutputSchema`]; override
/// them when you need custom structured I/O. Prefer [`BasicNanoAgent`] when you only hold an
/// [`AgentInner`]; implement this trait when you need extra state or to override
/// [`Self::run`](NanoAgent::run).
#[async_trait]
pub trait NanoAgent<
    I: Serialize + JsonSchema + Send + 'static = BasicChatInputSchema,
    O: Serialize + for<'de> Deserialize<'de> + JsonSchema + Send + 'static = BasicChatOutputSchema,
>: Send
{
    fn get_inner(&self) -> &AgentInner;
    fn get_inner_mut(&mut self) -> &mut AgentInner;

    fn register_context_provider(&mut self, provider: impl ContextProvider + Send + 'static) {
        self.get_inner_mut()
            .config
            .system_prompt_generator
            .get_context_providers_mut()
            .push(Box::new(provider));
    }

    /// Run the agent.
    async fn run(&mut self, input: I) -> Result<O, AgentError> {
        self.get_inner_mut().run_turn(input).await
    }
}

impl<I, O> NanoAgent<I, O> for BasicNanoAgent<I, O>
where
    I: Serialize + JsonSchema + Send + 'static,
    O: Serialize + for<'de> Deserialize<'de> + JsonSchema + Send + 'static,
{
    fn get_inner(&self) -> &AgentInner {
        &self.inner
    }

    fn get_inner_mut(&mut self) -> &mut AgentInner {
        &mut self.inner
    }
}

fn output_schema_instructions<T: JsonSchema>() -> String {
    let output_schema = schemars::schema_for!(T);
    format!(
        "Understand the request and respond with a single object that matches the following schema.
        Fill string fields with **new assistant-authored** text as appropriate; do not copy the user's wording into those fields unless the task explicitly requires repetition.

        {}

        Return only a response that validates against this schema, not the schema itself.",
        serde_json::to_string_pretty(&output_schema).unwrap()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ChatHistory;
    use genai::chat::{ChatRole, MessageContent};

    fn config_without_history() -> AgentConfig {
        AgentConfig::new("test-model")
    }

    #[test]
    fn prepare_messages_order_without_history_is_system_then_user() {
        let cfg = config_without_history();
        let input = BasicChatInputSchema {
            chat_message: "hello".into(),
        };
        let msgs = cfg.prepare_messages::<BasicChatInputSchema, BasicChatOutputSchema>(&input);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, ChatRole::System);
        assert_eq!(msgs[1].role, ChatRole::User);
        assert!(msgs[0].options.is_none());
        assert!(msgs[1].options.is_none());
    }

    #[test]
    fn prepare_messages_user_content_is_pretty_json_of_input() {
        let cfg = config_without_history();
        let input = BasicChatInputSchema {
            chat_message: "ping".into(),
        };
        let expected = serde_json::to_string_pretty(&input).unwrap();
        let msgs = cfg.prepare_messages::<BasicChatInputSchema, BasicChatOutputSchema>(&input);
        let user_text = msgs[1].content.first_text().expect("user message text");
        assert_eq!(user_text, expected.as_str());
        assert!(user_text.contains("ping"));
    }

    #[test]
    fn prepare_messages_system_includes_output_schema_instructions() {
        let cfg = config_without_history();
        let input = BasicChatInputSchema {
            chat_message: "x".into(),
        };
        let msgs = cfg.prepare_messages::<BasicChatInputSchema, BasicChatOutputSchema>(&input);
        let system = msgs[0].content.first_text().expect("system message text");
        assert!(
            system.contains("matches the following schema"),
            "system prompt should embed schema instructions"
        );
        assert!(
            system.contains("chat_message"),
            "system prompt should include output JSON schema field names"
        );
    }

    #[test]
    fn prepare_messages_inserts_chat_history_between_system_and_user() {
        let mut history = ChatHistory::new();
        history.add_message(ChatRole::User, MessageContent::from_text("prior user"));
        history.add_message(
            ChatRole::Assistant,
            MessageContent::from_text("prior assistant"),
        );
        let cfg = AgentConfig::new("test-model").with_chat_history(history);
        let input = BasicChatInputSchema {
            chat_message: "latest".into(),
        };
        let msgs = cfg.prepare_messages::<BasicChatInputSchema, BasicChatOutputSchema>(&input);
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, ChatRole::System);
        assert_eq!(
            msgs[1].content.first_text(),
            Some("prior user"),
            "first history turn"
        );
        assert_eq!(
            msgs[2].content.first_text(),
            Some("prior assistant"),
            "second history turn"
        );
        assert_eq!(msgs[3].role, ChatRole::User);
        let latest = msgs[3].content.first_text().expect("current user turn");
        assert!(
            latest.contains("latest"),
            "final user message should be serialized input"
        );
    }
}
