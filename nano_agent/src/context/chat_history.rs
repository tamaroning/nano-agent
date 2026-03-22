use genai::chat::{ChatMessage, ChatRole, MessageContent};
use serde::Serialize;

/// Error when building chat history from high-level helpers.
#[derive(Debug, thiserror::Error)]
pub enum ChatHistoryError {
    #[error("unknown chat role {0:?} (use \"user\", \"assistant\", or \"system\")")]
    UnknownRole(String),
    #[error("serialization failed: {0}")]
    Serialize(#[from] serde_json::Error),
}

/// Chat history for the agent.
pub struct ChatHistory {
    messages: Vec<ChatMessage>,
}

impl ChatHistory {
    /// Create a new chat history.
    pub fn new() -> Self {
        Self { messages: vec![] }
    }

    /// Add a message to the chat history.
    pub fn add_message(&mut self, role: ChatRole, message: MessageContent) {
        self.messages.push(ChatMessage {
            role,
            content: message,
            options: None,
        });
    }

    /// Serializes `payload` to JSON and stores it as one text turn for `role`.
    pub fn add_message_schema(
        &mut self,
        role: &str,
        payload: &impl Serialize,
    ) -> Result<(), ChatHistoryError> {
        let role = match role.to_ascii_lowercase().as_str() {
            "user" => ChatRole::User,
            "assistant" => ChatRole::Assistant,
            "system" => ChatRole::System,
            other => return Err(ChatHistoryError::UnknownRole(other.to_string())),
        };
        let json = serde_json::to_string(payload)?;
        self.add_message(role, MessageContent::from_text(json));
        Ok(())
    }

    /// Get the chat history as the LLM API request format.
    pub fn get_history(&self) -> &[ChatMessage] {
        &self.messages
    }
}

impl Default for ChatHistory {
    fn default() -> Self {
        Self::new()
    }
}
