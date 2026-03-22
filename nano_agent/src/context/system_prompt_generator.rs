use crate::context::ContextProvider;

#[derive(Default)]
pub struct SystemPromptGenerator {
    background: Option<Vec<String>>,
    steps: Option<Vec<String>>,
    output_instructions: Option<Vec<String>>,
    context_provider: Vec<Box<dyn ContextProvider + Send>>,
}

impl SystemPromptGenerator {
    /// Create a new system prompt generator.
    pub fn new() -> Self {
        Self {
            background: None,
            steps: None,
            output_instructions: None,
            context_provider: vec![],
        }
    }

    /// Set the background for the system prompt.
    pub fn with_background(mut self, background: Vec<String>) -> Self {
        self.background = Some(background);
        self
    }

    /// Set the steps for the system prompt.
    pub fn with_steps(mut self, steps: Vec<String>) -> Self {
        self.steps = Some(steps);
        self
    }

    /// Set the output instructions for the system prompt.
    pub fn with_output_instructions(mut self, output_instructions: Vec<String>) -> Self {
        self.output_instructions = Some(output_instructions);
        self
    }

    pub(crate) fn get_context_providers_mut(
        &mut self,
    ) -> &mut Vec<Box<dyn ContextProvider + Send>> {
        &mut self.context_provider
    }

    /// Generate the system prompt (same structure as Python `generate_prompt`).
    pub fn generate(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        let sections = [
            ("IDENTITY and PURPOSE", self.background.as_ref()),
            ("INTERNAL ASSISTANT STEPS", self.steps.as_ref()),
            ("OUTPUT INSTRUCTIONS", self.output_instructions.as_ref()),
        ];

        for (title, content) in sections {
            if let Some(items) = content {
                if !items.is_empty() {
                    parts.push(format!("# {title}"));
                    for item in items {
                        parts.push(format!("- {item}"));
                    }
                    parts.push(String::new());
                }
            }
        }

        if !self.context_provider.is_empty() {
            parts.push("# EXTRA INFORMATION AND CONTEXT".to_string());
            for provider in &self.context_provider {
                let info = provider.get_info();
                if !info.is_empty() {
                    parts.push(format!("## {}", provider.title()));
                    parts.push(info);
                    parts.push(String::new());
                }
            }
        }

        parts.join("\n").trim().to_string()
    }
}
