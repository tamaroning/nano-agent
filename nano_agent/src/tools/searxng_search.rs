//! SearXNG meta-search client (`format=json`) and I/O schemas for orchestration.

use std::time::Duration;

use genai::chat::Tool as GenaiTool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Tool input; mirrors `SearchToolInput` in `examples/orchestrator`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearxngSearchInput {
    /// Search queries to run
    #[schemars(description = "Search queries to execute")]
    pub queries: Vec<String>,
    /// Category (e.g. `general`); omit to use the instance default.
    #[schemars(
        description = "SearXNG categories parameter (e.g. general); omit for instance default"
    )]
    pub category: Option<String>,
}

/// One search result row.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearxngResultItem {
    /// Result URL
    pub url: String,
    /// Title
    pub title: String,
    /// Snippet, if any
    pub content: Option<String>,
    /// Query that produced this hit
    pub query: String,
}

/// Tool output.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearxngSearchOutput {
    /// Top hits per query, merged in iteration order
    pub results: Vec<SearxngResultItem>,
}

/// Errors from SearXNG requests or parsing.
#[derive(Debug, thiserror::Error)]
pub enum SearxngSearchError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("failed to parse JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid instance URL: {0}")]
    InvalidBaseUrl(String),
    #[error("queries must not be empty")]
    EmptyQueries,
}

/// Async HTTP client for a SearXNG instance.
#[derive(Debug, Clone)]
pub struct SearxngSearch {
    base_url: String,
    client: reqwest::Client,
    max_results_per_query: usize,
}

impl SearxngSearch {
    /// `base_url` like `https://searx.example.org` without a trailing slash.
    pub fn new(base_url: impl Into<String>) -> Result<Self, SearxngSearchError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("nano-agent/searxng_search")
            .build()?;
        Ok(Self {
            base_url: normalize_base_url(base_url.into()),
            client,
            max_results_per_query: 5,
        })
    }

    /// Reads base URL from `SEARXNG_URL`, defaulting to `http://127.0.0.1:8080`.
    pub fn from_env() -> Result<Self, SearxngSearchError> {
        let url =
            std::env::var("SEARXNG_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());
        Self::new(url)
    }

    /// Max number of results to keep per query.
    pub fn with_max_results_per_query(mut self, n: usize) -> Self {
        self.max_results_per_query = n.max(1);
        self
    }

    /// Metadata for LLM function calling (e.g. `genai::ChatRequest::with_tools`).
    pub fn as_genai_tool() -> GenaiTool {
        let schema: Value = serde_json::to_value(schemars::schema_for!(SearxngSearchInput))
            .unwrap_or_else(|_| Value::Object(Default::default()));
        GenaiTool::new("searxng_search")
            .with_description(
                "Run web search via a SearXNG instance. Use for factual questions, current events, or lookups.",
            )
            .with_schema(schema)
    }

    /// Runs the configured queries against SearXNG.
    pub async fn run(
        &self,
        input: SearxngSearchInput,
    ) -> Result<SearxngSearchOutput, SearxngSearchError> {
        if input.queries.is_empty() {
            return Err(SearxngSearchError::EmptyQueries);
        }

        let search_url = build_search_url(&self.base_url)?;
        let mut results = Vec::new();

        for q in &input.queries {
            let q = q.trim();
            if q.is_empty() {
                continue;
            }
            let mut req = self
                .client
                .get(&search_url)
                .query(&[("q", q), ("format", "json")]);
            if let Some(cat) = input
                .category
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                req = req.query(&[("categories", cat)]);
            }

            let resp = req.send().await?.error_for_status()?;
            let body: SearxngJsonResponse = resp.json().await?;

            for r in body.results.into_iter().take(self.max_results_per_query) {
                let Some(url) = r.url.filter(|u| !u.is_empty()) else {
                    continue;
                };
                results.push(SearxngResultItem {
                    url,
                    title: r.title.unwrap_or_default(),
                    content: r.content,
                    query: q.to_string(),
                });
            }
        }

        Ok(SearxngSearchOutput { results })
    }
}

fn normalize_base_url(mut s: String) -> String {
    while s.ends_with('/') {
        s.pop();
    }
    s
}

fn build_search_url(base: &str) -> Result<String, SearxngSearchError> {
    let base = base.trim();
    if base.is_empty() {
        return Err(SearxngSearchError::InvalidBaseUrl(base.to_string()));
    }
    Ok(format!("{}/search", base.trim_end_matches('/')))
}

#[derive(Debug, Deserialize)]
struct SearxngJsonResponse {
    #[serde(default)]
    results: Vec<SearxngJsonResult>,
}

#[derive(Debug, Deserialize)]
struct SearxngJsonResult {
    url: Option<String>,
    title: Option<String>,
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_and_build_url() {
        assert_eq!(
            build_search_url("https://x.org/").unwrap(),
            "https://x.org/search"
        );
    }
}
