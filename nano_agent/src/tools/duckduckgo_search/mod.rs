//! DuckDuckGo text search client and I/O schemas for orchestration.

use std::time::Duration;

use genai::chat::Tool as GenaiTool;
use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Tool input.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DuckDuckGoSearchInput {
    /// Search queries to run.
    #[schemars(description = "Search queries to execute")]
    pub queries: Vec<String>,
}

/// One search result row.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DuckDuckGoResultItem {
    /// Result URL.
    pub url: String,
    /// Title.
    pub title: String,
    /// Snippet, if any.
    pub content: Option<String>,
    /// Query that produced this hit.
    pub query: String,
}

/// Tool output.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DuckDuckGoSearchOutput {
    /// Top hits per query, merged in iteration order.
    pub results: Vec<DuckDuckGoResultItem>,
}

/// Errors from DuckDuckGo requests or parsing.
#[derive(Debug, thiserror::Error)]
pub enum DuckDuckGoSearchError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("failed to parse JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("queries must not be empty")]
    EmptyQueries,
    #[error("failed to extract vqd token")]
    MissingVqd,
}

/// Async HTTP client for DuckDuckGo text search.
#[derive(Debug, Clone)]
pub struct DuckDuckGoSearch {
    client: reqwest::Client,
    max_results_per_query: usize,
}

impl DuckDuckGoSearch {
    pub fn new() -> Result<Self, DuckDuckGoSearchError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("nano-agent/duckduckgo_search")
            .build()?;
        Ok(Self {
            client,
            max_results_per_query: 5,
        })
    }

    /// Max number of results to keep per query.
    pub fn with_max_results_per_query(mut self, n: usize) -> Self {
        self.max_results_per_query = n.max(1);
        self
    }

    /// Metadata for LLM function calling.
    pub fn as_genai_tool() -> GenaiTool {
        let schema: Value = serde_json::to_value(schemars::schema_for!(DuckDuckGoSearchInput))
            .unwrap_or_else(|_| Value::Object(Default::default()));
        GenaiTool::new("duckduckgo_search")
            .with_description(
                "Run web search via DuckDuckGo. Use for factual questions, current events, or lookups.",
            )
            .with_schema(schema)
    }

    /// Runs the configured queries against DuckDuckGo.
    pub async fn run(
        &self,
        input: DuckDuckGoSearchInput,
    ) -> Result<DuckDuckGoSearchOutput, DuckDuckGoSearchError> {
        if input.queries.is_empty() {
            return Err(DuckDuckGoSearchError::EmptyQueries);
        }

        let mut results = Vec::new();
        for q in &input.queries {
            let q = q.trim();
            if q.is_empty() {
                continue;
            }

            let vqd = self.fetch_vqd(q).await?;
            let body = self
                .client
                .get("https://links.duckduckgo.com/d.js")
                .query(&[
                    ("q", q),
                    ("l", "wt-wt"),
                    ("s", "0"),
                    ("dl", "en"),
                    ("vqd", &vqd),
                    ("bing_market", "en-US"),
                    ("ex", "-1"),
                ])
                .header(reqwest::header::REFERER, "https://duckduckgo.com/")
                .send()
                .await?
                .error_for_status()?
                .text()
                .await?;

            for item in parse_results(&body, q)?
                .into_iter()
                .take(self.max_results_per_query)
            {
                results.push(item);
            }
        }

        Ok(DuckDuckGoSearchOutput { results })
    }

    async fn fetch_vqd(&self, query: &str) -> Result<String, DuckDuckGoSearchError> {
        let html = self
            .client
            .get("https://duckduckgo.com/")
            .query(&[("q", query)])
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        extract_vqd(&html).ok_or(DuckDuckGoSearchError::MissingVqd)
    }
}

fn extract_vqd(html: &str) -> Option<String> {
    for pattern in [r#"vqd="([^"]+)""#, r#"vqd=([0-9-]+)\&"#, r#"vqd='([^']+)'"#] {
        let re = Regex::new(pattern).ok()?;
        if let Some(c) = re.captures(html) {
            if let Some(m) = c.get(1) {
                return Some(m.as_str().to_string());
            }
        }
    }
    None
}

fn parse_results(
    body: &str,
    query: &str,
) -> Result<Vec<DuckDuckGoResultItem>, DuckDuckGoSearchError> {
    if let Ok(rows) = parse_results_json_like(body)
        && !rows.is_empty()
    {
        return Ok(rows_to_items(rows, query));
    }
    Ok(parse_results_from_html(body, query))
}

fn extract_embedded_json(body: &str) -> Option<&str> {
    let start = body.find('{')?;
    let end = body.rfind('}')?;
    if start >= end {
        return None;
    }
    Some(&body[start..=end])
}

fn parse_results_json_like(body: &str) -> Result<Vec<DuckDuckGoJsonResult>, DuckDuckGoSearchError> {
    if let Ok(data) = serde_json::from_str::<DuckDuckGoResponse>(body)
        && !data.results.is_empty()
    {
        return Ok(data.results);
    }

    if let Some(obj) = extract_embedded_json(body)
        && let Ok(data) = serde_json::from_str::<DuckDuckGoResponse>(obj)
        && !data.results.is_empty()
    {
        return Ok(data.results);
    }

    if let Some(arr) = extract_js_results_array(body) {
        if let Ok(rows) = serde_json::from_str::<Vec<DuckDuckGoJsonResult>>(arr) {
            return Ok(rows);
        }

        let wrapped = format!(r#"{{"results":{arr}}}"#);
        if let Ok(data) = serde_json::from_str::<DuckDuckGoResponse>(&wrapped)
            && !data.results.is_empty()
        {
            return Ok(data.results);
        }
    }

    if let Some(obj_arr) = extract_first_json_array_of_objects(body) {
        if let Ok(rows) = serde_json::from_str::<Vec<DuckDuckGoJsonResult>>(obj_arr) {
            return Ok(rows);
        }
    }

    let json_text = extract_embedded_json(body).unwrap_or(body);
    let data: DuckDuckGoResponse = serde_json::from_str(json_text)?;
    Ok(data.results)
}

fn extract_js_results_array(body: &str) -> Option<&str> {
    let re = Regex::new(r#"(?s)DDG\.pageLayout\.load\([^,]+,\s*(\[[\s\S]*?\])\s*\)"#).ok()?;
    let cap = re.captures(body)?;
    cap.get(1).map(|m| m.as_str())
}

fn extract_first_json_array_of_objects(body: &str) -> Option<&str> {
    let bytes = body.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            let segment = &body[i..];
            if let Some(end) = find_matching_bracket(segment) {
                let arr = &segment[..=end];
                let trimmed = arr.trim();
                if trimmed.starts_with("[{") && trimmed.ends_with(']') {
                    return Some(arr);
                }
                i += end + 1;
                continue;
            }
        }
        i += 1;
    }
    None
}

fn find_matching_bracket(s: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, ch) in s.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '[' => depth += 1,
            ']' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
                if depth == 0 {
                    return Some(idx);
                }
            }
            _ => {}
        }
    }
    None
}

fn rows_to_items(rows: Vec<DuckDuckGoJsonResult>, query: &str) -> Vec<DuckDuckGoResultItem> {
    let mut out = Vec::new();
    for row in rows {
        let Some(url) = row.url.filter(|u| !u.trim().is_empty()) else {
            continue;
        };
        out.push(DuckDuckGoResultItem {
            url,
            title: row.title.unwrap_or_default(),
            content: row.snippet,
            query: query.to_string(),
        });
    }
    out
}

fn parse_results_from_html(body: &str, query: &str) -> Vec<DuckDuckGoResultItem> {
    let row_re = Regex::new(r#"<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#).ok();
    let tag_re = Regex::new(r"<[^>]+>").ok();
    let ws_re = Regex::new(r"\s+").ok();
    let (Some(row_re), Some(tag_re), Some(ws_re)) = (row_re, tag_re, ws_re) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for cap in row_re.captures_iter(body) {
        let Some(url) = cap.get(1).map(|m| m.as_str().trim().to_string()) else {
            continue;
        };
        if url.is_empty() || url.starts_with('#') || url.starts_with("javascript:") {
            continue;
        }

        let raw_title = cap.get(2).map(|m| m.as_str()).unwrap_or_default();
        let title_no_tags = tag_re.replace_all(raw_title, "");
        let title = ws_re.replace_all(title_no_tags.trim(), " ").to_string();
        if title.is_empty() {
            continue;
        }

        out.push(DuckDuckGoResultItem {
            url,
            title,
            content: None,
            query: query.to_string(),
        });
    }
    out
}

#[derive(Debug, Deserialize)]
struct DuckDuckGoResponse {
    #[serde(default, rename = "results")]
    results: Vec<DuckDuckGoJsonResult>,
}

#[derive(Debug, Deserialize)]
struct DuckDuckGoJsonResult {
    #[serde(alias = "u", alias = "url")]
    url: Option<String>,
    #[serde(alias = "t", alias = "title")]
    title: Option<String>,
    #[serde(alias = "a", alias = "snippet", alias = "body")]
    snippet: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_vqd_from_html() {
        let html = "foo vqd='4-12345678901234567890123456789012' bar";
        let vqd = extract_vqd(html).unwrap();
        assert_eq!(vqd, "4-12345678901234567890123456789012");
    }

    #[test]
    fn parse_results_from_json() {
        let body = r#"{"results":[{"u":"https://example.com","t":"Title","a":"Snippet"}]}"#;
        let out = parse_results(body, "test").unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].url, "https://example.com");
        assert_eq!(out[0].title, "Title");
    }

    #[test]
    fn parse_results_from_js_wrapped_array() {
        let body =
            r#"DDG.pageLayout.load('d',[{"u":"https://example.com","t":"Title","a":"Snippet"}]);"#;
        let out = parse_results(body, "test").unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].url, "https://example.com");
        assert_eq!(out[0].title, "Title");
    }

    #[test]
    fn parse_results_from_html_fallback() {
        let body =
            r#"<div class="body"><h2><a href="https://example.com">Example Title</a></h2></div>"#;
        let out = parse_results(body, "test").unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].url, "https://example.com");
        assert_eq!(out[0].title, "Example Title");
    }
}
