//! Fetch a page with [`reqwest`](https://docs.rs/reqwest), extract article HTML with
//! [`readability`](https://docs.rs/readability), convert to Markdown with [`htmd`](https://docs.rs/htmd),
//! and read `<meta>` tags via [`scraper`](https://docs.rs/scraper).

use std::io::Cursor;
use std::time::Duration;

use genai::chat::Tool as GenaiTool;
use htmd::options::{BulletListMarker, HeadingStyle, Options};
use htmd::{Element, HtmlToMarkdown};
use readability::extractor::Product;
use schemars::JsonSchema;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

/// Default browser-like User-Agent (same spirit as the Python tool).
const DEFAULT_USER_AGENT: &str = concat!(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ",
    "AppleWebKit/537.36 (KHTML, like Gecko) ",
    "Chrome/91.0.4472.124 Safari/537.36",
);

/// Tool input.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WebpageScraperInput {
    /// URL of the webpage to scrape.
    #[schemars(description = "URL of the webpage to scrape")]
    pub url: String,
    /// When false, link text is kept but Markdown links are not emitted.
    #[serde(default = "default_include_links")]
    #[schemars(
        description = "If true, preserve hyperlinks in Markdown; if false, keep link text only"
    )]
    pub include_links: bool,
}

fn default_include_links() -> bool {
    true
}

/// Page metadata (Open Graph / meta tags + domain).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WebpageMetadata {
    /// Page title from readability or `<title>`.
    pub title: String,
    /// `meta[name=author]` content, if present.
    pub author: Option<String>,
    /// `meta[name=description]` content, if present.
    pub description: Option<String>,
    /// `meta[property=og:site_name]` content, if present.
    pub site_name: Option<String>,
    /// Host name of the URL.
    pub domain: String,
}

/// Tool output (always returned; failures set `error` and minimal metadata).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WebpageScraperOutput {
    /// Article body as Markdown.
    pub content: String,
    pub metadata: WebpageMetadata,
    /// Set when the scrape or conversion pipeline failed.
    pub error: Option<String>,
}

/// Configuration mirroring the Python `WebpageScraperToolConfig`.
#[derive(Debug, Clone)]
pub struct WebpageScraperConfig {
    pub user_agent: String,
    pub timeout: Duration,
    pub max_content_length: usize,
}

impl Default for WebpageScraperConfig {
    fn default() -> Self {
        Self {
            user_agent: DEFAULT_USER_AGENT.to_string(),
            timeout: Duration::from_secs(30),
            max_content_length: 1_000_000,
        }
    }
}

/// Errors surfaced inside `run_inner` (mapped to `WebpageScraperOutput::error` for callers).
#[derive(Debug, thiserror::Error)]
enum WebpageScraperError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid URL: {0}")]
    Url(#[from] url::ParseError),
    #[error("only http and https URLs are supported")]
    UnsupportedScheme,
    #[error("content length exceeds maximum of {max} bytes")]
    ContentTooLong { max: usize },
    #[error("readability: {0}")]
    Readability(#[from] readability::error::Error),
    #[error("markdown conversion: {0}")]
    Markdown(#[from] std::io::Error),
}

/// Async HTTP client and limits for webpage scraping.
#[derive(Debug, Clone)]
pub struct WebpageScraper {
    client: reqwest::Client,
    user_agent: String,
    max_content_length: usize,
}

impl WebpageScraper {
    /// Builds a client using [`WebpageScraperConfig::default`].
    pub fn new() -> Result<Self, reqwest::Error> {
        Self::with_config(WebpageScraperConfig::default())
    }

    pub fn with_config(config: WebpageScraperConfig) -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder().timeout(config.timeout).build()?;
        Ok(Self {
            client,
            user_agent: config.user_agent,
            max_content_length: config.max_content_length,
        })
    }

    pub fn with_max_content_length(mut self, n: usize) -> Self {
        self.max_content_length = n.max(1);
        self
    }

    /// Metadata for LLM function calling.
    pub fn as_genai_tool() -> GenaiTool {
        let schema: Value = serde_json::to_value(schemars::schema_for!(WebpageScraperInput))
            .unwrap_or_else(|_| Value::Object(Default::default()));
        GenaiTool::new("webpage_scraper")
            .with_description(
                "Fetch a URL, extract the main readable article HTML, and return Markdown plus basic metadata (title, meta tags, domain).",
            )
            .with_schema(schema)
    }

    /// Runs the full pipeline; on failure returns `error` and empty content (same contract as the Python tool).
    pub async fn run(&self, input: WebpageScraperInput) -> WebpageScraperOutput {
        let domain = domain_from_url(&input.url);
        match self.run_inner(input).await {
            Ok(out) => out,
            Err(e) => WebpageScraperOutput {
                content: String::new(),
                metadata: WebpageMetadata {
                    title: "Error retrieving page".into(),
                    author: None,
                    description: None,
                    site_name: None,
                    domain,
                },
                error: Some(e.to_string()),
            },
        }
    }

    async fn run_inner(
        &self,
        input: WebpageScraperInput,
    ) -> Result<WebpageScraperOutput, WebpageScraperError> {
        let url = Url::parse(input.url.trim())?;
        if url.scheme() != "http" && url.scheme() != "https" {
            return Err(WebpageScraperError::UnsupportedScheme);
        }

        let html = self.fetch_html(url.as_str()).await?;
        let doc = Html::parse_document(&html);

        let mut cursor = Cursor::new(html.as_bytes());
        let product = readability::extractor::extract(&mut cursor, &url)?;

        let converter = build_htmd_converter(input.include_links);
        let md_raw = converter.convert(&product.content)?;
        let content = clean_markdown(&md_raw);

        let metadata = build_metadata(&doc, &product, &url);

        Ok(WebpageScraperOutput {
            content,
            metadata,
            error: None,
        })
    }

    async fn fetch_html(&self, url: &str) -> Result<String, WebpageScraperError> {
        let resp = self
            .client
            .get(url)
            .header(
                reqwest::header::ACCEPT,
                "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            )
            .header(reqwest::header::ACCEPT_LANGUAGE, "en-US,en;q=0.5")
            .header(reqwest::header::USER_AGENT, &self.user_agent)
            .header(reqwest::header::CONNECTION, "keep-alive")
            .send()
            .await?
            .error_for_status()?;

        let bytes = resp.bytes().await?;
        if bytes.len() > self.max_content_length {
            return Err(WebpageScraperError::ContentTooLong {
                max: self.max_content_length,
            });
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        Ok(text)
    }
}

fn domain_from_url(url_str: &str) -> String {
    Url::parse(url_str.trim())
        .ok()
        .and_then(|u| u.host_str().map(str::to_string))
        .unwrap_or_default()
}

fn build_htmd_converter(include_links: bool) -> HtmlToMarkdown {
    let opts = Options {
        heading_style: HeadingStyle::Atx,
        bullet_list_marker: BulletListMarker::Dash,
        ..Default::default()
    };

    let mut b = HtmlToMarkdown::builder()
        .options(opts)
        .skip_tags(vec!["script", "style"]);

    if !include_links {
        b = b.add_handler(
            vec!["a"],
            |handlers: &dyn htmd::element_handler::Handlers, element: Element| {
                Some(handlers.walk_children(element.node))
            },
        );
    }
    b.build()
}

fn clean_markdown(md: &str) -> String {
    let trimmed = md.trim();
    if trimmed.is_empty() {
        return "\n".to_string();
    }
    let mut s = trimmed.to_string();
    while s.contains("\n\n\n") {
        s = s.replace("\n\n\n", "\n\n");
    }
    format!("{}\n", s.trim_end())
}

fn build_metadata(doc: &Html, product: &Product, url: &Url) -> WebpageMetadata {
    let domain = url.host_str().unwrap_or("").to_string();
    let title = {
        let t = product.title.trim();
        if !t.is_empty() {
            t.to_string()
        } else {
            title_from_html(doc)
        }
    };
    WebpageMetadata {
        title,
        author: meta_attr(doc, r#"meta[name="author"]"#, "content"),
        description: meta_attr(doc, r#"meta[name="description"]"#, "content"),
        site_name: meta_attr(doc, r#"meta[property="og:site_name"]"#, "content"),
        domain,
    }
}

fn meta_attr(doc: &Html, selector: &str, attr: &str) -> Option<String> {
    let sel = Selector::parse(selector).ok()?;
    doc.select(&sel)
        .next()?
        .value()
        .attr(attr)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn title_from_html(doc: &Html) -> String {
    let Ok(sel) = Selector::parse("title") else {
        return String::new();
    };
    doc.select(&sel)
        .next()
        .map(|e| e.text().collect::<String>())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_markdown_collapses_blank_lines() {
        let raw = "a\n\n\n\nb";
        assert_eq!(clean_markdown(raw), "a\n\nb\n");
    }

    #[test]
    fn clean_markdown_empty_yields_newline() {
        assert_eq!(clean_markdown("   "), "\n");
    }
}
