#[cfg(feature = "calculator")]
mod calculator;

#[cfg(feature = "calculator")]
pub use calculator::{Calculator, CalculatorError, CalculatorInput, CalculatorOutput};

#[cfg(feature = "searxng")]
mod searxng_search;

#[cfg(feature = "searxng")]
pub use searxng_search::{
    SearxngResultItem, SearxngSearch, SearxngSearchError, SearxngSearchInput, SearxngSearchOutput,
};

#[cfg(feature = "duckduckgo_search")]
mod duckduckgo_search;

#[cfg(feature = "duckduckgo_search")]
pub use duckduckgo_search::{
    DuckDuckGoResultItem, DuckDuckGoSearch, DuckDuckGoSearchError, DuckDuckGoSearchInput,
    DuckDuckGoSearchOutput,
};

#[cfg(feature = "webpage_scraper")]
mod webpage_scraper;

#[cfg(feature = "webpage_scraper")]
pub use webpage_scraper::{
    WebpageMetadata, WebpageScraper, WebpageScraperConfig, WebpageScraperInput,
    WebpageScraperOutput,
};
