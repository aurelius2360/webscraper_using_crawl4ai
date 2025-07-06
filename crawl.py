import asyncio
import os
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from aiofiles import open as aio_open
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
START_URL = "https://docs.crawl4ai.com/"  # Replace with your target website
OUTPUT_DIR = "crawled_markdown"  # Directory to store Markdown files
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summaries.md")  # File for all summaries
MAX_DEPTH = 3  # Maximum depth for crawling subpages
MAX_PAGES = 50  # Maximum number of pages to crawl
REQUEST_DELAY = 1  # Delay between requests in seconds
BATCH_SIZE = 5  # Number of concurrent crawls
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Load API key from .env
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # Default model

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def save_markdown(url, markdown_content, page_number):
    """Save Markdown content to a file with a unique name."""
    parsed_url = urlparse(url)
    safe_filename = f"{parsed_url.netloc}_{page_number}.md".replace("/", "_").replace(":", "_")
    file_path = os.path.join(OUTPUT_DIR, safe_filename)
    
    async with aio_open(file_path, "w", encoding="utf-8") as f:
        await f.write(markdown_content)
    print(f"Saved Markdown for {url} to {file_path}")
    return file_path

async def summarize_content(content, url):
    """Summarize the Markdown content using GroqCloud API."""
    try:
        client = AsyncGroq(api_key=GROQ_API_KEY)
        prompt = (
            f"Summarize the following content in 2-3 sentences, capturing the main points:\n\n{content}\n\n"
            "Provide a concise summary suitable for a report."
        )
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
        return f"### Summary for {url}\n{summary}\n\n"
    except Exception as e:
        return f"### Summary for {url}\nFailed to generate summary: {str(e)}\n\n"

async def append_summary(summary):
    """Append a summary to the summaries Markdown file."""
    async with aio_open(SUMMARY_FILE, "a", encoding="utf-8") as f:
        await f.write(summary)

async def log_message(message):
    """Log messages to console."""
    print(message)

async def main():
    # Initialize summaries file
    async with aio_open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        await f.write("# Crawl Summaries\n\n")

    # Configure browser settings for handling JavaScript
    browser_config = BrowserConfig(
        headless=True,  # Run in headless mode
        java_script_enabled=True,  # Enable JavaScript execution
        verbose=True  # Enable logging for debugging
    )

    # Configure crawler settings
    crawler_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,  # Bypass cache for fresh data
        word_count_threshold=10,  # Minimum words per content block
        exclude_external_links=True,  # Exclude external links
        remove_overlay_elements=True,  # Remove popups/modals
        process_iframes=True,  # Process iframe content
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=0.5,  # Filter out low-value content
                threshold_type="fixed",
                min_word_threshold=10
            )
        )
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Perform deep crawl with breadth-first strategy
        crawl_generator = await crawler.adeep_crawl(
            start_url=START_URL,
            strategy="bfs",  # Breadth-first search
            max_depth=MAX_DEPTH,
            max_pages=MAX_PAGES,
            config=crawler_config
        )

        # Process results as they stream in
        pages_crawled = 0
        async for result in crawl_generator:
            if result.success:
                pages_crawled += 1
                markdown_content = result.markdown.fit_markdown if result.markdown.fit_markdown else result.markdown.raw_markdown
                file_path = await save_markdown(result.url, markdown_content, pages_crawled)
                
                # Summarize the content using Groq
                async with aio_open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                summary = await summarize_content(content, result.url)
                await append_summary(summary)
                
                await log_message(
                    f"[OK] Crawled: {result.url} (Depth: {result.depth}, Markdown Length: {len(markdown_content)})"
                )
            else:
                await log_message(f"[FAIL] URL: {result.url}, Error: {result.error_message}")

        await log_message(f"\nDeep crawl finished. Total pages successfully crawled: {pages_crawled}")
        await log_message(f"Summaries saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    asyncio.run(main())