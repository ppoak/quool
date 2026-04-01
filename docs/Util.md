# Util Module

Utility functions for logging, HTTP requests, email notifications, documentation generation, and web search.

## Functions

### setup_logger()

Create and configure a logger with optional stream/file handlers and rotation.

```python
from quool import setup_logger
```

```python
def setup_logger(
    name: str,
    level: int = logging.INFO,
    replace: bool = False,
    stream: bool = True,
    file: Union[Path, str] = None,
    clear: bool = False,
    style: Union[int, str] = 1,
    rotation: str = None,
    max_bytes: int = None,
    backup_count: int = None,
    when: str = None,
    interval: int = None,
) -> logging.Logger
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Logger name |
| `level` | `int` | `logging.INFO` | Logging level |
| `replace` | `bool` | `False` | Create new logger even if exists |
| `stream` | `bool` | `True` | Add StreamHandler for console output |
| `file` | `Union[Path, str]` | `None` | File path for file handler |
| `clear` | `bool` | `False` | Truncate file before use |
| `style` | `Union[int, str]` | `1` | Formatter style (1-4) or custom formatter |
| `rotation` | `str` | `None` | Rotation mode ("size" or "time") |
| `max_bytes` | `int` | `10*1024*1024` | Max bytes for size rotation |
| `backup_count` | `int` | `5` / `7` | Number of backup files |
| `when` | `str` | `"midnight"` | Time rotation trigger |
| `interval` | `int` | `1` | Time rotation interval |

**Returns:** `logging.Logger` - The configured logger instance

#### Formatter Styles

| Style | Format |
|-------|--------|
| 1 | `[LEVEL]@[TIME]-[NAME]: MESSAGE` |
| 2 | `[LEVEL]@[TIME]-[NAME]@[MODULE:LINENO]: MESSAGE` |
| 3 | `[LEVEL]@[TIME]-[NAME]@[MODULE:LINENO#FUNC]: MESSAGE` |
| 4 | `[LEVEL]@[TIME]-[NAME]@[MODULE:LINENO#FUNC~PID:THREAD]: MESSAGE` |

**Example:**
```python
from quool import setup_logger

logger = setup_logger(
    "my_strategy",
    level="INFO",
    file="/var/log/strategy.log",
    rotation="size",
    max_bytes=10*1024*1024,
    backup_count=5
)
logger.info("Strategy initialized")
```

---

### notify_task()

Decorator that runs a task and sends an email notification with its result.

```python
from quool import notify_task
```

```python
def notify_task(
    sender: str = None,
    password: str = None,
    receiver: str = None,
    smtp_server: str = None,
    smtp_port: int = None,
    cc: str = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `sender` | `str` | Sender email address |
| `password` | `str` | Sender email password or app password |
| `receiver` | `str` | Comma-separated recipient addresses |
| `smtp_server` | `str` | SMTP server host |
| `smtp_port` | `int` | SMTP server port |
| `cc` | `str` | Comma-separated CC addresses |

**Environment Variables:**
- `NOTIFY_TASK_SENDER`
- `NOTIFY_TASK_PASSWORD`
- `NOTIFY_TASK_RECEIVER`
- `NOTIFY_TASK_SMTP_SERVER`
- `NOTIFY_TASK_SMTP_PORT`
- `NOTIFY_TASK_CC`

#### Features

- Sends email on task success or failure
- Converts pandas DataFrame/Series to markdown (head/tail if large)
- Embeds local images (.png, .jpg, .gif) via Content-ID (CID)
- Attaches files as email attachments
- Includes execution time and parameters in email

**Example:**
```python
from quool import notify_task

@notify_task(
    sender="bot@example.com",
    password="app_password",
    receiver="admin@example.com",
    smtp_server="smtp.example.com",
    smtp_port=587
)
def daily_etl():
    # ETL processing...
    return {"status": "success", "records": 1000}
```

---

### proxy_request()

HTTP request with proxy failover and retry logic.

```python
from quool import proxy_request
```

```python
@retry(exceptions=(requests.exceptions.RequestException,), tries=5, delay=1, backoff=2)
def proxy_request(
    url: str,
    method: str = "GET",
    proxies: Union[dict, list] = None,
    delay: float = 1,
    **kwargs,
) -> requests.Response
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | Required | Target URL |
| `method` | `str` | `"GET"` | HTTP method |
| `proxies` | `Union[dict, list]` | `None` | Single proxy dict or list of proxies |
| `delay` | `float` | `1` | Seconds between proxy attempts |
| `**kwargs` | | | Additional arguments for `requests.request` |

**Returns:** `requests.Response` - The successful response object

**Behavior:**
- Tries each proxy in order
- Falls back to direct connection if all proxies fail
- Retry decorator will re-invoke on failure

**Example:**
```python
from quool import proxy_request

response = proxy_request(
    "https://api.example.com/data",
    proxies=[
        {"http": "http://proxy1:8080", "https": "http://proxy1:8080"},
        {"http": "http://proxy2:8080", "https": "http://proxy2:8080"},
    ]
)
```

---

### google_search()

Google search via SerpAPI.

```python
from quool import google_search
```

```python
def google_search(
    query: str,
    location: Literal["China", "United States", "Germany", "France"] = "China",
    country: str = "cn",
    language: str = "zh-cn",
    to_be_searched: Optional[str] = None,
    start: str = "1",
    num: str = "10",
) -> str
```

**Requires:** `SERPAPI_KEY` environment variable (comma-separated for multiple keys)

**Returns:** `str` - Markdown-formatted search report

**Example:**
```python
from quool import google_search

results = google_search(
    "DuckDB Parquet upsert",
    location="United States",
    num="5"
)
```

---

### read_url()

Fetch and summarize web page content via Jina reader proxy.

```python
from quool import read_url
```

```python
def read_url(
    url_or_urls: Union[str, List],
    engine: Literal["direct", "browser"] = "browser",
    return_format: Literal["markdown", "html", "text", "screeshot"] = "markdown",
    with_links_summary: Literal["all", "true"] = "true",
    with_image_summary: Literal["all", "true"] = "true",
    retain_image: bool = False,
    do_not_track: bool = True,
    set_cookie: Optional[str] = None,
    max_length_each: int = 100000,
) -> str
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url_or_urls` | `Union[str, List]` | Required | Single URL or list of URLs |
| `engine` | `Literal` | `"browser"` | Fetching engine ("direct" or "browser") |
| `return_format` | `Literal` | `"markdown"` | Return format |
| `with_links_summary` | `Literal` | `"true"` | Include links summary |
| `with_image_summary` | `Literal` | `"true"` | Include image summary |
| `retain_image` | `bool` | `False` | Retain image references |
| `do_not_track` | `bool` | `True` | Send DNT header |
| `set_cookie` | `Optional[str]` | `None` | Cookie header value |
| `max_length_each` | `int` | `100000` | Max characters per page |

**Returns:** `str` - Markdown-formatted report with results

**Example:**
```python
from quool import read_url

content = read_url("https://example.com/article")

results = read_url([
    "https://example.com/page1",
    "https://example.com/page2"
])
```

---

## Usage Examples

### Logging Setup

```python
from quool import setup_logger
import logging

logger = setup_logger(
    "backtest",
    level=logging.INFO,
    file="/var/log/backtest.log",
    rotation="time",
    when="midnight",
    backup_count=7,
    style=2
)
logger.info("Backtest started")
```

### Task Notification

```python
from quool import notify_task
import pandas as pd

@notify_task(
    sender="etl@example.com",
    password="app_password",
    receiver="team@example.com",
    smtp_server="smtp.example.com",
    smtp_port=587
)
def process_sales_data(date: str) -> pd.DataFrame:
    df = pd.DataFrame({"region": ["US", "EU"], "sales": [1000, 800]})
    return df
```

### Web Search and Reading

```python
from quool import google_search, read_url

# Search for information
search_results = google_search("quool backtesting framework")

# Read specific pages
pages = read_url([
    "https://github.com/ppoak/quool",
    "https://github.com/ppoak/quool/docs"
])
```
