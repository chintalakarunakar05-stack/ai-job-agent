"""Data normalization for job postings.

Standardizes raw attributes collected across different sources (Greenhouse, Lever, etc.)
into a unified, cleanly formatted JobCreate model.
"""

import html
import re
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from bs4 import BeautifulSoup

from src.models import JobCreate


# Tracking query parameters to strip from URLs
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gh_src",
    "lever-source",
    "ref",
    "fbclid",
    "gclid",
    "source",
}


def clean_html(raw_html: Optional[str]) -> str:
    """Convert raw HTML job description to clean, structured plain text.

    Preserves paragraph breaks, list bullets, and section headers
    while removing scripts, styles, and markup.

    Args:
        raw_html: String containing HTML or plain text.

    Returns:
        Cleaned plain text description.
    """
    if not raw_html:
        return ""

    # Unescape HTML entities first (e.g. &amp;, &lt;, &nbsp;)
    text = html.unescape(raw_html)

    # If it doesn't look like HTML, just clean whitespace
    if "<" not in text or ">" not in text:
        return clean_whitespace(text)

    try:
        soup = BeautifulSoup(text, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        # Add newline before block elements and bullets before list items
        for li in soup.find_all("li"):
            li.insert_before("\n• ")
        for block in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "br"]):
            block.insert_before("\n")

        cleaned = soup.get_text()
        return clean_whitespace(cleaned)
    except Exception:
        # Fallback to regex tag stripping if parser encounters unexpected syntax
        cleaned = re.sub(r"<[^>]+>", " ", text)
        return clean_whitespace(cleaned)


def clean_whitespace(text: Optional[str]) -> str:
    """Normalize whitespace by stripping leading/trailing spaces and collapsing excessive empty lines."""
    if not text:
        return ""

    # Replace carriage returns and tabs
    text = text.replace("\r", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    # Collapse multiple consecutive empty lines into a single blank line
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned_lines.append("")
                prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False

    return "\n".join(cleaned_lines).strip()


def normalize_url(url: Optional[str]) -> str:
    """Normalize and canonicalize URL by stripping tracking parameters, lowercase domain, and trailing slashes."""
    if not url:
        return ""

    url = url.strip()
    if not url:
        return ""

    try:
        parsed = urlparse(url)
        # Ensure scheme
        scheme = parsed.scheme.lower() if parsed.scheme else "https"
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")

        # Filter out tracking query parameters
        query_dict = parse_qs(parsed.query, keep_blank_values=False)
        cleaned_query_items = {
            k: v for k, v in query_dict.items() if k.lower() not in TRACKING_PARAMS
        }
        cleaned_query = urlencode(cleaned_query_items, doseq=True)

        return urlunparse((scheme, netloc, path, "", cleaned_query, ""))
    except Exception:
        return url.strip().rstrip("/")


def normalize_location(location: Optional[str]) -> str:
    """Standardize location strings, defaulting unspecified or blank values to 'Remote / Unspecified'."""
    if not location:
        return "Remote / Unspecified"

    cleaned = clean_whitespace(location)
    if not cleaned or cleaned.lower() in ("null", "none", "n/a", "undefined"):
        return "Remote / Unspecified"

    return cleaned


def normalize_employment_type(emp_type: Optional[str]) -> str:
    """Categorize employment type into standard terms: Internship, Full-time, Part-time, Contract."""
    if not emp_type:
        return "Full-time"

    lowered = emp_type.lower()
    if "intern" in lowered or "trainee" in lowered or "co-op" in lowered or "coop" in lowered:
        return "Internship"
    if "contract" in lowered or "temp" in lowered or "freelance" in lowered:
        return "Contract"
    if "part" in lowered:
        return "Part-time"
    return "Full-time"


def normalize_job_dict(raw: Dict[str, Any]) -> JobCreate:
    """Normalize raw job attributes dictionary into a validated JobCreate model.

    Args:
        raw: Dictionary containing raw job data from collectors.

    Returns:
        Validated JobCreate instance.
    """
    title = clean_whitespace(str(raw.get("title", "")))
    company = clean_whitespace(str(raw.get("company", "")))
    location = normalize_location(raw.get("location"))
    description = clean_html(raw.get("description", ""))
    url = normalize_url(raw.get("url", ""))
    source = clean_whitespace(str(raw.get("source", "unknown"))).lower()
    source_job_id = str(raw.get("source_job_id", "")).strip() or None
    employment_type = normalize_employment_type(raw.get("employment_type"))
    salary = clean_whitespace(str(raw.get("salary", ""))) or None

    # Handle application URL
    app_url = raw.get("application_url")
    application_url = normalize_url(app_url) if app_url else None

    # Handle posted date
    posted_date = raw.get("posted_date")
    if isinstance(posted_date, (int, float)):
        # Milliseconds epoch timestamp handling (common in Lever API)
        if posted_date > 1e11:
            posted_date = datetime.utcfromtimestamp(posted_date / 1000.0)
        else:
            posted_date = datetime.utcfromtimestamp(posted_date)
    elif isinstance(posted_date, str):
        try:
            # Handle ISO formats
            posted_date = datetime.fromisoformat(posted_date.replace("Z", "+00:00"))
        except Exception:
            posted_date = None

    return JobCreate(
        title=title,
        company=company,
        location=location,
        description=description,
        url=url,
        source=source,
        source_job_id=source_job_id,
        employment_type=employment_type,
        salary=salary,
        posted_date=posted_date,
        application_url=application_url,
    )
