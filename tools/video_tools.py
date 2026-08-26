import urllib.parse as urlparse
import subprocess
import sys
import os
import json
import asyncio
from typing import Optional, List, Dict, Any, Callable
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv('.env', override=True)

# ==========================================
# Pydantic Schemas for Structured Output
# ==========================================

class FinancialMetric(BaseModel):
    metric: str = Field(description="Name of the financial metric, e.g. 'Q4 Revenue', 'Diluted EPS', 'Data Center Revenue', 'Gaming Revenue'")
    reported_value: str = Field(description="Reported actual value, e.g. '$68.1B', '$1.62'")
    consensus_estimate: Optional[str] = Field(None, description="Wall Street consensus or expectation, e.g. '$65.8B', '$1.53'")
    beat_miss: Optional[str] = Field(None, description="'Beat', 'Miss', or 'In-Line'")
    segment: Optional[str] = Field(None, description="Business division or segment, e.g. 'Data Center', 'Gaming', 'Enterprise'")
    context: Optional[str] = Field(None, description="Context, YoY/QoQ growth or analyst reaction")

class GuidanceItem(BaseModel):
    metric_or_segment: str = Field(description="Forward looking metric, e.g. 'Q1 Revenue Guidance', 'Hyperscaler AI CapEx'")
    forward_period: str = Field(description="Target period, e.g. 'Q1 2027', 'FY2026'")
    guidance_range: str = Field(description="Guidance provided by company or consensus, e.g. '$78.0B', '$650B - $700B'")
    consensus_comparison: Optional[str] = Field(None, description="Comparison to prior whisper/consensus, e.g. 'vs $72.8B consensus'")
    strategic_context: Optional[str] = Field(None, description="Strategic drivers or management commentary")

class ThematicDriver(BaseModel):
    theme: str = Field(description="Thematic category: 'Supply Chain & Packaging', 'Geopolitical & Export Controls', 'Hyperscaler Demand', 'AI Monetization & SaaS'")
    summary: str = Field(description="Core takeaway regarding this theme")
    market_implication: Optional[str] = Field(None, description="Implications for industry, competitors, or valuation")

class ExecutiveQuote(BaseModel):
    speaker: str = Field(description="Speaker name and title (e.g. 'Colette Kress (CFO)', 'Patrick Moorhead (Analyst)')")
    timecode: str = Field(description="Absolute timecode in full video [MM:SS]")
    quote: str = Field(description="Direct or closely paraphrased quote")
    significance: str = Field(description="Why this statement matters")

class FinancialIntelligenceReport(BaseModel):
    video_url: str
    video_id: str
    executive_summary: str
    key_metrics: List[FinancialMetric] = Field(default_factory=list)
    guidance_and_outlook: List[GuidanceItem] = Field(default_factory=list)
    thematic_drivers: List[ThematicDriver] = Field(default_factory=list)
    executive_quotes: List[ExecutiveQuote] = Field(default_factory=list)
    key_risks_and_headwinds: List[str] = Field(default_factory=list)
    strategic_takeaways: List[str] = Field(default_factory=list)

# ==========================================
# Video Metadata & Utility Functions
# ==========================================

def get_video_id(url: str) -> str:
    """Extracts video ID from a YouTube URL."""
    try:
        parsed = urlparse.urlparse(url)
        if parsed.hostname == 'youtu.be':
            return parsed.path.lstrip('/')
        if parsed.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
            if parsed.path == '/watch':
                return urlparse.parse_qs(parsed.query).get('v', [''])[0]
            if parsed.path.startswith('/embed/'):
                return parsed.path.split('/')[2]
            if parsed.path.startswith('/v/'):
                return parsed.path.split('/')[2]
            if parsed.path.startswith('/shorts/'):
                return parsed.path.split('/')[2]
    except Exception:
        pass
    return ""

def get_video_duration(url: str) -> Optional[int]:
    """Fetches video duration in seconds using yt-dlp with graceful fallback."""
    try:
        result = subprocess.run(
            ["yt-dlp", "--print", "%(duration)s", "--no-warnings", url],
            capture_output=True,
            text=True,
            check=True,
            timeout=15
        )
        duration_str = result.stdout.strip()
        if duration_str and duration_str.isdigit():
            return int(duration_str)
    except Exception as e:
        print(f"Warning: Could not fetch video duration via yt-dlp: {e}", file=sys.stderr)
    return None

# ==========================================
# Dynamic Prompt Templates
# ==========================================

PROMPT_TEMPLATES = {
    "financial": """You are a Senior Financial Content Analyst extracting high-fidelity market and corporate earnings intelligence from this video segment.
The clip starts at exactly {start} seconds in the full video.

Extract a structured financial intelligence brief for ONLY this segment using the following sections:

### 1. Key Metrics & Financial Performance Table
| Absolute Timecode | Metric / Segment | Reported Value | Consensus / Prior | Beat / Miss / Inline | Key Commentary |
|---|---|---|---|---|---|
| [MM:SS] | e.g. Q4 Revenue | $68.1B | $65.8B | Beat | Exceeded top-end guide by 3.5% |

### 2. Forward Guidance & CapEx Outlook
| Absolute Timecode | Period / Item | Guided Value / Range | Consensus Comparison | Strategic Driver |
|---|---|---|---|---|
| [MM:SS] | e.g. Q1 Sales Guide | $78.0B | vs $72.8B consensus | Strong hyperscaler datacenter ramp |

### 3. Thematic Drivers & Strategic Headwinds/Tailwinds
- **Supply Chain & Manufacturing**: Note HBM memory, advanced packaging (CoWoS), wafer allocation, power/energy limits.
- **Geopolitics & Regulatory**: Note China export control status, licenses, competitor landscape.
- **Hyperscaler CapEx & Monetization**: Note cloud customer spend, enterprise ROI, SaaS transformation.

### 4. Direct Executive & Analyst Quotes
- `[MM:SS]` **Speaker Name (Role)**: "Verbatim or precise quote" — *Significance/Context*

Guidelines:
1. Always format timecodes with ABSOLUTE time markers mapped to the video timeline (start: {start}s).
2. DO NOT include introductory filler. Output ONLY the markdown sections above.
3. Be precise with financial figures ($ amounts, percentages, multiples).
""",

    "insights": """Provide a structured analysis table of Key Takeaways and Multimodal Insights for this segment.
The clip starts at exactly {start} seconds in the full video.

Guidelines:
1. Extract a Markdown Table with columns: `Absolute Timecode`, `Key Takeaway`, and `Multimodal Evidence (Visual/Audio)`.
2. Use ABSOLUTE timeline markers (e.g., [01:02]) mapped to the start offset ({start}s).
3. DO NOT output introductory filler text. Output ONLY the markdown table.

Example Row Syntax:
| Absolute Timecode | Key Takeaway | Multimodal Evidence (Visual/Audio) |
|---|---|---|
| [01:05] | Speaker discusses Q4 projections | Line graph trend on screen shows $68.1B total |
""",

    "chapters": """Analyze this video segment and formulate YouTube-style Chapter Markers.
The clip starts at exactly {start} seconds in the full video.

Guidelines:
1. Identify topic shifts, key speaker transitions, or thesis changes.
2. Outline absolute start bounds `[MM:SS]` for each sub-topic.
3. Format: `[MM:SS] Chapter Name: Continuous description summary sentence.`
4. DO NOT include introductory filler text. Output ONLY chapter list.
""",

    "transcript": """You are transcribing a video segment to text with absolute timeline precision.
The clip starts at exactly {start} seconds in the full video.

Please provide a detailed line-by-line transcript for ONLY this segment.
Guidelines:
1. Format each line with ABSOLUTE timecode markers in the full video timeline structure: `[MM:SS] Speaker: Text`.
2. DO NOT include introductory filler text. Output ONLY raw transcript lines.
3. Try to end on complete sentence bounds gracefully where possible.
"""
}

# ==========================================
# Segment Consolidation & Post-Processing
# ==========================================

def merge_financial_segments(segment_texts: List[str]) -> str:
    metrics_rows = []
    guidance_rows = []
    thematic_items = []
    quotes_items = []
    current_section = None
    
    for text in segment_texts:
        lines = text.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if "1." in lower and ("metric" in lower or "financial performance" in lower):
                current_section = "metrics"
                continue
            elif "2." in lower and ("guidance" in lower or "capex" in lower or "outlook" in lower):
                current_section = "guidance"
                continue
            elif "3." in lower and ("thematic" in lower or "headwinds" in lower or "tailwinds" in lower or "driver" in lower):
                current_section = "thematic"
                continue
            elif "4." in lower and ("quote" in lower or "executive" in lower):
                current_section = "quotes"
                continue
            elif stripped.startswith("###") or stripped.startswith("## ") or stripped.startswith("---"):
                current_section = "other"
                continue
                
            if current_section == "metrics":
                if stripped.startswith("|") and not ("---" in stripped or "Metric" in stripped or "metric" in stripped or "Reported" in stripped):
                    metrics_rows.append(stripped)
            elif current_section == "guidance":
                if stripped.startswith("|") and not ("---" in stripped or "Period" in stripped or "period" in stripped or "Guided" in stripped):
                    guidance_rows.append(stripped)
            elif current_section == "thematic":
                if stripped.startswith("-") or stripped.startswith("*") or stripped.startswith("•"):
                    thematic_items.append(stripped)
                elif thematic_items and not stripped.startswith("#"):
                    thematic_items[-1] += f" {stripped}"
            elif current_section == "quotes":
                if stripped.startswith("-") or stripped.startswith("*") or stripped.startswith("•") or stripped.startswith("`[") or stripped.startswith("[") or (len(stripped) > 5 and stripped[:5].replace(":", "").isdigit()):
                    quotes_items.append(stripped)
                elif quotes_items and not stripped.startswith("#"):
                    quotes_items[-1] += f" {stripped}"

    if metrics_rows or guidance_rows or thematic_items or quotes_items:
        out = []
        out.append("### 1. Key Metrics & Financial Performance Table")
        out.append("| Absolute Timecode | Metric / Segment | Reported Value | Consensus / Prior | Beat / Miss / Inline | Key Commentary |")
        out.append("|:---|:---|:---|:---|:---|:---|")
        out.extend(metrics_rows if metrics_rows else ["| N/A | No specific metric beats/misses cited | - | - | - | - |"])
        out.append("")
        
        out.append("### 2. Forward Guidance & CapEx Outlook")
        out.append("| Absolute Timecode | Period / Item | Guided Value / Range | Consensus Comparison | Strategic Driver |")
        out.append("|:---|:---|:---|:---|:---|")
        out.extend(guidance_rows if guidance_rows else ["| N/A | No forward guidance items cited | - | - | - |"])
        out.append("")
        
        out.append("### 3. Thematic Drivers & Strategic Headwinds/Tailwinds")
        out.extend(thematic_items if thematic_items else ["- *No specific thematic drivers noted.*"])
        out.append("")
        
        out.append("### 4. Direct Executive & Analyst Quotes")
        out.extend(quotes_items if quotes_items else ["- *No direct quotes extracted.*"])
        out.append("")
        
        return "\n".join(out)
    return "\n\n".join(segment_texts)

def merge_insights_segments(segment_texts: List[str]) -> str:
    rows = []
    for text in segment_texts:
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("|") and not ("---" in s or "Timecode" in s or "timecode" in s or "Takeaway" in s or "takeaway" in s):
                rows.append(s)
    if rows:
        out = [
            "### Consolidated Key Takeaways & Multimodal Evidence",
            "| Absolute Timecode | Key Takeaway | Multimodal Evidence (Visual/Audio) |",
            "|:---|:---|:---|",
            *rows
        ]
        return "\n".join(out)
    return "\n\n".join(segment_texts)

def merge_chapters_segments(segment_texts: List[str]) -> str:
    lines = []
    for text in segment_texts:
        for line in text.split("\n"):
            s = line.strip()
            if s and not s.startswith("#") and not s.startswith("---"):
                lines.append(s)
    if lines:
        return "### Consolidated Video Chapters\n\n" + "\n".join(lines)
    return "\n\n".join(segment_texts)

def merge_transcript_segments(segment_texts: List[str]) -> str:
    lines = []
    for text in segment_texts:
        for line in text.split("\n"):
            s = line.strip()
            if s and not s.startswith("#") and not s.startswith("---"):
                lines.append(s)
    if lines:
        return "### Full Video Transcript\n\n" + "\n".join(lines)
    return "\n\n".join(segment_texts)

def consolidate_segment_outputs(template: str, segment_texts: List[str]) -> str:
    if not segment_texts:
        return ""
    if len(segment_texts) == 1:
        return segment_texts[0].strip()
        
    if template == "financial":
        return merge_financial_segments(segment_texts)
    elif template == "insights":
        return merge_insights_segments(segment_texts)
    elif template == "chapters":
        return merge_chapters_segments(segment_texts)
    elif template == "transcript":
        return merge_transcript_segments(segment_texts)
    return "\n\n".join(segment_texts)

# ==========================================
# Core Parallel Chunk Processing Engine
# ==========================================

async def extract_youtube_chunks_api(
    video_url: str,
    model: str = "gemini-3.7-flash",
    template: str = "financial",
    chunk_size: int = 350,
    max_concurrency: int = 5,
    overlap_seconds: int = 0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> str:
    """
    Analyzes a YouTube video in parallel chunks using Gemini Multimodal Video API.
    Features:
      - Bounded concurrency with asyncio.Semaphore
      - Tenacity exponential backoff retries
      - Deterministic low-temperature sampling
      - Absolute timestamp alignment
      - Automated segment consolidation into unified tables
      
    Args:
        video_url: The YouTube video URL to analyze.
        model: The Gemini model name (default: gemini-3.7-flash).
        template: Prompt template ('financial', 'insights', 'chapters', 'transcript').
        chunk_size: Chunk duration in seconds (default: 350s).
        max_concurrency: Maximum simultaneous API requests (default: 5).
        overlap_seconds: Seconds of overlap between chunks (default: 0).
        progress_callback: Optional callable (chunk_index, total_chunks, status_msg).
        
    Returns:
        Aggregated Markdown report string.
    """
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "1") == "1"
    
    # Vertex AI serves gemini-3.7-* models exclusively on the global region endpoint
    if "3.7" in model or model.startswith("gemini-3.7"):
        location = "global"
    else:
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    
    client_kwargs = {"vertexai": use_vertex}
    if use_vertex:
        if location:
            client_kwargs["location"] = location
        if project:
            client_kwargs["project"] = project
            
    client = genai.Client(**client_kwargs)
    
    duration = get_video_duration(video_url)
    if duration and duration > 0:
        chunks = []
        for s in range(0, duration, chunk_size):
            start = max(0, s - overlap_seconds) if s > 0 else 0
            end = min(s + chunk_size, duration)
            chunks.append((start, end))
    else:
        chunks = [(None, None)]

    semaphore = asyncio.Semaphore(max_concurrency)

    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ]
    )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def call_gemini_with_retry(contents):
        return await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )

    async def process_chunk(start: Optional[int], end: Optional[int], chunk_idx: int, total_chunks: int):
        async with semaphore:
            try:
                start_val = start if start is not None else 0
                if progress_callback:
                    progress_callback(chunk_idx, total_chunks, f"Processing chunk {chunk_idx}/{total_chunks} ({start_val}s-{end or 'end'}s)...")

                if start is not None and end is not None:
                    part_video = types.Part.from_uri(file_uri=video_url, mime_type="video/*")
                    if duration and end == duration:
                        part_video.video_metadata = types.VideoMetadata(start_offset=f"{start}s")
                    else:
                        part_video.video_metadata = types.VideoMetadata(start_offset=f"{start}s", end_offset=f"{end}s")
                else:
                    part_video = types.Part.from_uri(file_uri=video_url, mime_type="video/*")

                template_str = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["financial"])
                prompt_text = template_str.format(start=start_val)

                contents = [
                    types.Content(role="user", parts=[part_video, types.Part.from_text(text=prompt_text)])
                ]

                response = await call_gemini_with_retry(contents)
                
                if progress_callback:
                    progress_callback(chunk_idx, total_chunks, f"Chunk {chunk_idx}/{total_chunks} completed.")

                return chunk_idx, start, end, response.text
            except Exception as e:
                err_msg = f"Chunk {chunk_idx} ({start}s-{end}s) failed: {e}"
                print(f"❌ {err_msg}", file=sys.stderr)
                return chunk_idx, start, end, f"\n> [!WARNING]\n> {err_msg}\n\n"

    tasks = [process_chunk(s, e, idx, len(chunks)) for idx, (s, e) in enumerate(chunks, start=1)]
    results = await asyncio.gather(*tasks)
    sorted_results = sorted(results, key=lambda x: x[0])

    # Compose Aggregated Markdown Report
    template_titles = {
        "financial": "Financial & Earnings Intelligence Brief",
        "insights": "Key Insights & Multimodal Evidence Report",
        "chapters": "Video Chapter Breakdown",
        "transcript": "Full Timeline Transcript"
    }
    title = template_titles.get(template, "Video Analysis Report")

    # Post-process segments into a unified seamless body
    segment_bodies = [text for _, _, _, text in sorted_results]
    unified_body = consolidate_segment_outputs(template, segment_bodies)

    final_output = [
        f"# {title}",
        f"**Source URL**: {video_url}",
        f"**Model**: `{model}` | **Template**: `{template}` | **Duration**: {f'{duration}s' if duration else 'Auto'}\n",
        "---",
        "",
        unified_body
    ]

    return "\n".join(final_output)
