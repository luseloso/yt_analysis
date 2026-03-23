# Bulk YouTube Analyzer

A Python tool designed to bulk extract detailed transcripts and key insights from a list of YouTube video URLs using the Google Gemini (Generative AI) API with Vertex AI.

## Overview

The `youtube_analyzer.py` script takes a JSON file containing a list of YouTube URLs and leverages the `google-genai` SDK’s multimodal understanding to process each video.

To handle **arbitrarily long videos** without hitting model context caps (1M input tokens) or transcription response buffer limits (8k output tokens), the script uses **Parallel Asynchronous Offsetting**. It partitions the video into smaller contiguous sub-intervals (chunks) and streams them to the cloud concurrently in batches for speed.

For every video, it produces a detailed transcript with absolute timestamp markers mapped correctly to the full video timeline.

> [!TIP]
> **Performance Pricing Benchmarks**: For example, executing a 15-minute video partitioned with `--chunk_size 60` (12 parallel intervals) runs and aggregates on the backend nodes in roughly **45-60 seconds** total!
> 
> A full sample of the generated output can be viewed directly in `outputs/video_3MwxX1ee_gI.md`.

---

## Prerequisites

1.  **Python 3.9+** installed on your system.
2.  **Google Cloud Project**: You need credentials available (either loaded with `gcloud auth application-default login` or an API Key).

---

## Setup Instructions

### 1. (Recommended) Create a Virtual Environment

Isolate dependencies so they don’t conflict with your global Python environment:

```bash
# Create the environment
python3 -m venv .venv

# Activate it (Mac/Linux)
source .venv/bin/activate

# For Windows:
# .venv\Scripts\activate
```

### 2. Install Dependencies

Use `requirements.txt` to install the latest SDK version:

```bash
pip install -r requirements.txt
```

### 3. Configure Credentials

Create a `.env` file from the provided `.env.example` template:

```bash
cp .env.example .env
```

Open `.env` and fill in your details:

```ini
# GCP Settings
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

---

## Usage

### 1. Prepare your URLs List

Add your YouTube links into the `youtube_urls.json` file in a simple JSON array format:

```json
[
  "https://www.youtube.com/watch?v=3MwxX1ee_gI",
  "https://www.youtube.com/watch?v=example2"
]
```

### 2. Run the Analyzer Script

By default, the script looks for `.env` and `youtube_urls.json` in the current working directory:

```bash
python youtube_analyzer.py
```

### Custom Arguments

You can provide custom paths for input and configuration:

```bash
python youtube_analyzer.py --urls_file data/list.json --env secrets/.env.prod --model gemini-2.5-pro --output_dir reports/
```

-   `--output_dir`: Sets the path where markdown reports are saved. Default is `outputs/`. Pass empty string `""` to disable and just print output directly to terminal.
-   `--chunk_size`: (Seconds) The chunk interval to break the video down into. Default is `600` (10 minutes). For extremely highly continuous detailed transcripts without sentence truncation loops, you can lower it (e.g., `300` or `120`). Lower increments execute faster per chunk concurrently!
-   `--template`: Choices: `['transcript', 'insights', 'chapters']`. Default is `'transcript'`.
    -   `transcript`: Detailed line-by-line transcript with timeline sync.
    -   `insights`: Extracts a high-level Markdown Table for each segment covering takeaways and concrete **Multimodal Evidence** seen on screen.
    -   `chapters`: Formulates traditional YouTube video navigator styles with absolute descriptive bookmarks.

---

## Technical Details

-   **Parallel Async Design**: Fires multiple intervals at the exact same moment on concurrent network threads utilizing `asyncio.gather` with `.aio` client wrappers, completing processing within seconds regardless of full stream lengths.
-   **Zero Dependency Mode**: Environment parsing is done manually inside the code logic for maximum compatibility without requiring `python-dotenv`.
-   **Form Factor**: Automatically translates raw YouTube URLs into multimodal prompt containers.

---

## ⚡ Gotchas & Customizations

### 1. **Prompt Customization**
If you want to tweak the transcript output formatting/instructions, do NOT edit the top of the file.
Locate the **`process_chunk()`** local function inside `youtube_analyzer.py` (around Line 95). You can customize the prompt string template dynamically constructed inside the loop:
```python
prompt = f"""You are transcribing a video segment to text...
The clip starts at {start} seconds...
"""
```

### 2. **API Rate Limiting (Quotas)**
Because the script fires all chunks **in parallel**, a clip broken into 20 intervals fires 20 simultaneous API requests.
- **Gotcha**: If you have tightly restricted Queries-Per-Minute (QPM) on your Vertex AI pricing tier, parallel gathers might trigger heavy throttles.
- **Solution**: Increase the `--chunk_size` flag on those videos to reduce parallel multiplier calls.

### 3. **Buffered Batch Writes**
To prevent race conditions with multiple concurrent text streams overlaps overlapping and mangling files on disk, writing of your `.md` report happens **all at once buffered** at the strict end of the request execution loop rather than letter-by-letter live streaming. If the task appears to sit for 30s-40s idle, it is compiling in the cloud and preparing to dump safely!
