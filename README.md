# Bulk YouTube Analyzer

A Python tool designed to bulk extract detailed transcripts and key insights from a list of YouTube video URLs using the Google Gemini (Generative AI) API with Vertex AI.

## Overview

The `youtube_analyzer.py` script takes a JSON file containing a list of YouTube URLs and leverages the `google-genai` SDK’s multimodal understanding to process each video.

> [!IMPORTANT]
> **Video Processing Time**: Analyzing rich video content (such as a 10+ minute conference segment) requires loading and indexing heavy assets. Response stream extraction can take a minute or two to finalize fully. Let it execute continuously while buffered streams load correctly.

For every video, it produces:
1.  A detailed transcript with timestamp markers.
2.  A structured table of insights and key takeaways with clear time-referenced citations.

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

-   `--output_dir`: Sets the path where markdown reports are saved. Subtitles use the video ID name e.g. `reports/video_3MwxX1ee_gI.md`. Default is `outputs/`. Pass empty string `""` to disable and just print output.

---

## Technical Details

-   **Zero Dependency Mode**: Environment parsing is done manually inside the code logic for maximum compatibility without requiring `python-dotenv`.
-   **Streaming Mode**: Enables high-velocity live stream replies directly into the console terminal.
-   **Form Factor**: Automatically translates raw YouTube URLs into multimodal prompt containers.
