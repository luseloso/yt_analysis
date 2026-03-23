#!/usr/bin/env python3
import os
import json
import argparse
import sys
import subprocess
import urllib.parse as urlparse
import asyncio
from google import genai
from google.genai import types

def load_dotenv(filename=".env"):
  """Loads environment variables from a .env file into os.environ."""
  if os.path.exists(filename):
    print(f"Loading environment from {filename}...")
    with open(filename) as f:
      for line in f:
        # Strip whitespace and ignore comments/empty lines
        line = line.strip()
        if not line or line.startswith("#"):
          continue
        # Split key and value on the first '='
        if "=" in line:
          key, value = line.split("=", 1)
          # Clean up quotes if present around string values
          value = value.strip().strip("'").strip('"')
          os.environ[key.strip()] = value
  else:
    print(f"No .env file found at {filename}. Proceeding with ambient environment variables.")

def get_video_id(url):
  """Extracts video ID from a YouTube URL."""
  try:
    parsed = urlparse.urlparse(url)
    if parsed.hostname == 'youtu.be':
        return parsed.path[1:]
    if parsed.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed.path == '/watch':
            return urlparse.parse_qs(parsed.query).get('v', [None])[0]
        if parsed.path.startswith('/embed/'):
            return parsed.path.split('/')[2]
        if parsed.path.startswith('/v/'):
            return parsed.path.split('/')[2]
  except Exception:
    pass
def get_video_duration(url):
  """Fetches video duration in seconds using yt-dlp."""
  try:
    result = subprocess.run(
        ["yt-dlp", "--print", "%(duration)s", url],
        capture_output=True,
        text=True,
        check=True
    )
    return int(result.stdout.strip())
  except Exception as e:
    print(f"Warning: Could not fetch video duration: {e}", file=sys.stderr)
    return None

async def analyze_video(client, video_url, model="gemini-2.5-flash", output_dir="outputs", index=1, chunk_size=600, template="transcript"):
  """Analyzes a single YouTube video using the Gemini API."""
  print(f"\n{'='*40}")
  print(f"Analyzing Video: {video_url}")
  print(f"{'='*40}\n")

  video_id = get_video_id(video_url)
  template_suffix = f"_{template}" if template != "transcript" else ""
  filename = f"video_{video_id}{template_suffix}.md" if video_id else f"output_{index}{template_suffix}.md"
  filepath = os.path.join(output_dir, filename)

  if output_dir:
      os.makedirs(output_dir, exist_ok=True)
      print(f"Saving output to {filepath}...\n")

  duration = get_video_duration(video_url)
  if duration:
      chunks = [(s, min(s + chunk_size, duration)) for s in range(0, duration, chunk_size)]
      print(f"Auto Partitioning: {len(chunks)} intervals of {chunk_size} seconds each.\n")
  else:
      chunks = [(None, None)]

  generate_content_config = types.GenerateContentConfig(
    temperature = 0.7,
    safety_settings = [
       types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
       types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
       types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
       types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
  )

  async def process_chunk(start, end, i):
      try:
          if start is not None and end is not None:
              print(f"--- Processing Chunk {i}/{len(chunks)}: {start}s to {end}s ---")
              msg1_video1 = types.Part.from_uri(file_uri=video_url, mime_type="video/*")
              msg1_video1.video_metadata = types.VideoMetadata(start_offset=f"{start}s", end_offset=f"{end}s")
          else:
              msg1_video1 = types.Part.from_uri(file_uri=video_url, mime_type="video/*")

          # Dynamic Prompt Templates
          prompt_templates = {
              "transcript": f"""You are transcribing a video segment to text.
The clip you are listening to starts at exactly {start if start is not None else 0} seconds in the full video.

Please provide a detailed transcript for ONLY this segment.
Guidelines:
1. Format each line with ABSOLUTE timecode markers in the full video timeline structure.
   - Example: if a word is spoken 5 seconds into this clip, and start is 60s, output [01:05] (65 seconds).
2. Layout format: `[MM:SS] Speaker: Text`
3. DO NOT include introductory filler (e.g., "Here is your transcript:"). Output ONLY raw transcript lines.
4. Try to end on complete sentence bounds gracefully where possible.
""",
              "insights": f"""Provide a structured analysis table of Key Takeaways and Insights for this segment.
The clip you are listening to starts at exactly {start if start is not None else 0} seconds in the full video.

Guidelines:
1. Extract a Markdown Table with columns: `Absolute Timecode`, `Key Takeaway`, and `Multimodal Evidence (Visual/Audio)`.
2. Use ABSOLUTE timeline markers (e.g., [01:05]) continuous to the start offset index {start if start is not None else 0}.
3. Note any distinct visual elements on screen (charts, text banners, speaker attire) if they enrich the point.
4. DO NOT include introductory filler text. Output ONLY the table.
""",
              "chapters": f"""Analyze this video segment and formulate YouTube-style Chapter Markers.
The clip you are listening to starts at exactly {start if start is not None else 0} seconds in the full video.

Guidelines:
1. Identify topic shifts or speaker changes.
2. Outline absolute start bounds `[MM:SS]` for each sub-topic.
3. Frame format: `[MM:SS] Chapter Name: Continuous description summary sentence.`
4. DO NOT include introductory filler text. Output ONLY chapter list.
"""
          }
          
          prompt = prompt_templates.get(template, prompt_templates["transcript"])

          contents = [
            types.Content(role="user", parts=[msg1_video1, types.Part.from_text(text=prompt)]),
          ]

          response = await client.aio.models.generate_content(
            model = model,
            contents = contents,
            config = generate_content_config,
          )
          print(f"✅ Chunk {i} ({start}s-{end}s) completed.")
          return i, start, end, response.text
      except Exception as e:
          print(f"❌ Chunk {i} FAILED: {e}", file=sys.stderr)
          return i, start, end, f"\n> [!WARNING]\n> Chunk {i} ({start}s-{end}s) streaming failed to complete: {e}\n\n"

  try:
      # Fire async tasks in Parallel
      tasks = [process_chunk(s, e, i) for i, (s, e) in enumerate(chunks, start=1)]
      results = await asyncio.gather(*tasks)

      # Sort by chunk index to ensure continuous timeline output
      sorted_results = sorted(results, key=lambda x: x[0])

      if output_dir:
           with open(filepath, 'w', encoding='utf-8') as f_out:
                f_out.write(f"# Analysis for {video_url}\n\n")
                for i, start, end, text in sorted_results:
                     if start is not None:
                         f_out.write(f"\n## Transcript Segment ({start} - {end} seconds)\n\n")
                     f_out.write(text)
                     f_out.write("\n")
                f_out.flush()
           print(f"Saved aggregated output to {filepath}\n")
      else:
           print(f"\n# Analysis for {video_url}\n")
           for i, start, end, text in sorted_results:
                if start is not None:
                    print(f"\n## Transcript Segment ({start} - {end} seconds)\n")
                print(text)
           print("\n")
  except Exception as e:
      print(f"\nError aggregating results for {video_url}: {e}", file=sys.stderr)

async def main():
    parser = argparse.ArgumentParser(description="Bulk Analyze YouTube Videos using Gemini API")
    parser.add_argument("--urls_file", default="youtube_urls.json", help="Path to JSON file with list of URLs")
    parser.add_argument("--env", default=".env", help="Path to .env file containing credentials")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use (e.g. gemini-3.0-pro or gemini-2.5-flash)")
    parser.add_argument("--output_dir", default="outputs", help="Directory where markdown reports are saved. Provide empty string to disable.")
    parser.add_argument("--chunk_size", type=int, default=600, help="Chunk size in seconds for offsetting (default: 600s = 10m)")
    parser.add_argument("--template", choices=['transcript', 'insights', 'chapters'], default='transcript', help="Prompt template to use (transcript, insights, chapters)")
    args = parser.parse_args()

    # 1. Load Environment file if provided
    load_dotenv(args.env)

    # 2. Check for basic auth variable just as a warning if missing
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")

    if not project_id and not api_key:
        print("Warning: Neither GOOGLE_CLOUD_PROJECT nor GOOGLE_CLOUD_API_KEY found in environment.", file=sys.stderr)
        print("Make sure you have set up your credentials or loaded them via --env.", file=sys.stderr)

    # 3. Load URLs from JSON
    if not os.path.exists(args.urls_file):
        print(f"Error: URLs file not found at {args.urls_file}", file=sys.stderr)
        return 1

    try:
        with open(args.urls_file, 'r') as f:
            urls = json.load(f)
    except Exception as e:
        print(f"Error parsing JSON from {args.urls_file}: {e}", file=sys.stderr)
        return 1

    if not isinstance(urls, list):
        print(f"Error: Expected a list in {args.urls_file}, found {type(urls).__name__}", file=sys.stderr)
        return 1

    if not urls:
        print("No URLs found to process.", file=sys.stderr)
        return 0

    # 4. Initialize Gemini Client
    # Using vertexai=True is recommended as per the user's initial setup
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "1") == "1"
    
    print(f"Initializing Gemini Client (Vertex AI={use_vertex})...")
    client = genai.Client(
        vertexai=use_vertex,
        # api_key parameter can be passed if needed, but SDK usually picks it up or uses ADC for Vertex
    )

    # 5. Process Each URL
    for i, url in enumerate(urls, start=1):
        await analyze_video(client, url, model=args.model, output_dir=args.output_dir, index=i, chunk_size=args.chunk_size, template=args.template)

    return 0

if __name__ == "__main__":
    asyncio.run(main())
