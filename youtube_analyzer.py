#!/usr/bin/env python3
import os
import json
import argparse
import sys
import urllib.parse as urlparse
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
  return None

def analyze_video(client, video_url, model="gemini-2.5-flash", output_dir="outputs", index=1):
  """Analyzes a single YouTube video using the Gemini API."""
  print(f"\n{'='*40}")
  print(f"Analyzing Video: {video_url}")
  print(f"{'='*40}\n")

  video_id = get_video_id(video_url)
  filename = f"video_{video_id}.md" if video_id else f"output_{index}.md"
  filepath = os.path.join(output_dir, filename)

  if output_dir:
      os.makedirs(output_dir, exist_ok=True)
      print(f"Saving output to {filepath}...\n")

  # Setup content
  msg1_video1 = types.Part.from_uri(
      file_uri=video_url,
      mime_type="video/*",
  )
  
  prompt = """Please provide a detailed transcript for the video with timestamp markers. 
Then, extract a table of key insights and takeaways with time citations referencing the video."""

  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_video1,
        types.Part.from_text(text=prompt)
      ]
    ),
  ]

  # Configuration similar to original script, but with modern Defaults
  generate_content_config = types.GenerateContentConfig(
    temperature = 0.7,  # Default reasonable temp for extraction tasks
    safety_settings = [
       types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
       types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
       types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
       types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ]
  )

  try:
      with open(filepath, 'w', encoding='utf-8') if output_dir else sys.stdout as f_out:
          if output_dir:
                f_out.write(f"# Analysis for {video_url}\n\n")
                f_out.flush()

          for chunk in client.models.generate_content_stream(
            model = model,
            contents = contents,
            config = generate_content_config,
            ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            text = chunk.text
            print(text, end="", flush=True)
            if output_dir:
                 f_out.write(text)
                 f_out.flush()
          print("\n")
  except Exception as e:
      print(f"\nError analyzing {video_url}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Bulk Analyze YouTube Videos using Gemini API")
    parser.add_argument("--urls_file", default="youtube_urls.json", help="Path to JSON file with list of URLs")
    parser.add_argument("--env", default=".env", help="Path to .env file containing credentials")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use (e.g. gemini-3.0-pro or gemini-2.5-flash)")
    parser.add_argument("--output_dir", default="outputs", help="Directory where markdown reports are saved. Provide empty string to disable.")
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
        analyze_video(client, url, model=args.model, output_dir=args.output_dir, index=i)

    return 0

if __name__ == "__main__":
    sys.exit(main())
