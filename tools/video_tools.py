import urllib.parse as urlparse
import subprocess
import sys
import asyncio
from google import genai
from google.genai import types

def get_video_id(url: str) -> str:
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
    return ""

def get_video_duration(url: str) -> int:
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

async def extract_youtube_chunks_api(video_url: str, model: str = "gemini-2.5-flash", template: str = "insights", chunk_size: int = 600) -> str:
    """
    Analyzes a YouTube video in chunks using the Gemini API.
    Returns the aggregated markdown string instead of writing to a file.
    
    Args:
        video_url: The youtube video URL to analyze.
        model: The Gemini model to use.
        template: The prompt template (transcript, insights, chapters).
        chunk_size: Size of video chunks to process in seconds.
        
    Returns:
        A Markdown string containing the analysis.
    """
    client = genai.Client()
    duration = get_video_duration(video_url)
    if duration:
        chunks = [(s, min(s + chunk_size, duration)) for s in range(0, duration, chunk_size)]
    else:
        chunks = [(None, None)]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.7,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ]
    )

    async def process_chunk(start, end, i):
        try:
            if start is not None and end is not None:
                msg1_video1 = types.Part.from_uri(file_uri=video_url, mime_type="video/*")
                if end == duration:
                    msg1_video1.video_metadata = types.VideoMetadata(start_offset=f"{start}s")
                else:
                    msg1_video1.video_metadata = types.VideoMetadata(start_offset=f"{start}s", end_offset=f"{end}s")
            else:
                msg1_video1 = types.Part.from_uri(file_uri=video_url, mime_type="video/*")

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
2. Use ABSOLUTE timeline markers (e.g., [01:02]) continuous to the start offset index {start if start is not None else 0}.
3. DO NOT output introducing filler text. Output ONLY the markdown table.

Example Row Syntax:
| Absolute Timecode | Key Takeaway | Multimodal Evidence (Visual/Audio) |
|---|---|---|
| [01:05] | Speaker discusses Q4 projections | Line graph trend on screen shows $68.1B total |
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
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            return i, start, end, response.text
        except Exception as e:
            return i, start, end, f"\n> [!WARNING]\n> Chunk {i} ({start}s-{end}s) streaming failed to complete: {e}\n\n"

    tasks = [process_chunk(s, e, i) for i, (s, e) in enumerate(chunks, start=1)]
    results = await asyncio.gather(*tasks)

    sorted_results = sorted(results, key=lambda x: x[0])
    
    final_output = f"# Analysis for {video_url}\n\n"
    for i, start, end, text in sorted_results:
        if start is not None:
            final_output += f"\n## Transcript Segment ({start} - {end} seconds)\n\n"
        final_output += text + "\n"
        
    return final_output
