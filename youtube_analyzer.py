#!/usr/bin/env python3
import os
import json
import argparse
import sys
import asyncio
import time
from dotenv import load_dotenv

load_dotenv('.env', override=True)
from tools.video_tools import (
    extract_youtube_chunks_api,
    get_video_id,
    get_video_duration
)
from tools.history_manager import save_run_to_history

async def process_video_url(
    url: str,
    index: int,
    total: int,
    model: str,
    template: str,
    chunk_size: int,
    max_concurrency: int,
    output_dir: str
):
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] Processing: {url}")
    print(f"{'='*60}")

    video_id = get_video_id(url)
    template_suffix = f"_{template}" if template != "financial" else "_financial"
    filename = f"video_{video_id}{template_suffix}.md" if video_id else f"output_{index}{template_suffix}.md"
    filepath = os.path.join(output_dir, filename) if output_dir else None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output Target: {filepath}")

    start_time = time.time()
    
    def on_progress(chunk_idx, total_chunks, msg):
        print(f"  [Chunk {chunk_idx}/{total_chunks}] {msg}")

    try:
        report_markdown = await extract_youtube_chunks_api(
            video_url=url,
            model=model,
            template=template,
            chunk_size=chunk_size,
            max_concurrency=max_concurrency,
            progress_callback=on_progress
        )

        elapsed = time.time() - start_time
        print(f"✅ Completed in {elapsed:.1f}s")

        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f_out:
                f_out.write(report_markdown)
            print(f"💾 Saved report to {filepath}\n")
        else:
            print("\n" + report_markdown + "\n")

        duration = get_video_duration(url)
        save_run_to_history(
            video_url=url,
            video_id=video_id,
            template=template,
            model=model,
            duration_seconds=duration,
            elapsed_seconds=elapsed,
            report_markdown=report_markdown,
            output_file=filepath
        )

    except Exception as e:
        print(f"❌ Error processing {url}: {e}", file=sys.stderr)

async def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Multimodal Video Intelligence Analyzer using Google Gemini & Vertex AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--urls_file", default="youtube_urls.json", help="Path to JSON file with array of video URLs")
    parser.add_argument("--url", default=None, help="Single YouTube URL to analyze directly")
    parser.add_argument("--env", default=".env", help="Path to .env configuration file")
    parser.add_argument("--model", default="gemini-3.7-flash", help="Gemini model (e.g. gemini-3.7-flash, gemini-2.5-flash, gemini-2.5-pro)")
    parser.add_argument("--template", choices=['financial', 'insights', 'chapters', 'transcript'], default='financial',
                        help="Analysis template to execute")
    parser.add_argument("--chunk_size", type=int, default=350, help="Chunk interval size in seconds for parallel offsetting")
    parser.add_argument("--max_concurrency", type=int, default=5, help="Maximum concurrent Gemini API requests (rate-limit safeguard)")
    parser.add_argument("--output_dir", default="outputs", help="Directory where markdown reports are saved (pass '' for stdout only)")

    args = parser.parse_args()

    # 1. Load environment variables
    if os.path.exists(args.env):
        load_dotenv(args.env)
    else:
        print(f"Notice: {args.env} not found; proceeding with ambient environment variables.", file=sys.stderr)

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")

    if not project_id and not api_key:
        print("Warning: Neither GOOGLE_CLOUD_PROJECT nor GOOGLE_CLOUD_API_KEY found in environment.", file=sys.stderr)

    # 2. Collect target URLs
    if args.url:
        urls = [args.url]
    else:
        if not os.path.exists(args.urls_file):
            print(f"Error: URLs file not found at {args.urls_file}", file=sys.stderr)
            return 1
        try:
            with open(args.urls_file, 'r') as f:
                urls = json.load(f)
        except Exception as e:
            print(f"Error parsing JSON from {args.urls_file}: {e}", file=sys.stderr)
            return 1

    if not isinstance(urls, list) or not urls:
        print("No valid URLs found to process.", file=sys.stderr)
        return 0

    print(f"🚀 Starting Video Intelligence Pipeline")
    print(f"   Model: {args.model} | Template: {args.template} | Chunk Size: {args.chunk_size}s | Concurrency Cap: {args.max_concurrency}")
    print(f"   Total Videos: {len(urls)}")

    for i, u in enumerate(urls, start=1):
        await process_video_url(
            url=u,
            index=i,
            total=len(urls),
            model=args.model,
            template=args.template,
            chunk_size=args.chunk_size,
            max_concurrency=args.max_concurrency,
            output_dir=args.output_dir
        )

    return 0

if __name__ == "__main__":
    asyncio.run(main())
