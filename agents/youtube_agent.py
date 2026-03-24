import logging
from google.adk.agents.llm_agent import LlmAgent
from tools.video_tools import extract_youtube_chunks_api

logger = logging.getLogger("youtube_agent")

def get_youtube_agent(model: str = "gemini-2.5-flash") -> LlmAgent:
    """
    Returns an ADK LlmAgent configured to extract raw transcripts or insights from a YouTube video URL.
    """
    system_instruction = """
You are a Video Data Extraction Worker.
Your sole responsibility is to extract formatted analysis (transcripts, insights, or chapters) from a YouTube video given its URL.
You must use the `extract_youtube_chunks_api` tool to fetch this data.
When the user asks for analysis of a url, trigger the tool with the appropriate template argument (e.g. 'insights', 'chapters', 'transcript').
Return the complete retrieved output back to the user or orchestrator exactly as retrieved without summarizing it, so that the raw precision is preserved.
"""

    agent = LlmAgent(
        name="youtube_agent",
        model=model,
        instruction=system_instruction,
        tools=[extract_youtube_chunks_api]
    )
    return agent
