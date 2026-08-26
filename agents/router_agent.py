import logging
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from agents.youtube_agent import get_youtube_agent
from agents.analyst_agent import get_analyst_agent

logger = logging.getLogger("router_agent")

def get_router_agent(model: str = "gemini-3.7-flash") -> LlmAgent:
    """
    Returns an ADK LlmAgent configured to route the user's prompt as either EXTRACT or SYNTHESIZE.
    """
    system_instruction = """
You are the Master Orchestrator for the YouTube Video Intelligence Pipeline.
Your goal is to evaluate the user's request and delegate to the appropriate specialist agent if needed.

- **EXTRACT**: If the user provides a YouTube URL, or asks to extract, fetch, or analyze a video (financial intelligence, earnings, guidance, transcript, insights, chapters), delegate to the `youtube_agent` by calling its tool. Do not try to answer it yourself. Pass the URL precisely and specify the appropriate template ('financial', 'insights', 'chapters', 'transcript').
- **SYNTHESIZE**: If the user wants to summarize, analyze, write an executive briefing, or synthesize a deep report from already extracted video data in the conversation, delegate to the `analyst_agent` by calling its tool. Provide the extracted context to format.

If the user is asking a general question, you can answer it directly.
If the user asks about batch processing, explain that they can run `python youtube_analyzer.py --urls_file youtube_urls.json` or query via the web application.
"""

    youtube_agent = get_youtube_agent(model)
    # Give analyst agent the more powerful reasoning model
    analyst_agent = get_analyst_agent("gemini-2.5-pro")

    agent = LlmAgent(
        name="router_agent",
        model=model,
        instruction=system_instruction,
        tools=[
            AgentTool(agent=youtube_agent),
            AgentTool(agent=analyst_agent)
        ]
    )
    return agent
