import os
from dotenv import load_dotenv
from agents.router_agent import get_router_agent
import asyncio
import sys

# Load environment logic
load_dotenv(".env")

# 2. Check for basic auth variable just as a warning if missing
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")

if not project_id and not api_key:
    print("Warning: Neither GOOGLE_CLOUD_PROJECT nor GOOGLE_CLOUD_API_KEY found in environment.", file=sys.stderr)
    print("Make sure you have set up your credentials.", file=sys.stderr)

model_env = os.environ.get("GOOGLE_ADK_MODEL", "gemini-2.5-flash")

# Instantiate and export the ADK router agent
agent = get_router_agent(model_env)

async def run_cli():
    print(f"✅ Google ADK Hierarchy loaded using `{model_env}`.")
    print("Welcome to the YouTube Analyzer Agent. Type your request (e.g., 'Extract insights from https://youtube.com/...').")
    print("Type 'exit' or 'quit' to exit.")
    
    try:
        from google.adk.runners import InMemoryRunner
        runner = InMemoryRunner(agent=agent)
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input.strip():
                    continue
                
                print("\nAgent is thinking...")
                events = await runner.run_debug(user_input, quiet=True)
                # Print the final output from the router agent
                for event in events:
                    if event.author == agent.name and event.content:
                        for part in event.content.parts:
                            if part.text:
                                print(f"\n{agent.name}: {part.text}")
            except EOFError:
                break
    except Exception as e:
        print(f"Error starting ADK execution loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
     asyncio.run(run_cli())
