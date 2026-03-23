import os
from google import genai

def main():
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip("'").strip('"')

    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "1") == "1"
    print(f"Initializing Client (Vertex AI={use_vertex})...")
    
    try:
        client = genai.Client(vertexai=use_vertex)
        model = "gemini-2.5-flash"
        print(f"Running test prompt on model {model}...")
        response = client.models.generate_content(model=model, contents="Say 'Connection Successful'")
        print(f"\nResult: {response.text}")
    except Exception as e:
        print(f"\nConnection Failed: {e}")

if __name__ == "__main__":
    main()
