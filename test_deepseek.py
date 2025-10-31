import os
import json
import requests
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not set")

    url = "https://api.deepseek.com/chat/completions"
    payload = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": 'Return a JSON object {"answer": 1}.'},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    print("status:", resp.status_code)
    print("body:", resp.text)


if __name__ == "__main__":
    main()
