from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aiohttp
import nest_asyncio
import asyncio
import json

# Apply nest_asyncio for nested event loops in environments like Jupyter Notebooks
nest_asyncio.apply()

app = FastAPI()

# URL to your prompts.json file in GitHub (replace with your actual URL)
PROMPTS_URL = "https://raw.githubusercontent.com/nikhilstoryteller/prompts/main/prompts.json"

# Model for request body
class QwenRequest(BaseModel):
    org_id: str
    image_urls: list
    prompt_keys: list

# Function to fetch prompts from GitHub
async def fetch_prompts():
    async with aiohttp.ClientSession() as session:
        async with session.get(PROMPTS_URL) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch prompts from GitHub.")
            response_text = await response.text()
            return json.loads(response_text)

# Function to send a single request for an image and prompt
async def send_qwen_request(session, org_id, image_url, prompt_text):
    headers = {
        'accept': 'application/json',
        'content-type': 'application/json',
    }

    json_data = {
        "model": "qwen-qwen2-7b-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ],
        "temperature": 0.8,
        "top_p": 1,
        "n": 1,
        "max_tokens": 1024,
        "stop": ["string"],
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_k": -1,
        "ignore_eos": False,
        "use_beam_search": False,
        "stop_token_ids": [],
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
        "add_generation_prompt": True,
        "repetition_penalty": 1
    }

    async with session.post(
        'https://qwen2-vl-7b-instruct2-storyteller-staging-8000.app-staging.thestoryteller.tech/v1/chat/completions',
        headers=headers,
        json=json_data,
    ) as response:
        if response.status != 200:
            return org_id, image_url, f"Error: HTTP {response.status}"
        response_json = await response.json()
        if 'choices' in response_json and 'message' in response_json['choices'][0]:
            assistant_message = response_json['choices'][0]['message']['content']
            return org_id, image_url, assistant_message
        else:
            return org_id, image_url, "Error: Unexpected response structure."

# Endpoint to handle requests for sending Qwen API calls
@app.post("/process-qwen-requests")
async def process_qwen_requests(request: QwenRequest):
    org_id = request.org_id
    image_urls = request.image_urls
    prompt_keys = request.prompt_keys

    # Fetch prompts from GitHub
    try:
        prompts = await fetch_prompts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prompts: {e}")

    # Validate prompt keys
    for key in prompt_keys:
        if key not in prompts:
            raise HTTPException(status_code=400, detail=f"Invalid prompt key: {key}")

    # Process requests concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_qwen_request(session, org_id, image_url, prompts[prompt_key])
            for image_url in image_urls
            for prompt_key in prompt_keys
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Format results for the response
    formatted_results = []
    for result in results:
        if isinstance(result, Exception):
            formatted_results.append({"error": str(result)})
        else:
            org_id, image_url, response_content = result
            formatted_results.append({
                "org_id": org_id,
                "image_url": image_url,
                "response": response_content
            })

    return formatted_results

# Start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
