from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aiohttp
import asyncio
import os
import cv2
import requests
from google.cloud import storage

# Initialize FastAPI
app = FastAPI()

# Define the data model for the request
class VideoRequest(BaseModel):
    org_id: str
    video_urls: list
    prompt_keys: list

# Helper class for handling video processing
class VideoFrameUploader:
    def __init__(self, organization_id, max_frame_count=8):
        self.organization_id = organization_id
        self.max_frame_count = max_frame_count
        self.gcs_bucket_name = "storyteller-vertex-ai"
        self.gcs_folder = f"{organization_id}_frames"
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.gcs_bucket_name)

    def create_gcs_folder(self):
        # Ensure the GCS folder exists
        blob = self.bucket.blob(f"{self.gcs_folder}/")
        blob.upload_from_string('')
        print(f"GCS folder created: {self.gcs_folder}")

    def upload_frame_to_gcs(self, local_frame_path, frame_name):
        gcs_path = f"{self.gcs_folder}/{frame_name}"
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(local_frame_path)
        public_url = f"https://storage.googleapis.com/{self.gcs_bucket_name}/{gcs_path}"
        return public_url

    def extract_and_upload_frames(self, video_url):
        # Step 1: Download video locally
        local_video_path = "downloaded_video.mp4"
        response = requests.get(video_url, stream=True)
        with open(local_video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Video downloaded to: {local_video_path}")

        # Step 2: Extract frames
        video = cv2.VideoCapture(local_video_path)
        frame_count = 0
        frame_urls = []

        os.makedirs("frames", exist_ok=True)
        while frame_count < self.max_frame_count:
            success, frame = video.read()
            if not success:
                break

            frame_name = f"frame_{frame_count + 1}.jpg"
            local_frame_path = os.path.join("frames", frame_name)
            cv2.imwrite(local_frame_path, frame)

            # Step 3: Upload each frame to GCS
            frame_url = self.upload_frame_to_gcs(local_frame_path, frame_name)
            frame_urls.append(frame_url)
            frame_count += 1

        video.release()
        return frame_urls

# Helper function to fetch prompts based on keys
async def fetch_prompts():
    PROMPTS_URL = "https://raw.githubusercontent.com/nikhilstoryteller/prompts/main/prompts.json"
    async with aiohttp.ClientSession() as session:
        async with session.get(PROMPTS_URL) as response:
            if response.status != 200:
                raise HTTPException(status_code=500, detail="Failed to fetch prompts from GitHub.")
            response_text = await response.text()
            return json.loads(response_text)

# Helper function to query the Qwen API
async def query_video_analysis(session, frame_urls, prompt_text):
    headers = {
        'accept': 'application/json',
        'content-type': 'application/json',
    }

    all_responses = []
    batch_size = 8

    for i in range(0, len(frame_urls), batch_size):
        batch_frames = frame_urls[i:i + batch_size]
        content = [{"type": "image_url", "image_url": {"url": url}} for url in batch_frames]
        content.append({"type": "text", "text": prompt_text})

        json_data = {
            "model": "qwen-qwen2-7b-instruct",
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": 0.8,
            "top_p": 1,
            "n": 1,
            "max_tokens": 2048,
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
                return f"Error: HTTP {response.status}"
            response_json = await response.json()
            if 'choices' in response_json and 'message' in response_json['choices'][0]:
                assistant_message = response_json['choices'][0]['message']['content']
                all_responses.append(assistant_message)

    return "\n".join(all_responses)

# Endpoint to handle video processing requests
@app.post("/process-videos")
async def process_videos(request: VideoRequest):
    org_id = request.org_id
    video_urls = request.video_urls
    prompt_keys = request.prompt_keys

    # Fetch all prompts
    try:
        prompts = await fetch_prompts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching prompts: {e}")

    # Validate prompt keys
    for key in prompt_keys:
        if key not in prompts:
            raise HTTPException(status_code=400, detail=f"Invalid prompt key: {key}")

    results = []

    # Process each video asynchronously
    async with aiohttp.ClientSession() as session:
        for video_url in video_urls:
            try:
                # Initialize uploader and create GCS folder
                uploader = VideoFrameUploader(org_id)
                uploader.create_gcs_folder()

                # Extract and upload frames
                frame_urls = uploader.extract_and_upload_frames(video_url)

                # Process each prompt for the current video
                video_results = []
                for prompt_key in prompt_keys:
                    prompt_text = prompts[prompt_key]
                    analysis_result = await query_video_analysis(session, frame_urls, prompt_text)
                    video_results.append({
                        "prompt_key": prompt_key,
                        "response": analysis_result
                    })

                results.append({
                    "org_id": org_id,
                    "video_url": video_url,
                    "results": video_results
                })
            except Exception as e:
                results.append({
                    "org_id": org_id,
                    "video_url": video_url,
                    "error": str(e)
                })

    return results

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
