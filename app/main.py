from fastapi import FastAPI, HTTPException, Request, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx
import os
from dotenv import load_dotenv
import re
import logging
import traceback
from fastapi.middleware.cors import CORSMiddleware
import base64
import xml.etree.ElementTree as ET
import json
from typing import Optional, Union, Dict, Any, List
from pydantic import BaseModel
from bs4 import BeautifulSoup
import html
from urllib.parse import parse_qs, urlparse
import subprocess
import sys
import aiohttp
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get YouTube API key
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    logger.error("YouTube API key not found in environment variables!")

# Check if youtube_transcript_api is installed, if not install it
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    logger.info("YouTubeTranscriptApi already installed")
except ImportError:
    logger.info("Installing YouTubeTranscriptApi...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "youtube-transcript-api"])
    from youtube_transcript_api import YouTubeTranscriptApi

# Add OpenAI integration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="YouTube Transcript Processor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you might want to restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/.well-known", StaticFiles(directory=".well-known"), name="well-known")
templates = Jinja2Templates(directory="templates")

class VideoURL(BaseModel):
    video_url: str

class TranscriptSegment(BaseModel):
    start_time: int  # milliseconds
    duration: int    # milliseconds
    text: str

class TranscriptEnhanceRequest(BaseModel):
    segments: List[TranscriptSegment]
    language: str = "dutch"  # default to Dutch since your video has Dutch captions

def extract_video_id(url: str) -> str:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Regular watch URLs
        r'youtu\.be\/([0-9A-Za-z_-]{11})',   # Short URLs
        r'youtube\.com\/embed\/([0-9A-Za-z_-]{11})',  # Embed URLs
        r'youtube\.com\/shorts\/([0-9A-Za-z_-]{11})',  # Shorts URLs
        r'youtube\.com\/v\/([0-9A-Za-z_-]{11})'  # Video URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

async def extract_captions_from_html(video_id: str) -> Optional[Dict]:
    try:
        logger.info(f"Extracting captions for video ID: {video_id}")
        
        # Use youtube-transcript-api instead of direct HTML parsing
        from youtube_transcript_api import YouTubeTranscriptApi
        
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_transcript(['nl'])  # Try Dutch first
        except:
            # If Dutch not available, get the first available transcript
            transcript = transcript_list.find_transcript(['en'])
        
        transcript_data = transcript.fetch()
        
        # Convert to our format
        events = []
        for item in transcript_data:
            events.append({
                'start': item['start'],
                'text': item['text']
            })
        
        return {'events': events}
        
    except Exception as e:
        logger.error(f"Error extracting captions: {str(e)}")
        return None

async def get_captions_from_api(video_id: str) -> Optional[Dict]:
    try:
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            logger.error("YouTube API key not found in environment variables!")
            return None
            
        # First, get the caption track ID
        url = f"https://www.googleapis.com/youtube/v3/captions?videoId={video_id}&part=snippet&key={api_key}"
        logger.info(f"Fetching caption tracks from: {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to fetch captions from API. Status: {response.status}, Error: {error_text}")
                    return None
                    
                data = await response.json()
                logger.info(f"Received caption tracks data: {data}")
                
                if not data.get('items'):
                    logger.error("No caption tracks found")
                    return None
                
                # Get the first caption track
                caption_track = data['items'][0]
                caption_id = caption_track['id']
                logger.info(f"Found caption track ID: {caption_id}")
                
                # Now get the actual transcript
                transcript_url = f"https://www.googleapis.com/youtube/v3/captions/{caption_id}?key={api_key}"
                logger.info(f"Fetching transcript from: {transcript_url}")
                
                async with session.get(transcript_url) as transcript_response:
                    if transcript_response.status != 200:
                        error_text = await transcript_response.text()
                        logger.error(f"Failed to fetch transcript. Status: {transcript_response.status}, Error: {error_text}")
                        return None
                    
                    transcript_data = await transcript_response.text()
                    logger.info(f"Received transcript data: {transcript_data[:200]}...")  # Log first 200 chars
                    
                    # Parse the transcript data
                    events = []
                    for line in transcript_data.split('\n'):
                        if line.strip():
                            try:
                                # Parse timestamp and text
                                timestamp, text = line.split(' ', 1)
                                start_time = float(timestamp)
                                events.append({
                                    'start': start_time,
                                    'text': text.strip()
                                })
                            except ValueError as e:
                                logger.error(f"Error parsing line '{line}': {str(e)}")
                                continue
                    
                    logger.info(f"Parsed {len(events)} events from transcript")
                    return {
                        'events': events
                    }
                
    except Exception as e:
        logger.error(f"Error fetching captions from API: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_video(video_data: VideoURL):
    try:
        logger.info(f"Processing video URL: {video_data.video_url}")
        
        # Extract video ID from URL
        video_id = extract_video_id(video_data.video_url)
        if not video_id:
            logger.error("Invalid YouTube URL")
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
            
        logger.info(f"Extracted video ID: {video_id}")
        
        try:
            # Try to get Dutch transcript first
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                transcript = transcript_list.find_transcript(['nl'])
            except:
                # If Dutch not available, get the first available transcript
                transcript = transcript_list.find_transcript(['en'])
            
            transcript_data = transcript.fetch()
            
            # Convert to our format
            events = []
            for item in transcript_data:
                events.append({
                    'start': item['start'],
                    'text': item['text']
                })
            
            return {'events': events}
            
        except Exception as e:
            logger.error(f"Error fetching transcript: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=404, detail="No captions found for this video")
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

async def enhance_transcript_with_api(segments: List[TranscriptSegment], language: str) -> List[TranscriptSegment]:
    """Enhance transcript using direct OpenAI API call"""
    try:
        logger.info(f"Starting enhancement of {len(segments)} segments")
        
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not found in environment variables")
            raise ValueError("OpenAI API key is required. Set it in your .env file.")
            
        # Join segments into one text (with indices for mapping back)
        transcript_text = ""
        for i, segment in enumerate(segments):
            transcript_text += f"[{i}] {segment.text}\n"
        
        logger.info(f"Prepared transcript text ({len(transcript_text)} chars)")
        
        # Create the API request
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        # Build the payload
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": f"You are an assistant that fixes transcript formatting in {language}."},
                {"role": "user", "content": f"""
                Below is a transcript from a YouTube video in {language} with line numbers in [brackets].
                Fix the text by adding proper punctuation, capitalization, and fixing obvious grammar errors.
                IMPORTANT: Keep the line numbers [0], [1], etc. at the beginning of each line.
                DO NOT change the meaning or add/remove content.
                
                Transcript:
                {transcript_text}
                """}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        logger.info("Sending request to OpenAI")
        
        # Make the API call
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(api_url, json=payload, headers=headers)
                logger.info(f"OpenAI response status: {response.status_code}")
                
                if response.status_code != 200:
                    error_text = response.text
                    logger.error(f"OpenAI API error: {error_text}")
                    raise ValueError(f"OpenAI API returned {response.status_code}: {error_text}")
                    
                # Parse the response
                result = response.json()
                enhanced_text = result["choices"][0]["message"]["content"].strip()
                logger.info(f"Received enhanced text ({len(enhanced_text)} chars)")
                
                # Process the enhanced text, keeping line mappings
                enhanced_segments = [TranscriptSegment(
                    start_time=segment.start_time,
                    duration=segment.duration,
                    text=segment.text  # Default to original
                ) for segment in segments]
                
                # Parse enhanced lines and update segments
                for line in enhanced_text.split('\n'):
                    # Extract index from [index] format
                    match = re.search(r'\[(\d+)\](.*)', line)
                    if match:
                        idx = int(match.group(1))
                        text = match.group(2).strip()
                        if 0 <= idx < len(enhanced_segments):
                            enhanced_segments[idx].text = text
                
                logger.info(f"Successfully enhanced {len(enhanced_segments)} segments")
                return enhanced_segments
                
            except httpx.RequestError as e:
                logger.error(f"HTTP request error: {str(e)}")
                raise ValueError(f"Error connecting to OpenAI: {str(e)}")
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                raise ValueError(f"Invalid response from OpenAI: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in enhance_transcript_with_api: {str(e)}")
        logger.error(traceback.format_exc())
        raise  # Re-raise to be caught by the endpoint

def enhance_transcript_locally(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """Enhance transcript using local rules."""
    try:
        logger.info(f"Enhancing {len(segments)} segments locally")
        enhanced_segments = []
        
        for segment in segments:
            text = segment.text
            
            # Basic text cleaning and enhancement rules
            # 1. Capitalize first letter of sentences
            if text and len(text) > 0:
                text = text[0].upper() + text[1:]
            
            # 2. Add period at the end if missing
            if text and not text.endswith(('.', '!', '?')):
                text += '.'
            
            # 3. Fix common spacing issues
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            text = re.sub(r'\s([.,!?])', r'\1', text)  # Remove space before punctuation
            
            # 4. Fix capitalization after periods
            def capitalize_after_period(match):
                return match.group(1) + match.group(2).upper()
            
            text = re.sub(r'([.!?])\s+([a-z])', capitalize_after_period, text)
            
            # Create a new segment with enhanced text
            enhanced_segments.append(TranscriptSegment(
                start_time=segment.start_time,
                duration=segment.duration,
                text=text
            ))
        
        logger.info(f"Enhanced {len(enhanced_segments)} segments locally")
        return enhanced_segments
        
    except Exception as e:
        logger.error(f"Error in local enhancement: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to enhance transcript locally: {str(e)}")

@app.post("/enhance-transcript")
async def enhance_transcript(request: TranscriptEnhanceRequest):
    """Enhance transcript with local rules"""
    try:
        logger.info(f"Received request to enhance {len(request.segments)} segments in {request.language}")
        
        if not request.segments:
            return JSONResponse(
                status_code=400,
                content={"detail": "No transcript segments provided"}
            )
            
        # Use local enhancement
        enhanced_segments = enhance_transcript_locally(request.segments)
        
        # Convert to the format expected by the frontend
        events = []
        for segment in enhanced_segments:
            events.append({
                "tStartMs": segment.start_time,
                "dDurationMs": segment.duration,
                "segs": [{"utf8": segment.text}]
            })
            
        logger.info(f"Returning {len(events)} enhanced events")
        return {"events": events}
        
    except ValueError as e:
        logger.error(f"Value error in enhance_transcript: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"detail": str(e)}
        )
    except Exception as e:
        logger.error(f"Unexpected error in enhance_transcript: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

@app.post("/simple-enhance")
async def enhance_transcript(transcript_data: dict):
    try:
        transcript = transcript_data.get('transcript', '')
        if not transcript:
            raise HTTPException(status_code=400, detail="No transcript provided")

        # Split into lines and enhance each line
        lines = transcript.split('\n')
        enhanced_lines = []
        
        for line in lines:
            if line.strip():
                # Capitalize first letter
                enhanced_line = line[0].upper() + line[1:]
                # Add period if missing
                if not enhanced_line.endswith(('.', '!', '?')):
                    enhanced_line += '.'
                enhanced_lines.append(enhanced_line)

        # Join lines back together
        enhanced_text = '\n'.join(enhanced_lines)
        
        return Response(content=enhanced_text, media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error enhancing transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve OpenAPI spec
@app.get("/openapi.json")
async def get_openapi_spec():
    with open("app/static/openapi.json", "r") as f:
        return JSONResponse(content=json.loads(f.read()))

# Serve plugin manifest
@app.get("/.well-known/ai-plugin.json")
async def get_plugin_manifest():
    with open(".well-known/ai-plugin.json", "r") as f:
        return JSONResponse(content=json.loads(f.read()))

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    # Extract YouTube URL from message, process, and return transcript/enhanced transcript
    # Example logic:
    video_url = extract_video_url_from_message(message)
    if not video_url:
        return {"response": "Please provide a valid YouTube URL."}
    video_id = extract_video_id(video_url)
    captions = await extract_captions_from_html(video_id)
    if not captions or not captions.get("events"):
        return {"response": "No captions found for this video."}
    # Optionally enhance transcript here
    return {
        "response": "Here is the transcript:",
        "transcript": captions
    }

def extract_video_url_from_message(message: str) -> Optional[str]:
    """
    Extract YouTube video URL from a user message.
    Handles various YouTube URL formats and common message patterns.
    
    Args:
        message (str): The user's message text
        
    Returns:
        Optional[str]: The extracted YouTube URL if found, None otherwise
    """
    # Common YouTube URL patterns
    youtube_patterns = [
        r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'(https?://)?(www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
        r'(https?://)?(www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})'
    ]
    
    # Combine all patterns
    combined_pattern = '|'.join(f'({pattern})' for pattern in youtube_patterns)
    
    # Find all matches in the message
    matches = re.finditer(combined_pattern, message)
    
    # Extract the first valid URL
    for match in matches:
        # Get the full match
        url = match.group(0)
        
        # If URL doesn't start with http, add it
        if not url.startswith('http'):
            url = 'https://' + url
            
        # Validate the URL format
        if any(pattern in url for pattern in ['youtube.com/watch?v=', 'youtu.be/', 'youtube.com/embed/', 'youtube.com/shorts/', 'youtube.com/v/']):
            return url
    
    return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 