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

# Load environment variables
load_dotenv()

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    """Extract video ID from YouTube URL."""
    logger.info(f"Extracting video ID from URL: {url}")
    
    # Multiple patterns to match different YouTube URL formats
    patterns = [
        r'youtube\.com\/watch\?v=([-\w]+)',           # Standard watch URL
        r'youtu\.be\/([-\w]+)',                       # Shortened URL
        r'youtube\.com\/embed\/([-\w]+)',             # Embed URL
        r'v=([-\w]+)(?:&|$)'                          # v= parameter
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            logger.info(f"Extracted video ID: {video_id}")
            return video_id
            
    logger.error(f"Could not extract video ID from URL: {url}")
    raise ValueError(f"Invalid YouTube URL: {url}")

async def extract_captions_from_html(video_id: str) -> Optional[Dict]:
    try:
        # Log the video ID
        logger.info(f"Extracting captions for video ID: {video_id}")
        
        # Fetch the YouTube page
        url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"Fetching URL: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch YouTube page. Status: {response.status}")
                    return None
                
                html = await response.text()
                logger.info("Successfully fetched YouTube page")
                
                # Log the first 500 characters of HTML to check content
                logger.info(f"HTML preview: {html[:500]}")
                
                # Find the ytInitialPlayerResponse
                match = re.search(r'var\s+ytInitialPlayerResponse\s*=\s*({.+?});', html)
                if not match:
                    logger.error("Could not find ytInitialPlayerResponse in HTML")
                    return None
                
                try:
                    player_response = json.loads(match.group(1))
                    logger.info("Successfully parsed player response")
                    
                    # Log the structure of player_response
                    logger.info(f"Player response keys: {player_response.keys()}")
                    
                    # Check for captions
                    if 'captions' not in player_response:
                        logger.error("No captions key in player response")
                        return None
                        
                    captions = player_response['captions']
                    logger.info(f"Captions data: {captions}")
                    
                    if 'playerCaptionsTracklistRenderer' in captions:
                        tracks = captions['playerCaptionsTracklistRenderer'].get('captionTracks', [])
                        logger.info(f"Found {len(tracks)} caption tracks")
                        
                        if tracks:
                            # Find Dutch caption track
                            dutch_track = None
                            for track in tracks:
                                lang = track.get('languageCode', '')
                                logger.info(f"Found track with language: {lang}")
                                if lang == 'nl' or lang.startswith('nl-'):
                                    dutch_track = track
                                    logger.info(f"Selected Dutch track: {json.dumps(dutch_track)[:200]}...")
                                    break
                            
                            if not dutch_track:
                                # If no Dutch track, take the first one
                                dutch_track = tracks[0]
                                logger.info(f"No Dutch track found, using first track: {json.dumps(dutch_track)[:200]}...")
                            
                            # Get the caption content URL
                            base_url = dutch_track.get('baseUrl')
                            if base_url:
                                logger.info(f"Found caption URL: {base_url}")
                                
                                # Fetch the actual captions
                                async with httpx.AsyncClient() as client:
                                    caption_response = await client.get(base_url)
                                    logger.info(f"Caption response status: {caption_response.status_code}")
                                    
                                    if caption_response.status_code == 200:
                                        caption_content = caption_response.text
                                        logger.info(f"Caption content preview: {caption_content[:200]}...")
                                        
                                        # Parse the XML content
                                        try:
                                            root = ET.fromstring(caption_content)
                                            events = []
                                            
                                            for text in root.findall('.//text'):
                                                start = float(text.get('start', 0))
                                                dur = float(text.get('dur', 0))
                                                content = text.text or ""
                                                
                                                if content:
                                                    content = (content.replace('&#39;', "'")
                                                                    .replace('&quot;', '"')
                                                                    .replace('\n', ' ')
                                                                    .replace('&amp;', '&')
                                                                    .strip())
                                                    
                                                    events.append({
                                                        "tStartMs": int(start * 1000),
                                                        "dDurationMs": int(dur * 1000),
                                                        "segs": [{"utf8": content}]
                                                    })
                                            
                                            if events:
                                                logger.info(f"Successfully parsed {len(events)} caption events")
                                                return {"events": events}
                                            else:
                                                logger.warning("No caption events found in XML")
                                        except ET.ParseError as e:
                                            logger.error(f"Failed to parse caption XML: {str(e)}")
                                    else:
                                        logger.error(f"Failed to fetch captions from URL: {base_url}")
                            else:
                                logger.error("No baseUrl found in caption track")
                        else:
                            logger.error("No caption tracks found")
                    else:
                        logger.error("No captions data found in player response")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse player response: {str(e)}")
                    return None
                    
    except Exception as e:
        logger.error(f"Error extracting captions: {str(e)}")
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_video(video_data: VideoURL):
    try:
        logger.info(f"Processing video URL: {video_data.video_url}")
        
        # Extract video ID
        video_id = extract_video_id(video_data.video_url)
        if not video_id:
            logger.error("Invalid YouTube URL")
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
            
        logger.info(f"Extracted video ID: {video_id}")
        
        # Get captions
        captions = await extract_captions_from_html(video_id)
        if not captions:
            logger.error("No captions found")
            raise HTTPException(status_code=404, detail="No captions found for this video")
            
        logger.info("Successfully extracted captions")
        
        return captions
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
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

@app.get("/simple-enhance")
async def simple_enhance(transcript: str = ""):
    """Enhance transcript text and return the enhanced version."""
    try:
        # Decode the transcript if provided
        import urllib.parse
        transcript_text = urllib.parse.unquote(transcript) if transcript else ""
        
        # Split into lines and enhance each line
        lines = transcript_text.split('\n')
        enhanced_lines = []
        
        for line in lines:
            if line.strip():
                # Capitalize first letter
                enhanced_line = line[0].upper() + line[1:] if line else ""
                # Add period if missing
                if not enhanced_line.endswith(('.', '!', '?')):
                    enhanced_line += '.'
                enhanced_lines.append(enhanced_line)
        
        # Join the enhanced lines
        enhanced_text = '\n'.join(enhanced_lines)
        
        # Return plain text response
        return Response(content=enhanced_text, media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error in simple_enhance: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

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