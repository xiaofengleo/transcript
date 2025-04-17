from fastapi import FastAPI, HTTPException, Request, Form
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
from typing import Optional, Union, Dict, Any
from pydantic import BaseModel
from bs4 import BeautifulSoup
import html
from urllib.parse import parse_qs, urlparse
import subprocess
import sys

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

# Initialize FastAPI app
app = FastAPI(title="YouTube Transcript Processor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class VideoURL(BaseModel):
    video_url: str

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

async def extract_captions_from_html(video_id: str) -> Optional[Dict[str, Any]]:
    """Extract captions directly from YouTube page HTML."""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        logger.info(f"Fetching YouTube page: {url}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch YouTube page, status code: {response.status_code}")
                return None
                
            html_content = response.text
            logger.info(f"Successfully fetched YouTube page ({len(html_content)} bytes)")
            
            # Look for caption data in the HTML
            # Pattern to find ytInitialPlayerResponse
            pattern = r'ytInitialPlayerResponse\s*=\s*({.+?});'
            match = re.search(pattern, html_content)
            
            if not match:
                logger.error("Could not find ytInitialPlayerResponse in the HTML")
                return None
                
            player_response_str = match.group(1)
            
            try:
                player_response = json.loads(player_response_str)
                logger.info("Successfully parsed player response JSON")
                
                # Extract captions data
                if 'captions' in player_response:
                    captions_data = player_response['captions']
                    logger.info(f"Found captions data: {json.dumps(captions_data)[:200]}...")
                    
                    if 'playerCaptionsTracklistRenderer' in captions_data:
                        tracks = captions_data['playerCaptionsTracklistRenderer'].get('captionTracks', [])
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
                        logger.error("No playerCaptionsTracklistRenderer found")
                else:
                    logger.error("No captions data found in player response")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse player response JSON: {str(e)}")
                
        return None
    except Exception as e:
        logger.error(f"Error extracting captions from HTML: {str(e)}")
        logger.error(traceback.format_exc())
        return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_video(video_url: str = Form(...)):
    """Process video URL and return captions."""
    try:
        logger.info(f"Processing video URL: {video_url}")
        
        video_id = extract_video_id(video_url)
        logger.info(f"Extracted video ID: {video_id}")
        
        # Extract captions directly from the YouTube page
        captions = await extract_captions_from_html(video_id)
        
        if captions and captions.get("events"):
            logger.info(f"Successfully extracted captions with {len(captions['events'])} events")
            return captions
        else:
            logger.error("Failed to extract captions")
            return JSONResponse(
                status_code=404,
                content={"detail": "No captions found for this video"}
            )
        
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={"detail": str(e)}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 