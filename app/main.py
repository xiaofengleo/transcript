from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
import openai
import os
from dotenv import load_dotenv
import re
import logging
from typing import Optional
import traceback
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Configure OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OpenAI API key not found in environment variables!")
else:
    openai.api_key = api_key
    logger.info("OpenAI API key loaded")

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    logger.info(f"Extracting video ID from URL: {url}")
    
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            logger.info(f"Successfully extracted video ID: {video_id}")
            return video_id
            
    raise ValueError(f"Could not extract video ID from URL: {url}")

def get_transcript(video_id: str) -> str:
    """Get transcript for a YouTube video."""
    try:
        logger.info(f"Fetching transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format transcript with timestamps
        formatted_transcript = []
        for entry in transcript_list:
            timestamp = int(float(entry['start']))
            minutes = timestamp // 60
            seconds = timestamp % 60
            time_str = f"{minutes:02d}:{seconds:02d}"
            formatted_transcript.append(f"[{time_str}] {entry['text']}")
        
        return "\n".join(formatted_transcript)
        
    except TranscriptsDisabled:
        logger.error(f"Transcripts are disabled for video {video_id}")
        raise HTTPException(status_code=400, detail="Transcripts are disabled for this video")
    except NoTranscriptFound:
        logger.error(f"No transcript found for video {video_id}")
        raise HTTPException(status_code=400, detail="No transcript found for this video")
    except Exception as e:
        logger.error(f"Error getting transcript: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get transcript: {str(e)}")

async def improve_transcript(transcript: str) -> str:
    """Improve transcript using ChatGPT."""
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional editor. Your task is to improve the syntax and punctuation of transcripts while maintaining their original meaning."},
                {"role": "user", "content": f"Please improve the syntax and punctuation of this transcript:\n\n{transcript}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_video(video_url: str = Form(...)):
    try:
        logger.info(f"Processing video URL: {video_url}")
        
        # Extract video ID
        video_id = extract_video_id(video_url)
        
        # Get transcript
        transcript = get_transcript(video_id)
        
        return {
            "original": transcript,
            "improved": transcript  # For now, returning same text
        }
        
    except ValueError as ve:
        logger.error(f"Invalid URL format: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/improve")
async def improve_transcript_endpoint(video_url: str = Form(...)):
    logger.info(f"Received request to improve transcript for video: {video_url}")
    
    try:
        # Check OpenAI API key first
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="OpenAI API key not configured. Check server configuration."
            )

        # Extract video ID
        try:
            video_id = video_url.split('v=')[1].split('&')[0]
            logger.info(f"Extracted video ID: {video_id}")
        except Exception as e:
            logger.error(f"Failed to extract video ID: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid YouTube URL format. Please use a URL like 'https://www.youtube.com/watch?v=VIDEO_ID'"
            )

        # Get transcript
        try:
            logger.info(f"Attempting to get transcript for video {video_id}")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            formatter = TextFormatter()
            transcript = formatter.format_transcript(transcript_list)
            logger.info("Successfully retrieved transcript")
            
            if not transcript:
                raise ValueError("Empty transcript received")
                
        except TranscriptsDisabled:
            raise HTTPException(
                status_code=400,
                detail="Transcripts are disabled for this video"
            )
        except NoTranscriptFound:
            raise HTTPException(
                status_code=400,
                detail="No transcript found for this video"
            )
        except Exception as e:
            logger.error(f"Transcript error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=400,
                detail=f"Failed to get transcript: {str(e)}"
            )

        # Improve transcript with OpenAI
        try:
            logger.info("Attempting to improve transcript with OpenAI")
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional editor. Improve the syntax and punctuation while maintaining the meaning."},
                    {"role": "user", "content": f"Improve this transcript:\n\n{transcript}"}
                ]
            )
            improved_transcript = response.choices[0].message.content
            logger.info("Successfully improved transcript")
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to improve transcript: {str(e)}"
            )

        return JSONResponse(
            content={
                "original": transcript,
                "improved": improved_transcript
            },
            status_code=200
        )

    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        return JSONResponse(
            content={"detail": he.detail},
            status_code=he.status_code
        )
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"detail": error_msg},
            status_code=500
        ) 