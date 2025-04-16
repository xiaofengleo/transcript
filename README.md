# YouTube Transcript Processor

This application downloads YouTube video transcripts and uses ChatGPT to improve their syntax and punctuation.

## Features
- Download transcripts from YouTube videos
- Process transcripts using ChatGPT for improved readability
- Clean and modern web interface

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running Locally
```bash
uvicorn app.main:app --reload
```

## Deployment on Render

1. Create a new account on [Render](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Configure the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add your environment variables:
   - OPENAI_API_KEY
6. Deploy!

## Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key for ChatGPT access

## Tech Stack
- FastAPI (Backend)
- YouTube Transcript API
- OpenAI API (ChatGPT)
- HTML/CSS/JavaScript (Frontend) 