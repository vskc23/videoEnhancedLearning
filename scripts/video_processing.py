import os
import configparser
import logging
import time
from datetime import datetime
from pymongo import MongoClient, ASCENDING
import whisper
import openai
from moviepy.editor import VideoFileClip
import uuid
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pytz
import json

# Ensure necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy English model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # If the model is not found, download it
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load('en_core_web_sm')

# Load configuration
def load_config():
    """Loads the configuration from a hardcoded file path."""
    # Hardcoded path to your config.ini file
    config_path = "C:\path\to\your\config.ini"
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Check if the config file was loaded properly
    if not config.sections():
        raise FileNotFoundError(f"Config file not found or is empty at path: {config_path}")
    
    return config

# Generate a unique ID with a given prefix
def generate_id(prefix):
    """Generates a unique ID with the given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

# Extract audio from video
def extract_audio_from_video(video_path, output_audio_path):
    """
    Extracts audio from the video and saves it as a .wav file.
    """
    print(f"Extracting audio from {video_path}...")
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_audio_path, codec='pcm_s16le')
        logging.info(f"Audio extracted to: {output_audio_path}")
        print(f"Audio extracted to: {output_audio_path}")
        return output_audio_path
    except Exception as e:
        logging.error(f"Error extracting audio from {video_path}: {e}")
        print(f"Error extracting audio from {video_path}: {e}")
        return None

# Transcribe audio using Whisper
def transcribe_audio(audio_path, model):
    """
    Transcribes the audio using Whisper with the specified model.
    """
    print(f"Transcribing audio: {audio_path}...")
    try:
        result = model.transcribe(audio_path)
        logging.info(f"Transcription complete for: {audio_path}")
        print(f"Transcription complete for: {audio_path}")
        return result["text"], result["segments"]
    except Exception as e:
        logging.error(f"Error transcribing audio {audio_path}: {e}")
        print(f"Error transcribing audio {audio_path}: {e}")
        return None, None

# Refine transcript using OpenAI's GPT-4
def refine_transcript(transcript):
    """
    Refines the transcript using OpenAI's GPT-4 model with retry logic.
    """
    print("Refining transcript...")
    max_retries = 2
    retry_delay = 60  # in seconds
    attempts = 0

    while attempts < max_retries:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "Refine the following transcript for grammar, spelling, and clarity"
                    },
                    {
                        "role": "user",
                        "content": transcript
                    }
                ]
            )
            refined_transcript =  response.choices[0].message.content
            logging.info("Transcript refined successfully.")
            print("Transcript refined successfully.")
            return refined_transcript
        except openai.OpenAIError:
            logging.warning(f"Rate limit exceeded, retrying in {retry_delay} seconds...")
            print(f"Rate limit exceeded, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
            attempts += 1
        except Exception as e:
            logging.error(f"Error refining transcript: {e}")
            print(f"Error refining transcript: {e}")
            return None
    return None

# Function to generate embedding using OpenAI
def generate_embedding(text, model="text-embedding-ada-002"):
    """
    Generates an embedding for the given text using OpenAI's API.

    Parameters:
    - text (str): The text to generate an embedding for.
    - model (str): The OpenAI model to use for embedding.

    Returns:
    - list: The embedding vector if successful, None otherwise.
    """
    try:
        response = openai.embeddings.create(
            input=[text],  # OpenAI expects a list of texts
            model=model
        )
        # Access the embedding correctly
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding for text: {e}")
        print(f"Error generating embedding for text: {e}")
        return None

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    if len(vec1) != len(vec2):
        logging.error(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Segment transcript into topics using LDA and generate embeddings

# Segment transcript into topics using LDA and generate embeddings
def segment_transcript(transcript_segments, transcript_data):
    """
    Segments the transcript into topics using NLP techniques.
    Utilizes transcript segment timestamps for accurate timing.
    Generates embeddings for each segment.

    Parameters:
    - transcript_segments (list): List of transcript segments with timestamps.
    - transcript_data (dict): Contains transcript_id and video_id.

    Returns:
    - list: List of segment dictionaries with embeddings.
    """
    print("Segmenting transcript...")
    segments = []

    # Debugging: Log the number of segments
    logging.debug(f"Number of transcript segments: {len(transcript_segments)}")
    print(f"Number of transcript segments: {len(transcript_segments)}")

    # Extract texts from segments for topic modeling
    texts = [seg['text'] for seg in transcript_segments if 'text' in seg]
    
    if not texts:
        logging.warning("No texts available for segmentation.")
        print("No texts available for segmentation.")
        return segments

    # Vectorize the texts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(texts)

    # Apply LDA for topic modeling
    num_topics = 5  # Adjust based on your data
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf)
    topic_assignments = lda.transform(tfidf)
    topics = np.argmax(topic_assignments, axis=1)

    for idx, segment_data in enumerate(transcript_segments):
        if 'text' not in segment_data:
            continue

        text = segment_data['text'].strip()
        if not text:
            continue

        segment_id = generate_id("segment")
        start_time = segment_data.get('start', None)
        end_time = segment_data.get('end', None)
        topic_num = topics[idx] if idx < len(topics) else "Unknown"
        topic = f"Topic {topic_num}" if topic_num != "Unknown" else "Unknown"
        keywords = extract_keywords(text)

        # Generate embedding for the segment
        embedding = generate_embedding(text)
        if embedding is None:
            logging.warning(f"Failed to generate embedding for segment_id: {segment_id}. Skipping.")
            continue

        segment = {
            "segment_id": segment_id,
            "transcript_id": transcript_data['transcript_id'],
            "video_id": transcript_data['video_id'],
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "topic": topic,
            "keywords": keywords,
            "embedding": embedding
        }
        segments.append(segment)
    
    print("Segmentation and embedding generation complete.")
    logging.debug(f"Number of segments generated: {len(segments)}")
    return segments




# Extract keywords using spaCy
def extract_keywords(text):
    """
    Extracts keywords from the given text using NLP techniques.
    """
    doc = nlp(text.lower())
    keywords = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop]
    return list(set(keywords))

# Store data in MongoDB
def store_data_in_db(video_metadata, transcript_data, segments_data, db):
    """
    Stores the processed data in MongoDB, including embeddings.
    """
    print("Storing data in MongoDB...")
    try:
        videos_collection = db.videos
        transcripts_collection = db.transcripts
        segments_collection = db.segments

        # Insert video metadata
        videos_collection.insert_one(video_metadata)

        # Insert transcript data
        transcripts_collection.insert_one(transcript_data)

        # Insert segments data
        if segments_data:
            segments_collection.insert_many(segments_data)

        logging.info(f"Data stored for video ID: {video_metadata['video_id']}")
        print(f"Data stored for video ID: {video_metadata['video_id']}")
    except Exception as e:
        logging.error(f"Error storing data in MongoDB: {e}")
        print(f"Error storing data in MongoDB: {e}")

# Helper function to convert seconds to 'HH:MM:SS' format using timezone-aware datetime
def seconds_to_hms(seconds):
    """
    Converts seconds to 'HH:MM:SS' format using timezone-aware datetime.
    """
    try:
        int_seconds = int(seconds)
        # Use timezone-aware datetime object in UTC
        utc_dt = datetime.fromtimestamp(int_seconds, pytz.UTC)
        return utc_dt.strftime('%H:%M:%S')
    except Exception as e:
        logging.error(f"Error converting seconds to HH:MM:SS: {e}")
        return "00:00:00"

# Main video processing function
def process_video(video_file, model, db, config):
    """
    Processes a single video file: extraction, transcription, refinement, segmentation, storage.
    """
    try:
        logging.info(f"Processing video: {video_file}")
        print(f"Processing video: {video_file}")
        video_path = os.path.join(config['LOCATIONS']['input_folder'], video_file)

        # Generate paths for outputs
        audio_output_path = os.path.join(config['LOCATIONS']['audio_folder'], f"{os.path.splitext(video_file)[0]}.wav")
        transcript_output_path = os.path.join(config['LOCATIONS']['transcripts_folder'], f"{os.path.splitext(video_file)[0]}_transcript.txt")
        refined_transcript_output_path = os.path.join(config['LOCATIONS']['refined_transcripts_folder'], f"{os.path.splitext(video_file)[0]}_refined_transcript.txt")
        transcript_segments_path = os.path.join(
            config['LOCATIONS']['transcripts_folder'],
            f"{os.path.splitext(video_file)[0]}_transcript_segments.json"
        )

        # Step 1: Extract audio if it doesn't exist
        if not os.path.exists(audio_output_path):
            extract_audio_from_video(video_path, audio_output_path)
        else:
            logging.info(f"Audio already extracted: {audio_output_path}")
            print(f"Audio already extracted: {audio_output_path}")

        # Step 2: Transcribe audio if transcript doesn't exist
        if os.path.exists(transcript_output_path):
            with open(transcript_output_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            logging.info(f"Transcript already exists for: {video_file}")
            print(f"Transcript already exists for: {video_file}")

            # Load transcript segments from the JSON file if it exists
            if os.path.exists(transcript_segments_path):
                with open(transcript_segments_path, 'r', encoding='utf-8') as f:
                    transcript_segments = json.load(f)
                logging.info(f"Transcript segments loaded for: {video_file}")
                print(f"Transcript segments loaded for: {video_file}")
            else:
                logging.warning(f"Transcript segments file not found for: {video_file}. Proceeding with empty segments.")
                print(f"Transcript segments file not found for: {video_file}. Proceeding with empty segments.")
                transcript_segments = []
        else:
            transcript_text, transcript_segments = transcribe_audio(audio_output_path, model)
            if not transcript_text:
                logging.error(f"Transcription failed for {video_file}")
                print(f"Transcription failed for {video_file}")
                return
            # Save the original transcript
            with open(transcript_output_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            print(f"Original transcript saved to: {transcript_output_path}")

            # Save the transcript segments to a JSON file
            with open(transcript_segments_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_segments, f, ensure_ascii=False, indent=4)
            print(f"Transcript segments saved to: {transcript_segments_path}")

        # Step 3: Refine transcript if refined transcript doesn't exist
        if os.path.exists(refined_transcript_output_path):
            with open(refined_transcript_output_path, 'r', encoding='utf-8') as f:
                refined_transcript_text = f.read()
            logging.info(f"Refined transcript already exists for: {video_file}")
            print(f"Refined transcript already exists for: {video_file}")
        else:
            refined_transcript_text = refine_transcript(transcript_text)
            if not refined_transcript_text:
                logging.error(f"Refinement failed for {video_file}")
                print(f"Refinement failed for {video_file}")
                refined_transcript_text = transcript_text  # Use original if refinement fails
            # Save the refined transcript
            with open(refined_transcript_output_path, 'w', encoding='utf-8') as f:
                f.write(refined_transcript_text)
            print(f"Refined transcript saved to: {refined_transcript_output_path}")

        # **Step 4: Generate unique IDs for video and transcript**
        video_id = generate_id("video")
        transcript_id = generate_id("transcript")

        # Step 5: Segment transcript and generate embeddings
        segments = segment_transcript(
            transcript_segments,
            {
                "transcript_id": transcript_id,
                "video_id": video_id
            }
        )
        if not segments:
            logging.warning(f"No segments found for {video_file}")
            print(f"No segments found for {video_file}")

        # Step 6: Store data in MongoDB
        video_metadata = {
            "video_id": video_id,
            "file_path": video_path,
            "title": os.path.splitext(video_file)[0],
            "upload_date": datetime.now(),
            "duration": VideoFileClip(video_path).duration,
            "tags": [],  # Optional: Implement tag detection
            "description": "Automatically processed video",
            "processed": True
        }

        # Construct transcript data with timestamps
        transcript_data = {
            "transcript_id": transcript_id,
            "video_id": video_id,
            "text": transcript_text,
            "refined_text": refined_transcript_text,
            "language": "en",  # Assuming English; implement detection if needed
            "timestamps": [
                {
                    "timestamp": seg.get('start', None),
                    "text_segment": seg.get('text', "")
                } for seg in transcript_segments
            ] if transcript_segments else []
        }

        # Update segments with transcript_id and video_id
        for segment in segments:
            segment['transcript_id'] = transcript_id
            segment['video_id'] = video_id

        store_data_in_db(video_metadata, transcript_data, segments, db)

        logging.info(f"Processing completed for: {video_file}")
        print(f"Processing completed for: {video_file}")

    except Exception as e:
        logging.error(f"Error occurred while processing video {video_file}: {e}")
        print(f"Error occurred while processing video {video_file}: {e}")

# Main function
def main():
    print("Loading configuration...")
    # Load configuration
    config = load_config()

    # Ensure output directories exist
    os.makedirs(config['LOCATIONS']['logs_folder'], exist_ok=True)
    os.makedirs(config['LOCATIONS']['audio_folder'], exist_ok=True)
    os.makedirs(config['LOCATIONS']['transcripts_folder'], exist_ok=True)
    os.makedirs(config['LOCATIONS']['refined_transcripts_folder'], exist_ok=True)

    # Setup logging to both file and console with UTF-8 encoding for the file handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config['LOCATIONS']['logs_folder'], 'processing.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    # MongoDB setup
    try:
        client = MongoClient(host=config['DATABASE']['host'], port=int(config['DATABASE']['port']))
        db = client[config['DATABASE']['db_name']]
        logging.info(f"Connected to MongoDB at {config['DATABASE']['host']} on port {config['DATABASE']['port']}")
        print(f"Connected to MongoDB at {config['DATABASE']['host']} on port {config['DATABASE']['port']}")
    except Exception as e:
        logging.critical(f"Failed to connect to MongoDB: {e}")
        print(f"Failed to connect to MongoDB: {e}")
        return

    # OpenAI setup
    openai.api_key = config['OPENAI']['api_key']

    # Whisper setup - load model only once globally
    transcription_model = config['PROCESSING']['transcription_model'].strip()
    try:
        print(f"Loading Whisper model '{transcription_model}'...")
        model = whisper.load_model(transcription_model)  # Use 'tiny' model as per your request
        logging.info(f"Whisper model '{transcription_model}' loaded successfully.")
        print(f"Whisper model '{transcription_model}' loaded successfully.")
    except Exception as e:
        logging.critical(f"Failed to load Whisper model '{transcription_model}': {e}")
        print(f"Failed to load Whisper model '{transcription_model}': {e}")
        return

    # Get list of video files from the input folder
    try:
        # Read the video files specified in the config
        video_files_config = config['FILES']['video_files']
        video_files = [f.strip() for f in video_files_config.split(',') if f.strip()]
        if not video_files:
            logging.warning("No video files specified in the config file.")
            print("No video files specified in the config file.")
            return
        else:
            print(f"Videos to process: {video_files}")
    except Exception as e:
        logging.error(f"Error reading video files from config: {e}")
        print(f"Error reading video files from config: {e}")
        return

    # Process each video file
    for video_file in video_files:
        process_video(video_file, model, db, config)

if __name__ == "__main__":
    main()
