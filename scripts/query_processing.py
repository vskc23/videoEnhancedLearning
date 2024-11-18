import os
import configparser
import logging
from datetime import datetime
from pymongo import MongoClient, ASCENDING
import openai
import numpy as np
import uuid
from crop_video import crop_video  # Importing from the separate crop_video.py script

# Constants for minimum and maximum duration in seconds
MIN_DURATION = 180  # 3 minutes
MAX_DURATION = 720  # 12 minutes

# Helper function to convert seconds to 'HH:MM:SS' format
def seconds_to_hms(seconds):
    """
    Converts seconds to 'HH:MM:SS' format.
    """
    try:
        int_seconds = int(seconds)
        # Use UTC to avoid timezone issues
        return str(datetime.utcfromtimestamp(int_seconds).time())
    except Exception as e:
        logging.error(f"Error converting seconds to HH:MM:SS: {e}")
        return "00:00:00"

# Function to load configuration
def load_config(config_path):
    """
    Loads the configuration from the specified file.
    """
    print("Loading configuration...")
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
        if not config.sections():
            raise FileNotFoundError(f"Config file not found or is empty at path: {config_path}")
        print("Configuration loaded successfully.")
        return config
    except (configparser.Error, FileNotFoundError) as e:
        logging.critical(f"Error loading configuration: {e}")
        print(f"Error loading configuration: {e}")
        exit(1)

# Function to generate a unique ID with a given prefix
def generate_id(prefix):
    """
    Generates a unique ID with the given prefix.
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

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

# Function to generate embedding for the query using OpenAI's API
def generate_query_embedding(query):
    """
    Generates an embedding for the query using OpenAI's API.

    Parameters:
    - query (str): The student's query.

    Returns:
    - list: Embedding vector.
    """
    try:
        response = openai.embeddings.create(
            input=[query],  # Ensure input is a list
            model="text-embedding-ada-002"
        )
        # Access the embedding correctly
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        logging.error(f"Error generating embedding for query: {e}")
        print(f"Error generating embedding for query: {e}")
        return None

# Main function to process the query
def process_query():
    """
    Handles the workflow from receiving the student's query to extracting and saving the relevant video clips.
    """
    # Load configuration
    config = "C:\path\to\your\config.ini"

    # Setup logging
    os.makedirs(config['LOCATIONS']['logs_folder'], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config['LOCATIONS']['logs_folder'], 'query_processing.log')),
            logging.StreamHandler()
        ]
    )

    # MongoDB setup
    try:
        client_mongo = MongoClient(host=config['DATABASE']['host'], port=int(config['DATABASE']['port']))
        db = client_mongo[config['DATABASE']['db_name']]
        logging.info(f"Connected to MongoDB at {config['DATABASE']['host']} on port {config['DATABASE']['port']}")
        print(f"Connected to MongoDB at {config['DATABASE']['host']} on port {config['DATABASE']['port']}")
    except Exception as e:
        logging.critical(f"Failed to connect to MongoDB: {e}")
        print(f"Failed to connect to MongoDB: {e}")
        return

    # OpenAI API key setup
    openai_api_key = config['OPENAI'].get('api_key', None)
    if not openai_api_key:
        logging.critical("OpenAI API key not found in configuration.")
        print("OpenAI API key not found in configuration.")
        return

    # Initialize OpenAI API key
    openai.api_key = openai_api_key

    # Prompt the user for their query
    question = input("Hi, please enter your query: ").strip()
    if not question:
        print("Empty query entered. Exiting.")
        return

    logging.info(f"Received query: {question}")

    # Generate embedding for the query
    query_embedding = generate_query_embedding(question)
    if query_embedding is None:
        print("Failed to generate embedding for the query.")
        return

    # Store the student query in the database
    query_id = generate_id("query")
    try:
        db.studentQueries.insert_one({
            "query_id": query_id,
            "text": question,
            "source": "Manual Input",  # Adjust as needed (e.g., "CampusWire", "Piazza")
            "date_collected": datetime.now(),
            "embedding": query_embedding
        })
        logging.info(f"Stored student query with ID: {query_id}")
    except Exception as e:
        logging.error(f"Error storing student query: {e}")
        print(f"Error storing student query: {e}")
        return

    # Fetch segments from the database
    segments_collection = db.segments
    try:
        segments = list(segments_collection.find({}))
        if not segments:
            print("No segments found in the database.")
            logging.warning("No segments found in the database.")
            return
    except Exception as e:
        logging.error(f"Error fetching segments from database: {e}")
        print(f"Error fetching segments from database: {e}")
        return

    # Compute similarities
    similarities = []
    for segment in segments:
        segment_embedding = segment.get('embedding', None)
        if segment_embedding is None:
            continue  # Skip if no embedding

        if len(segment_embedding) != len(query_embedding):
            segment_id = segment.get('segment_id', 'Unknown')
            logging.warning(f"Segment ID {segment_id} has an embedding of length {len(segment_embedding)}, expected {len(query_embedding)}. Skipping.")
            continue

        similarity = cosine_similarity(query_embedding, segment_embedding)
        similarities.append((similarity, segment))

    if not similarities:
        print("No segments with embeddings found.")
        logging.warning("No segments with embeddings found.")
        return

    # Sort segments by similarity in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Select top N segments
    top_n = 3  # Adjust as needed
    top_segments = similarities[:top_n]

    if not top_segments:
        print("No relevant segments found for the query.")
        logging.info("No relevant segments found for the query.")
        return

    # Output folder for video segments
    output_folder = config['LOCATIONS']['output_folder']
    os.makedirs(output_folder, exist_ok=True)

    # Process each top segment
    for idx, (similarity, segment) in enumerate(top_segments):
        video_id = segment.get('video_id', None)
        segment_id = segment.get('segment_id', None)

        if not video_id:
            logging.warning(f"Segment ID {segment_id} has no associated video ID.")
            continue

        # Fetch all segments from the same video, sorted by start_time
        try:
            video_segments = list(segments_collection.find({'video_id': video_id}).sort('start_time', ASCENDING))
            if not video_segments:
                logging.warning(f"No segments found for video_id: {video_id}")
                continue
        except Exception as e:
            logging.error(f"Error fetching segments for video_id {video_id}: {e}")
            continue

        # Find the index of the matching segment
        try:
            segment_indices = {seg['segment_id']: idx for idx, seg in enumerate(video_segments)}
            matching_idx = segment_indices.get(segment_id, None)
            if matching_idx is None:
                logging.warning(f"Segment ID {segment_id} not found in segments of video_id {video_id}")
                continue
        except Exception as e:
            logging.error(f"Error processing segments for video_id {video_id}: {e}")
            continue

        # Initialize start and end indices
        start_idx = matching_idx
        end_idx = matching_idx

        # Initialize start_time and end_time
        start_time = video_segments[start_idx].get('start_time', None)
        end_time = video_segments[end_idx].get('end_time', None)

        if start_time is None or end_time is None:
            logging.warning(f"Missing timestamps for segment_id: {segment_id}")
            continue

        # Expand to include adjacent segments until desired duration is reached
        total_duration = end_time - start_time

        while total_duration < MIN_DURATION:
            expanded = False
            # Try to expand backward
            if start_idx > 0:
                start_idx -= 1
                new_start_time = video_segments[start_idx].get('start_time', start_time)
                if new_start_time is not None:
                    start_time = new_start_time
                    expanded = True
            # Try to expand forward
            if total_duration < MIN_DURATION and end_idx < len(video_segments) - 1:
                end_idx += 1
                new_end_time = video_segments[end_idx].get('end_time', end_time)
                if new_end_time is not None:
                    end_time = new_end_time
                    expanded = True
            if not expanded:
                # Cannot expand further
                break
            # Update total duration
            total_duration = end_time - start_time
            # Break if total duration exceeds maximum duration
            if total_duration >= MAX_DURATION:
                break

        # Ensure total_duration is within desired range
        if total_duration < MIN_DURATION:
            logging.warning(f"Could not find enough content to reach minimum duration for segment_id: {segment_id}")
            continue
        if total_duration > MAX_DURATION:
            # Adjust end_time to limit to MAX_DURATION
            end_time = start_time + MAX_DURATION

        # Convert timestamps from seconds to 'HH:MM:SS' format
        start_time_formatted = seconds_to_hms(start_time)
        end_time_formatted = seconds_to_hms(end_time)

        # Retrieve video metadata
        video_metadata = db.videos.find_one({"video_id": video_id})
        if not video_metadata:
            logging.warning(f"No video metadata found for video_id: {video_id}")
            continue

        video_path = video_metadata.get('file_path', None)
        video_title = video_metadata.get('title', f"video_{video_id}")

        if not video_path:
            logging.warning(f"Video metadata for video_id {video_id} lacks 'file_path'.")
            continue

        # Generate output video path
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_video_path = os.path.join(
            output_folder,
            f"{video_title}_segment_{idx+1}_{timestamp_str}.mp4"
        )

        # Crop the video using the separate crop_video.py script
        try:
            crop_video(video_path, start_time_formatted, end_time_formatted, output_video_path)
            logging.info(f"Cropped video saved to: {output_video_path}")
            print(f"Cropped video saved to: {output_video_path}")
        except Exception as e:
            logging.error(f"Error cropping video for segment {segment_id}: {e}")
            print(f"Error cropping video for segment {segment_id}: {e}")
            continue

        # Store the query match in the database
        match_id = generate_id("match")
        try:
            db.queryMatches.insert_one({
                "match_id": match_id,
                "query_id": query_id,
                "segment_id": segment_id,
                "similarity_score": similarity,
                "matched_on": datetime.now()
            })
            logging.info(f"Stored query match with ID: {match_id}")
        except Exception as e:
            logging.error(f"Error storing query match: {e}")
            print(f"Error storing query match: {e}")

    print("Processing complete.")

if __name__ == "__main__":
    process_query()
