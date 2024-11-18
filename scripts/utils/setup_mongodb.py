from pymongo import MongoClient, ASCENDING
from datetime import datetime
import uuid

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Access the database
db = client.videoEnhancedLearning

# Helper function to generate unique IDs
def generate_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

# 1. Create the `videos` collection and apply the schema with indexes
if "videos" not in db.list_collection_names():
    videos_collection = db.create_collection("videos")
    videos_collection.create_index([("video_id", ASCENDING)], unique=True)
    videos_collection.create_index([("tags", ASCENDING)])
else:
    videos_collection = db.videos

# 2. Create the `transcripts` collection and apply the schema with indexes
if "transcripts" not in db.list_collection_names():
    transcripts_collection = db.create_collection("transcripts")
    transcripts_collection.create_index([("transcript_id", ASCENDING)], unique=True)
    transcripts_collection.create_index([("video_id", ASCENDING)])
else:
    transcripts_collection = db.transcripts

# 3. Create the `segments` collection and apply the schema with indexes
if "segments" not in db.list_collection_names():
    segments_collection = db.create_collection("segments")
    segments_collection.create_index([("segment_id", ASCENDING)], unique=True)
    segments_collection.create_index([("transcript_id", ASCENDING)])
    segments_collection.create_index([("video_id", ASCENDING)])
    segments_collection.create_index([("keywords", ASCENDING)])
else:
    segments_collection = db.segments

# 4. Create the `studentQueries` collection and apply the schema with indexes
if "studentQueries" not in db.list_collection_names():
    queries_collection = db.create_collection("studentQueries")
    queries_collection.create_index([("query_id", ASCENDING)], unique=True)
    queries_collection.create_index([("embedding", ASCENDING)])
else:
    queries_collection = db.studentQueries

# 5. Create the `queryMatches` collection and apply the schema with indexes
if "queryMatches" not in db.list_collection_names():
    matches_collection = db.create_collection("queryMatches")
    matches_collection.create_index([("match_id", ASCENDING)], unique=True)
    matches_collection.create_index([("query_id", ASCENDING)])
    matches_collection.create_index([("segment_id", ASCENDING)])
else:
    matches_collection = db.queryMatches

print("All collections and indexes created successfully.")

# Optional: Sample Data Insertion to Demonstrate Schema

# 1. Insert Sample Video Document
sample_video = {
    "video_id": generate_id("video"),
    "file_path": "./data/videos/lecture1.mp4",
    "title": "Lecture 1 - Introduction to Algorithms",
    "upload_date": datetime.now(),
    "duration": 3600,  # Duration in seconds (1 hour)
    "tags": ["algorithms", "computer science", "lecture"],
    "description": "This is a sample lecture on algorithms.",
    "processed": False  # Indicates if processing (steps 2-6) is complete
}

videos_collection.insert_one(sample_video)
print(f"Inserted sample video with ID: {sample_video['video_id']}")

# 2. Insert Sample Transcript Document
sample_transcript = {
    "transcript_id": generate_id("transcript"),
    "video_id": sample_video["video_id"],
    "text": "This is the full transcript of the video...",
    "refined_text": "This is the refined transcript...",
    "language": "en",
    "timestamps": [
        {"timestamp": 0, "text_segment": "Introduction to Algorithms..."},
        {"timestamp": 300, "text_segment": "Sorting Algorithms..."}
    ]
}

transcripts_collection.insert_one(sample_transcript)
print(f"Inserted sample transcript with ID: {sample_transcript['transcript_id']}")

# 3. Insert Sample Segment Documents
sample_segments = [
    {
        "segment_id": generate_id("segment"),
        "transcript_id": sample_transcript["transcript_id"],
        "video_id": sample_video["video_id"],
        "start_time": 0,
        "end_time": 300,
        "text": "Introduction to Algorithms...",
        "topic": "Introduction",
        "keywords": ["introduction", "algorithms"],
        "embedding": [0.1, 0.2, 0.3]  # Example embedding vector
    },
    {
        "segment_id": generate_id("segment"),
        "transcript_id": sample_transcript["transcript_id"],
        "video_id": sample_video["video_id"],
        "start_time": 300,
        "end_time": 600,
        "text": "Sorting Algorithms...",
        "topic": "Sorting",
        "keywords": ["sorting", "algorithms", "quick sort"],
        "embedding": [0.4, 0.5, 0.6]
    }
]

segments_collection.insert_many(sample_segments)
print(f"Inserted {len(sample_segments)} sample segments.")

# 4. Insert Sample Student Query Document
sample_query = {
    "query_id": generate_id("query"),
    "text": "How does quick sort algorithm work?",
    "source": "Online Forum",
    "date_collected": datetime.now(),
    "embedding": [0.35, 0.45, 0.55]  # Example embedding vector
}

queries_collection.insert_one(sample_query)
print(f"Inserted sample student query with ID: {sample_query['query_id']}")

# 5. Insert Sample Query Match Document
sample_match = {
    "match_id": generate_id("match"),
    "query_id": sample_query["query_id"],
    "segment_id": sample_segments[1]["segment_id"],  # Assuming it matches the second segment
    "similarity_score": 0.95,
    "matched_on": datetime.now()
}

matches_collection.insert_one(sample_match)
print(f"Inserted sample query match with ID: {sample_match['match_id']}")
