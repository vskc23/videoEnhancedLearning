from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client.videoEnhancedLearning
segments_collection = db.segments

# Fetch a sample of segments
sample_segments = segments_collection.find({}, {"segment_id": 1, "embedding": 1}).limit(5)

for segment in sample_segments:
    embedding = segment.get("embedding", [])
    if isinstance(embedding, list):
        print(f"Segment ID: {segment['segment_id']}, Embedding Length: {len(embedding)}")
    else:
        print(f"Segment ID: {segment['segment_id']}, Embedding: Not a list")
