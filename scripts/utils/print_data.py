import os
from pymongo import MongoClient
from bson import ObjectId

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client.videoEnhancedLearning

# Path where the files will be saved
output_path = os.path.dirname(os.path.abspath(__file__))

# Helper function to convert MongoDB documents to a readable format
def format_record(record):
    # Convert ObjectId to a string for plain text compatibility
    for key, value in record.items():
        if isinstance(value, ObjectId):
            record[key] = str(value)
    return record

# Function to save top 15 records from a collection into a plain text file
def save_top_records(collection, filename):
    records = list(collection.find().limit(15))
    with open(os.path.join(output_path, filename), 'w', encoding='utf-8') as f:
        for record in records:
            formatted_record = format_record(record)
            f.write(str(formatted_record) + '\n\n')
    print(f"Top 15 records from '{collection.name}' saved to '{filename}'.")

# Retrieve and save data from each collection
save_top_records(db.videos, 'top_15_videos.txt')
save_top_records(db.transcripts, 'top_15_transcripts.txt')
save_top_records(db.segments, 'top_15_segments.txt')

print("Data saved successfully for analysis.")
