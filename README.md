
# **VIDIQ - Video Query Processing System**

## **Overview**

VIDIQ is a video processing system designed to:
- Process educational videos to extract, refine, and store transcripts and segments in a structured format.
- Match user queries with relevant video segments and produce cropped video clips tailored to the query.

This system leverages cutting-edge tools such as **OpenAI GPT**, **Whisper**, **MongoDB**, and **FFmpeg**.

---

## **Features**

- **Video Processing**:
  - Extract audio and generate transcripts.
  - Refine transcripts and break them into topic-based segments.
  - Generate embeddings for transcript segments and store metadata in MongoDB.
  
- **Query Processing**:
  - Match user queries with relevant video segments using embedding similarity.
  - Automatically crop videos to match the query context.

---

## **Setup Instructions**

### **1. Prerequisites**

Ensure the following are installed:
- **Python** (>= 3.8)
- **MongoDB**
- **FFmpeg**
- Whisper dependencies
- SpaCy with `en_core_web_sm`

### **2. Install Dependencies**

Run the following command to install all required Python packages:
```bash
pip install -r requirements.txt
```

### **3. Configuration**

1. **Create Configuration File**:
   - Rename `config_template.ini` to `config.ini` and update the following fields:
     - MongoDB connection details.
     - OpenAI API key.
     - Folder paths for input/output directories.

2. **Environment Variables**:
   - Set `OPENAI_API_KEY` in your environment for secure usage:
     ```bash
     export OPENAI_API_KEY=your_api_key
     ```

### **4. Run MongoDB Setup**

Set up the database and its collections by running:
```bash
python scripts/setup_mongodb.py
```

### **5. Process Videos**

To process videos in the `input/videos` directory, run:
```bash
python scripts/video_processing.py
```

### **6. Query Processing**

To match user queries with relevant video segments:
```bash
python scripts/query_processing.py
```

---

## **Project Structure**

- **input/**: Contains input videos to be processed.
- **output/**: Stores generated outputs:
  - `audio/`: Extracted audio files.
  - `logs/`: Log files for debugging.
  - `transcripts/`: Raw transcript files.
  - `refined_transcripts/`: Processed transcript files.
  - `video_segments/`: Cropped video files.

- **scripts/**: Core Python scripts for processing and querying.
  - `video_processing.py`: Processes videos to generate transcripts and segments.
  - `query_processing.py`: Matches user queries with video segments.
  - `setup_mongodb.py`: Initializes MongoDB collections.
  - `crop_video.py`: Handles video cropping.

- **utils/**: Helper functions and utility scripts.

- **config.ini**: Configuration file for the system.
- **requirements.txt**: Python dependencies.
- **README.md**: Documentation.

---

## **Contributing**

Feel free to fork this repository and contribute enhancements. Ensure all changes are documented and tested before submitting a pull request.

---

## **License**

This project is licensed under the MIT License.
