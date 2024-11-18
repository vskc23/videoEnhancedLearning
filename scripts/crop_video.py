import subprocess

def crop_video(input_file, start_time, end_time, output_file):
    """
    Crop a video between start_time and end_time using FFmpeg.

    Parameters:
    - input_file (str): Path to the input video file.
    - start_time (str): Starting timestamp in the format 'HH:MM:SS'.
    - end_time (str): Ending timestamp in the format 'HH:MM:SS'.
    - output_file (str): Path for the output cropped video file.
    """
    # FFmpeg command to crop the video between start_time and end_time
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ss', start_time,
        '-to', end_time,
        '-c', 'copy',  # Copy codec for faster processing
        output_file
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video successfully cropped: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error cropping video: {e.stderr.decode('utf-8')}")
