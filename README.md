# WhisperScribe

WhisperScribe is a Python script that transcribes audio files and performs speaker diarization. It supports MP3 and FLAC audio formats and utilizes OpenAI's Whisper model for transcription and Resemblyzer combined with SpectralCluster for speaker recognition. The script is user-friendly, allowing you to specify the number of speakers or a range if unknown. It outputs a text file with the transcribed speech, including speaker labels.

## Features

- **Automatic Transcription**: Converts audio files into text using OpenAI's Whisper model.
- **Speaker Diarization**: Identifies and labels different speakers within the audio.
- **Multiple Audio Formats**: Supports both MP3 and FLAC files.
- **User Interaction**: Allows users to specify the number of speakers or a range per audio file.
- **Clean Up**: Automatically deletes temporary WAV files after processing.
- **Colorful Console Output**: Enhances user experience with color-coded messages using Colorama.

## Table of Contents

- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Detailed Explanation](#detailed-explanation)
  - [Transcription with Whisper](#transcription-with-whisper)
  - [Speaker Embeddings with Resemblyzer](#speaker-embeddings-with-resemblyzer)
  - [Clustering with SpectralCluster](#clustering-with-spectralcluster)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/fersenhaIter/WhisperScribe.git
cd WhisperScribe
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
pydub
openai-whisper
resemblyzer
spectralcluster
colorama
```

Ensure that FFmpeg is installed on your system:

### For Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

### For Fedora:

```bash
sudo dnf install ffmpeg
```

## Prerequisites

- Python 3.7 or higher
- FFmpeg
- An internet connection for downloading models (only needed on first run)

## Usage

Place your MP3 or FLAC audio files in the same directory as the script. Run the script using:

```bash
python transkription.py
```

For each audio file, you will be prompted to enter the number of speakers or a range if unknown:

```
Processing file: interview.mp3
Enter the number of speakers for 'interview.mp3'.
If unknown, enter a range (e.g., '2-5').
Number of speakers or range: 2
```

After processing, a text file with the same name as the audio file (but with a `.txt` extension) will be created in the same directory.

### Command-Line Arguments (Optional)

You can modify the script to accept command-line arguments for batch processing or integrate it into other workflows.

## Detailed Explanation

### Transcription with Whisper

The script uses OpenAI's Whisper model to transcribe audio files. Whisper is a state-of-the-art speech recognition model that provides highly accurate transcriptions.

- **Model Loading**: The `base` Whisper model is loaded at the beginning of the script.
  ```python
  modell = whisper.load_model("base")
  ```
- **Transcription**: The audio file is transcribed into text with timestamps for each segment.
  ```python
  ergebnis = modell.transcribe(wav_datei, language='de', task='transcribe', verbose=False)
  ```
- **Result**: The transcription result includes the full text and segments with start and end times.

### Speaker Embeddings with Resemblyzer

Resemblyzer is used to extract speaker embeddings from audio segments to differentiate between speakers.

- **Audio Preprocessing**: The audio is converted into a WAV file and loaded for processing.
  ```python
  wav_audio = preprocess_wav(wav_datei)
  ```
- **Segment Extraction**: For each transcribed segment, a 0.5-second audio clip centered in the segment is extracted.
  ```python
  start_extract = max(0, middle_time - 0.25)
  end_extract = min(len(wav_audio) / 16000, middle_time + 0.25)
  audio_segment = wav_audio[int(start_extract * 16000):int(end_extract * 16000)]
  ```
- **Embedding Extraction**: Each audio segment is passed through the Voice Encoder to obtain a speaker embedding.
  ```python
  embedding = encoder.embed_utterance(audio_segment)
  ```

### Clustering with SpectralCluster

SpectralCluster performs speaker diarization by clustering the embeddings.

- **Clustering Initialization**: The number of clusters is set based on user input.
  ```python
  clusterer = SpectralClusterer(
      min_clusters=min_speakers,
      max_clusters=max_speakers,
  )
  ```
- **Embedding Clustering**: The embeddings are clustered to identify different speakers.
  ```python
  clusters = clusterer.predict(embeddings_array)
  ```
- **Speaker Assignment**: Each transcribed segment is assigned a speaker label based on the clustering results.
  ```python
  speaker = f"Speaker_{clusters[cluster_index]}"
  ```

### Cleanup

- **Temporary Files**: The temporary WAV files created during processing are deleted to save disk space.
  ```python
  os.remove(wav_datei)
  ```

## Examples

### Sample Output

```
Speaker_0: Hello, how are you today?
Speaker_1: I'm doing well, thank you!
Speaker_0: Glad to hear that.
```

### Console Output

```
Loading Whisper model...
Initializing Voice Encoder...

Processing file: conversation.mp3
Enter the number of speakers for 'conversation.mp3'.
If unknown, enter a range (e.g., '2-5').
Number of speakers or range: 2
Initializing Spectral Clusterer...
Converting audio...
Performing transcription...
Loading audio for speaker recognition...
Extracting embeddings for each segment...
Performing speaker clustering...
Creating transcript with speaker labels...
Transcript saved as conversation.txt
Deleting temporary file: conversation.wav

All files have been processed.
```

## Troubleshooting

- **Issue**: `ModuleNotFoundError` when running the script.
  - **Solution**: Ensure all required packages are installed by running `pip install -r requirements.txt`.

- **Issue**: FFmpeg errors or audio conversion failures.
  - **Solution**: Verify that FFmpeg is correctly installed and accessible in your system's PATH.

- **Issue**: Poor transcription accuracy.
  - **Solution**: Consider using a larger Whisper model (e.g., `medium`, `large`) for better accuracy. Note that larger models require more resources.
    ```python
    modell = whisper.load_model("medium")
    ```

- **Issue**: Speaker diarization is inaccurate.
  - **Solution**: Ensure that the number of speakers is correctly specified. If unknown, provide a reasonable range.
