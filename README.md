# NanoAssistant

NanoAssistant is a simple, voice-based personal assistant built using OpenAI Whisper and the GPT-3.5-turbo API. It listens to the user's speech, transcribes it, generates a response, converts the text response to speech, and plays it back to the user.

## Installation

To get started with NanoAssistant, first clone the repository:

```
git clone https://github.com/username/nanoassistant.git
```

Next, navigate to the `nanoassistant` directory:

```
cd nanoassistant
```

Install the required dependencies using conda. First, create a new conda environment:

```
conda create --name nanoassistant_env
```

Activate the new environment:

```
conda activate nanoassistant_env
```

Install the required packages from the `requirements.txt` file:

```
conda install --file requirements.txt
```

## Setup

Before running the NanoAssistant, you'll need to set your OpenAI API key. In the `VoiceAssistant` class, replace the empty string in the following line with your OpenAI API key:

```python
openai.api_key = ""
```

## Usage

To start using the NanoAssistant, run the following command:

```
python nanoassistant.py
```

The assistant will listen to your speech, transcribe it, generate a response using OpenAI's GPT-3.5-turbo, and play the response back to you. To exit the program, simply say "goodbye."

## Dependencies

- openai
- gtts
- sounddevice
- numpy
- scipy
- tempfile
- simpleaudio
- librosa
- soundfile

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
