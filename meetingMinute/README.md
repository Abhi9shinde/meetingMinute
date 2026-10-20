# Meeting Minute Generator

Meeting Minute Generator is a small, notebook-based speech-to-text and
large-language-model (LLM) application. It takes an English meeting recording,
transcribes it with OpenAI Whisper, and turns the transcript into structured
meeting minutes with Meta Llama.

The current prototype is implemented in
[Untitled1.ipynb](./Untitled1.ipynb) and is designed to run in Google Colab
with a CUDA-enabled GPU. The notebook currently uses a Denver council meeting
extract as its example input, but the same workflow can be used for other
English meeting recordings.

## What it produces

The LLM is instructed to produce Markdown minutes containing:

- A meeting summary
- Attendees, location, and date when they are available in the transcript
- Main discussion points
- Key takeaways
- Action items and their owners

The output is displayed directly in the notebook as rendered Markdown.

## How it works

The notebook follows this pipeline:

```text
Audio file
   |
   v
Whisper Medium English
   |
   v
Plain-text transcript
   |
   v
Prompt containing the transcript
   |
   v
Llama 3.2 3B Instruct, 4-bit quantized
   |
   v
Markdown meeting minutes
```

### 1. Install the model dependencies

The first notebook cell installs or upgrades:

- `transformers==4.57.6`
- `bitsandbytes`
- `accelerate`

Other imports used by the notebook come from the Colab runtime and the
Hugging Face/PyTorch ecosystem, including `torch`, `huggingface_hub`, and
`IPython`.

### 2. Load the input recording

The notebook currently expects the recording at:

```python
audio_filename = "/content/denver_extract.mp3"
```

Place the audio file at that path in the Colab runtime, or change
`audio_filename` to the path of the file you want to process. The current
notebook opens the file before transcription, although the opened file handle
is not otherwise used.

### 3. Transcribe the audio

The automatic speech-recognition pipeline uses:

```text
openai/whisper-medium.en
```

It is configured to use CUDA and returns timestamps:

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium.en",
    dtype=torch.float16,
    device="cuda",
    return_timestamps=True,
)
```

The notebook passes the audio path to the pipeline and stores the resulting
text in `transcription`.

### 4. Generate meeting minutes

The transcript is inserted into a system/user chat prompt. The system
instruction asks the model to produce meeting minutes in Markdown without
wrapping the result in a code block.

The generation model is:

```text
meta-llama/Llama-3.2-3B-Instruct
```

It is loaded with a `BitsAndBytesConfig` using:

- 4-bit loading
- NF4 quantization
- Double quantization
- `bfloat16` compute
- Automatic device mapping

The notebook generates up to 2,000 new tokens and streams generated text to
the notebook with `TextStreamer`. It then decodes and displays the response
using `IPython.display.Markdown`.

## Requirements

### Recommended runtime

The notebook metadata is configured for a Python 3 Google Colab runtime with a
GPU accelerator. A Colab T4 GPU is the original tested target.

Recommended minimum assumptions:

- Python 3
- A CUDA-enabled GPU
- Enough GPU memory for Whisper Medium and the quantized Llama model
- Internet access to download models from Hugging Face
- An English audio recording in a format supported by the installed audio
  backend, such as MP3 or WAV

CPU-only execution is not the intended configuration. It would require
changing the Whisper pipeline device and likely reducing model sizes or
accepting substantially slower execution.

### Hugging Face access

The notebook authenticates with Hugging Face using a Colab secret named
`HF_TOKEN`:

```python
hf_token = userdata.get("HF_TOKEN")
login(hf_token, add_to_git_credential=True)
```

Create a Hugging Face access token with permission to download the models,
accept any applicable model terms on Hugging Face, and add the token to
Google Colab:

1. Open the notebook in Google Colab.
2. Open **Secrets** in the left sidebar.
3. Add a secret named `HF_TOKEN`.
4. Paste the token value.
5. Enable notebook access to the secret if Colab asks for permission.

Do not paste the token directly into the notebook or commit it to this
repository.

## Running in Google Colab

1. Open [Untitled1.ipynb](./Untitled1.ipynb) in Google Colab.
2. Select **Runtime > Change runtime type**.
3. Choose a GPU accelerator.
4. Add the `HF_TOKEN` secret as described above.
5. Upload the meeting audio file to the Colab runtime, or mount Google Drive
   and update `audio_filename`.
6. Run the cells from top to bottom:
   - Install dependencies.
   - Import libraries.
   - Set the audio path.
   - Set the Llama model name.
   - Authenticate with Hugging Face.
   - Transcribe the audio.
   - Build the prompt.
   - Load the quantized Llama model.
   - Generate and display the minutes.

For a direct upload, the file can be uploaded from a Colab cell with:

```python
from google.colab import files

uploaded = files.upload()
audio_filename = next(iter(uploaded))
```

Run that cell before the transcription cell and make sure the resulting value
of `audio_filename` points to the uploaded file.

For a file on Google Drive, mount Drive and use its mounted path:

```python
from google.colab import drive

drive.mount("/content/drive")
audio_filename = "/content/drive/MyDrive/path/to/meeting.mp3"
```

## Running outside Colab

The notebook is currently Colab-oriented. A local setup needs additional
adaptation because it imports `google.colab.drive` and
`google.colab.userdata`, and it assumes that CUDA is available.

At a high level, a local environment should:

1. Create and activate a Python virtual environment.
2. Install a CUDA-compatible PyTorch build for the installed NVIDIA driver.
3. Install the notebook dependencies:

   ```bash
   pip install --upgrade "transformers==4.57.6" bitsandbytes accelerate \
       torch huggingface_hub ipython
   ```

4. Replace `userdata.get("HF_TOKEN")` with an environment variable or a local
   Hugging Face login, for example:

   ```python
   import os
   from huggingface_hub import login

   login(os.environ["HF_TOKEN"])
   ```

5. Remove or replace the `google.colab` imports.
6. Change the hard-coded audio path to a local path.
7. Run the equivalent notebook cells or extract them into a Python script.

Install PyTorch using the command recommended by the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/) so
that its CUDA version matches the local machine. `bitsandbytes` also requires
a compatible environment; consult its
[installation documentation](https://huggingface.co/docs/bitsandbytes/installation)
if model loading fails.

## Configuration

The main values to customize are near the beginning of the notebook:

| Setting | Current value | Purpose |
| --- | --- | --- |
| `audio_filename` | `/content/denver_extract.mp3` | Input audio path |
| `LLAMA` | `meta-llama/Llama-3.2-3B-Instruct` | Text-generation model |
| Whisper model | `openai/whisper-medium.en` | Speech-recognition model |
| `max_new_tokens` | `2000` | Maximum generated output length |
| Whisper `device` | `cuda` | Device used for transcription |
| Whisper `dtype` | `torch.float16` | Whisper inference data type |

The prompt itself is defined in the cell that creates `system_message` and
`user_prompt`. That is the place to change the required sections, writing
style, language, or domain-specific instructions.

## Practical considerations

- **Audio language:** The configured Whisper checkpoint is English-specific.
  Use an appropriate multilingual Whisper checkpoint if recordings are in
  another language.
- **Long recordings:** Very long meetings may exceed the context available to
  the Llama model when the complete transcript is inserted into one prompt.
  A production implementation should chunk the transcript and combine the
  intermediate summaries.
- **Timestamps:** Whisper is configured with `return_timestamps=True`, but the
  current code uses only `result["text"]`. Timestamp segments are not included
  in the final prompt.
- **Accuracy:** Names, dates, attendees, and action-item owners should be
  reviewed against the recording. The model may infer or incorrectly assign
  details when the transcript is unclear.
- **Output decoding:** The notebook decodes the complete generated sequence.
  Depending on the tokenizer/model output, this can include the original
  prompt before the generated answer. If only the new answer is desired,
  decode the tokens after the input length.
- **Model downloads:** The first run downloads large model files and may take
  time. Later runs can use the runtime cache until the Colab session is
  discarded.
- **Temporary storage:** Files uploaded to a Colab runtime are temporary and
  are removed when the runtime is reset or disconnected.

## Troubleshooting

### `401`, `403`, or model access errors

Confirm that:

- `HF_TOKEN` exists in Colab Secrets and notebook access is enabled.
- The token has sufficient read access.
- Any license or access request for the selected Llama model has been
  accepted on Hugging Face.

### CUDA or device errors

Confirm that the runtime has a GPU and that PyTorch can see it:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU")
```

If no GPU is available, do not use the current `device="cuda"` configuration.
Use a compatible CPU configuration or switch to a GPU runtime, keeping in mind
that CPU inference will be considerably slower.

### Out-of-memory errors

Try one or more of the following:

- Use a smaller Whisper checkpoint.
- Use a smaller Llama checkpoint.
- Reduce `max_new_tokens`.
- Process a shorter audio segment.
- Use a runtime with more GPU memory.
- Reduce the transcript length by chunking and summarizing it before final
  minute generation.

### Audio loading or format errors

Verify that the path exists and that the runtime has an audio decoding backend
available. Converting the input to WAV or MP3 and retrying is often useful.

## Project status and limitations

This repository contains a proof-of-concept notebook rather than a packaged
application. It currently does not provide:

- A command-line interface
- A web UI
- A `requirements.txt` or `pyproject.toml`
- Persistent result storage
- Automatic export to a Markdown or document file
- Speaker diarization
- Transcript chunking for arbitrarily long meetings
- Automated tests

The notebook is therefore best suited for experimentation and demonstrations.
Before using it for sensitive or official records, review the generated
minutes, protect the audio/transcript data, and define an appropriate retention
and access policy.

## Repository layout

```text
meetingMinute/
├── README.md          # Project documentation
└── Untitled1.ipynb    # Colab notebook implementing the full workflow
```

## License and model terms

This project does not add a separate license or redistribute the downloaded
model weights. Review and comply with the licenses and usage terms for:

- [Transformers](https://github.com/huggingface/transformers)
- [Whisper](https://huggingface.co/openai/whisper-medium.en)
- [Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

The model terms may impose requirements that are separate from the code in
this repository.