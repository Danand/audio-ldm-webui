import torch
import scipy

from diffusers.pipelines.audioldm2.pipeline_audioldm2 import AudioLDM2Pipeline
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput

import librosa

import numpy as np
import streamlit as st

import time
import hashlib
import json

from typing import List, cast
from os.path import join, realpath, exists
from os import makedirs, remove
from platform import system
from dataclasses import dataclass

SAMPLE_RATE_DEFAULT = 16000
TEMP_DIR = "temp"
SESSION_STATE_PATH = f"{TEMP_DIR}/session-state.json"

@dataclass(frozen=True)
class OutputAudioInfo:
    model: str
    positive_prompt: str
    negative_prompt: str
    seed: int
    steps: int
    guidance_scale: float
    duration: float
    index: int
    tempo: float
    note: str

def get_available_devices() -> List[str]:
    devices = [
        "cpu",
        "cuda",
        "mps",
    ]

    if torch.cuda.is_available():
        devices.remove("cuda")
        devices.insert(0, "cuda")

    if torch.backends.mps.is_available() and system() == "Darwin":
        devices.remove("mps")
        devices.insert(0, "mps")

    return devices

@st.cache_resource
def load_pipeline(
    model_repo: str,
    device: str | None,
) -> AudioLDM2Pipeline:
    pipe: AudioLDM2Pipeline = AudioLDM2Pipeline.from_pretrained(
        pretrained_model_name_or_path=model_repo,
        torch_dtype=torch.float32,
    ) # type: ignore

    pipe: AudioLDM2Pipeline = pipe.to(device)

    return pipe

def format_output_audio_file_name(info: OutputAudioInfo) -> str:
    keyword = info.positive_prompt.split(",")[0]

    keyword_clean = keyword.replace(" ", "-")
    keyword_clean = keyword_clean.replace("_", "-")
    keyword_clean = keyword_clean.lower()

    file_name = keyword_clean

    note = info.note
    note_clean = note.replace("#", "sharp")

    file_name += f"_{note_clean}"

    file_name += f"-{round(info.tempo)}bpm"
    file_name += f"-{info.steps}"
    file_name += f"-{info.guidance_scale:.2f}"
    file_name += f"-{info.duration:.2f}"
    file_name += f"-{info.index}"

    info_json = json.dumps(info.__dict__)
    info_hash = hashlib.sha1(info_json.encode()).hexdigest()[:8]

    file_name += f"-{info_hash}"

    file_name += ".wav"

    return file_name

title = "AudioLDM 2: Web UI"

st.set_page_config(
    page_title=title,
)

st.title(title)

with st.container(border=True):
    st.subheader(
        body="Settings",
        divider="orange",
    )

    if exists(SESSION_STATE_PATH):
        if st.button("Reset"):
            remove(SESSION_STATE_PATH)
            st.session_state.clear()

    if not st.session_state.get("is_loaded_session_state", False):
        if exists(SESSION_STATE_PATH):
            with open(SESSION_STATE_PATH, "r") as file:
                session_state_loaded = json.loads(file.read())

                for key, value in dict(session_state_loaded).items():
                    st.session_state[key] = value

        st.session_state["is_loaded_session_state"] = True

    model_repo = st.selectbox(
        label="Model",
        options=[
            "cvssp/audioldm2",
            "cvssp/audioldm2-large",
            "cvssp/audioldm2-music",
        ],
        key="model_repo",
    )

    devices = get_available_devices()

    device_chosen = st.radio(
        label="Device",
        options=devices,
        horizontal=True,
        help="Please check either device type is supported on your machine.",
        index=devices.index(st.session_state.get("device_chosen", devices[0])),
    )

    st.session_state["device_chosen"] = device_chosen

    positive_prompt = st.text_area(
        label="Positive Prompt",
        value=st.session_state.get("positive_prompt", ""),
    )

    st.session_state["positive_prompt"] = positive_prompt

    negative_prompt = st.text_area(
        label="Negative Prompt",
        value=st.session_state.get("negative_prompt", ""),
    )

    st.session_state["negative_prompt"] = negative_prompt

    steps = st.number_input(
        label="Steps",
        format="%i",
        value=st.session_state.get("steps", 200),
        min_value=1,
    )

    st.session_state["steps"] = steps

    guidance_scale = st.number_input(
        label="Guidance Scale",
        value=st.session_state.get("guidance_scale", 3.5),
        min_value=0.0,
    )

    st.session_state["guidance_scale"] = guidance_scale

    seed = st.number_input(
        label="Seed",
        format="%i",
        value=st.session_state.get("seed", 0),
    )

    st.session_state["seed"] = seed

    duration = st.number_input(
        label="Duration (seconds)",
        value=st.session_state.get("duration", 1.0),
        min_value=0.04,
    )

    st.session_state["duration"] = duration

    amount = st.number_input(
        label="Audio clips amount",
        format="%i",
        value=st.session_state.get("amount", 1),
        min_value=1,
    )

    st.session_state["amount"] = amount

    button_generate = st.empty()

container_progress = st.empty()

container_output = st.empty()

if button_generate.button(
    label="Generate",
    type="primary",
    use_container_width=True,
):
    with container_progress.container(border=True):
        st.subheader("Progress")
        progress_steps = st.empty()
        text_time_elapsed = st.empty()

    def on_progress_steps(step: int, timestep: int, tensor: torch.FloatTensor | None):
        steps_completed = step + 1

        progress_steps.progress(
            value=steps_completed / float(steps),
            text=f"Steps completed: {steps_completed}/{steps}",
        )

    progress_steps.text("Initializing pipeline...")

    pipe: AudioLDM2Pipeline = load_pipeline(cast(str, model_repo), device_chosen)

    generator = torch.Generator(device_chosen).manual_seed(int(seed))

    time_start = time.time()

    pipe_output: AudioPipelineOutput = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=cast(int, steps),
        guidance_scale=cast(float, guidance_scale),
        audio_length_in_s=cast(float, duration),
        num_waveforms_per_prompt=cast(int, amount),
        generator=generator,
        callback=on_progress_steps,
        callback_steps=1,
    ) # type: ignore

    time_end = time.time()
    time_elapsed = time_end - time_start

    text_time_elapsed.text("Completed.")
    text_time_elapsed.text(f"Time elapsed: {time_elapsed:.2f} s.")

    audios: np.ndarray = pipe_output.audios

    with container_output.container(border=True):
        st.subheader(
            body="Generated Audio",
            divider="orange"
        )

        index = 0

        for audio in audios:
            with st.container(border=True):
                onset_envelope = librosa.onset.onset_strength(
                    y=audio,
                    sr=SAMPLE_RATE_DEFAULT,
                    hop_length=512,
                )

                tempo, _ = librosa.beat.beat_track(
                    onset_envelope=onset_envelope,
                    sr=SAMPLE_RATE_DEFAULT,
                )

                spectral_rolloff = librosa.feature.spectral_rolloff(
                    y=audio,
                    sr=SAMPLE_RATE_DEFAULT,
                )

                frequency_median = np.median(spectral_rolloff)
                note_median = librosa.hz_to_note(frequency_median)
                note_median = note_median.replace("\u266f", "#")

                output_audio_info = OutputAudioInfo(
                    model=cast(str, model_repo),
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    seed=cast(int, seed),
                    steps=cast(int, steps),
                    guidance_scale=cast(float, guidance_scale),
                    duration=cast(float, duration),
                    index=index,
                    tempo=tempo,
                    note=note_median,
                )

                output_dir = "outputs"

                makedirs(name=output_dir, exist_ok=True)

                file_name = format_output_audio_file_name(output_audio_info)

                output_path = realpath(
                    join(
                        output_dir,
                        file_name,
                    ),
                )

                scipy.io.wavfile.write(
                    filename=output_path,
                    rate=SAMPLE_RATE_DEFAULT,
                    data=audio,
                )

                st.code(
                    body=output_path,
                    language="bash",
                )

                st.audio(output_path)

                st.json(
                    body=output_audio_info.__dict__,
                    expanded=False,
                )

            index += 1

makedirs(TEMP_DIR, exist_ok=True)

with open(SESSION_STATE_PATH, "w") as file:
    file.write(
        json.dumps(
            obj=dict(st.session_state),
            indent=2,
        )
    )
