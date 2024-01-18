import torch
import scipy

from diffusers.pipelines.audioldm2.pipeline_audioldm2 import AudioLDM2Pipeline
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput

from numpy import ndarray

import streamlit as st

import time
import hashlib

from typing import cast
from os.path import join, realpath
from os import makedirs

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

def get_hash(
    positive_prompt: str,
    negative_prompt: str,
    steps: int,
    duration: float,
    amount: int
):
    hash_source = f"Positive prompt: {positive_prompt}\n"
    hash_source += f"Negative prompt: {negative_prompt}\n"
    hash_source += f"Steps: {steps}\n"
    hash_source += f"Duration: {duration}\n"
    hash_source += f"Amount: {amount}"

    return hashlib.sha1(hash_source.encode()).hexdigest()

st.title("AudioLDM 2: Web UI")

with st.container(border=True):
    st.subheader("Settings")

    model_repo = st.selectbox(
        label="Model",
        options=[
            "cvssp/audioldm2",
            "cvssp/audioldm2-large",
            "cvssp/audioldm2-music",
        ],
    )

    device_chosen = st.radio(
        label="Device",
        options=[
            "cpu",
            "cuda",
            "mps",
        ],
        horizontal=True,
        help="Please check either device type is supported on your machine.",
    )

    positive_prompt = st.text_area(
        label="Positive Prompt",
    )

    negative_prompt = st.text_area(
        label="Negative Prompt",
    )

    steps = st.number_input(
        label="Steps",
        format="%i",
        value=200,
        min_value=1,
    )

    seed = st.number_input(
        label="Seed",
        format="%i",
        value=0,
    )

    duration = st.number_input(
        label="Duration (seconds)",
        value=1.0,
        min_value=0.04,
    )

    amount = st.number_input(
        label="Audio clips amount",
        format="%i",
        value=1,
        min_value=1,
    )

    button_generate = st.empty()

container_progress = st.empty()

container_output = st.empty()

if button_generate.button(
    label="Generate",
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
        positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=cast(int, steps),
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

    audios: ndarray = pipe_output.audios

    sample_rate = audios.shape[1]

    with container_output.container(border=True):
        st.subheader("Generated Audio")

        index = 0

        for audio in audios:
            with st.container(border=True):
                st.audio(
                    data=audio,
                    sample_rate=sample_rate,
                )

                output_dir = "outputs"

                makedirs(name=output_dir, exist_ok=True)

                output_hash = get_hash(
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    steps=cast(int, steps),
                    duration=cast(float, duration),
                    amount=cast(int, amount),
                )

                output_path = realpath(
                    join(
                        output_dir,
                        f"{output_hash}-{index}.wav",
                    ),
                )

                scipy.io.wavfile.write(
                    filename=output_path,
                    rate=sample_rate,
                    data=audio,
                )

                st.code(
                    body=output_path,
                    language="bash",
                )

            index += 1

