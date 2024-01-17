import torch

from diffusers.pipelines.audioldm2.pipeline_audioldm2 import AudioLDM2Pipeline
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput

from numpy import ndarray

import streamlit as st

import time

from typing import cast

MODEL_REPO_DEFAULT="cvssp/audioldm2"

st.title("AudioLDM 2: Web UI")

with st.container(border=True):
    st.subheader("Settings")

    model_repo = st.text_input(
        label="Repo",
        value=MODEL_REPO_DEFAULT,
        placeholder=MODEL_REPO_DEFAULT,
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

    pipe: AudioLDM2Pipeline = AudioLDM2Pipeline.from_pretrained(
        pretrained_model_name_or_path=model_repo,
        torch_dtype=torch.float32,
    ) # type: ignore

    pipe: AudioLDM2Pipeline = pipe.to(device_chosen)

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

        for audio in audios:
            st.audio(
                data=audio,
                sample_rate=sample_rate,
            )

