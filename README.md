# 🧠 LLM Hub

A prompt-generator for ComfyUI, utilizing the power of a language model to turn a provided
text-to-image prompt into a more detailed and improved prompt.


## Requirements

Create a directory named "LLMs" in "ComfyUI/models/text_encoders/LLMs"
Place your LLM models there.
Create a new directory for each LLM with the model name, so you don't get confused and the node doesn't use the wrong one.
Every .safetensors model needs the .json files AND the model has to be named "model.safetensors" (Not my choice, that's HuggingFace because we are using Transformers for inference)
Reasoning behind the directory is if you use HiDream and tried to generate without the llama as a T.E (Text Encoder) it would produce garbage, this way you can use the same model for both T.E and as a prompt generator.

If you get error message about missing `llama-cpp`, try these manual steps:

- Run the following commands:
```
python -m pip install --verbose llama-cpp-python --config-settings=cmake.args="-DGGML_CUDA=on"
```
Delete "--config-settings=cmake.args="-DGGML_CUDA=on" if you don't have a GPU.
Note: "--verbose" is just to see the process of the compling, you can delete it if you wish.


## LLM Settings 
The `LLM Settings` offers a range of configurable parameters allowing for precise control over the text
generation process and model behavior.

*The values on this node are also the defaults that `LLM_Hub`*
*uses when `LLM_Settings` isn't connected.*

Below is a detailed overview of these parameters:

- **Temperature (`temperature`):** Controls the randomness in the text generation process. Lower values make the model
  more confident in its predictions, leading to less variability in output. Higher values increase diversity but can
  also introduce more randomness. Default: `1.0`.
- **Top-p (`top_p`):** Also known as nucleus sampling, this parameter controls the cumulative probability distribution
  cutoff. The model will only consider the top p% of tokens with the highest probabilities for sampling. Reducing this
  value helps in controlling the generation quality by avoiding low-probability tokens. Default: `0.9`.
- **Top-k (`top_k`):** Limits the number of highest probability tokens considered for each step of the generation. A
  value of `0` means no limit. This parameter can prevent the model from focusing too narrowly on the top choices,
  promoting diversity in the generated text. Default: `50`.
- **Repetition Penalty (`repetition_penalty`):** Adjusts the likelihood of tokens that have already appeared in the
  output, discouraging repetition. Values greater than `1` penalize tokens that have been used, making them less likely
  to appear again. Default: `1.2`.

These parameters provide granular control over the text generation capabilities of the `Searge_LLM_Node`, allowing
users to fine-tune the behavior of the underlying models to best fit their application requirements.


## License
Released under the MIT License. Feel free to use and modify it for your projects, commercial or personal.
