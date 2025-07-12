## 🧠 LLM Hub

Utilize the power of an LLM into ComfyUI to transform your text-to-image and text-to-video ideas into highly detailed and optimized prompts for generation while giving you full control. 

## 🤖 Usage

<img src="https://raw.githubusercontent.com/company8/ComfyUI_LLM_Hub/refs/heads/main/img/README.png" alt="LLM Hub for ComfyUI">

## 📝 Requirements

- Create a directory named **`LLMs`** inside **`ComfyUI/models/text_encoders/`**
- Create a another new directory for each LLM with the model name inside **`LLMs`**.
- Place your LLM models in their respective directory.
- Every .safetensors model needs the .json files and the model should be named **`model.safetensors`**

How your directory structure should look like:
```
ComfyUI/
└── models/
    └── text_encoders/
        └── LLMs/
            └── GGUF_model/
                └── model.gguf
            └── safetensors_model/
                └── model.safetensors
                └── config.json
                └── tokenizer.json
                └── tokenizer_config.json
                └── generation_config.json (Optional)
                └── special_tokens_map.json (Optional)
```

GGUF models don't need to be named "model.gguf".

## 🛠️ Installation

- Run the following command:
```
pip install -r --verbose requirements.txt
```
You can delete "--verbose" if you don't want to see the process of the compiling.

## ⚙️ LLM Settings 

The `LLM Settings` offers a range of configurable parameters allowing for precise control over the text
generation process and model behavior.

The values on this node are also the defaults that `LLM Hub` uses when `LLM Settings` isn't connected.

Breif overview of the parameters:

- **`temperature`**: Controls the randomness in the text generation process. Lower values make the model
  more confident in its predictions, leading to less variability in output. Higher values increase diversity but can
  also introduce more randomness. Default: `0.8`.
- **`top_p`**: Also known as nucleus sampling, this parameter controls the cumulative probability distribution
  cutoff. The model will only consider the top p% of tokens with the highest probabilities for sampling. Reducing this
  value helps in controlling the generation quality by avoiding low-probability tokens. Default: `0.8`.
- **`top_k`**: Limits the number of highest probability tokens considered for each step of the generation. A
  value of `0` means no limit. This parameter can prevent the model from focusing too narrowly on the top choices,
  promoting diversity in the generated text. Default: `10`.
- **`repetition_penalty`**: Adjusts the likelihood of tokens that have already appeared in the
  output, discouraging repetition. Values greater than `1` penalize tokens that have been used, making them less likely
  to appear again. Default: `1.3`.


## 📚 Resources

- <a href="https://rumn.medium.com/setting-top-k-top-p-and-temperature-in-llms-3da3a8f74832" target="_SEJ" rel="noreferrer">Parameter settings</a>

## 📄 License
Released under the MIT License. Feel free to use and modify it for your projects, commercial or personal.
