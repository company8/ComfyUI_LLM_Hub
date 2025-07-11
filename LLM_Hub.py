import importlib
import os
import random
import folder_paths
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, set_seed, BitsAndBytesConfig

GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, "text_encoders", "LLMs")

WEB_DIRECTORY = "./web/js"

try:
    Llama = importlib.import_module("llama_cpp_cuda").Llama
except ImportError:
    Llama = importlib.import_module("llama_cpp").Llama


try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Transformers library not found. Please install it to use safetensors models: pip install transformers torch sentencepiece")
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None


_loaded_model = None
_loaded_tokenizer = None
_current_model_id = None
_model_is_on_gpu = False

def free_gpu_memory():
    global _loaded_model, _loaded_tokenizer, _current_model_id, _model_is_on_gpu
    if _loaded_model is not None:
        del _loaded_model
        _loaded_model = None
    if _loaded_tokenizer is not None:
        del _loaded_tokenizer
        _loaded_tokenizer = None
    _current_model_id = None
    _model_is_on_gpu = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU memory freed.")


class AnyType(str):

    def __ne__(self, __value: object) -> bool:
        return False


anytype = AnyType("*")


class LLM_Hub:
    @classmethod
    def INPUT_TYPES(cls):
        model_options = []
        if os.path.isdir(GLOBAL_MODELS_DIR):
            listed_items = set()
            for root, dirs, files in os.walk(GLOBAL_MODELS_DIR):

                for file in files:
                    if file.endswith('.gguf'):
                        relative_path_to_file = os.path.relpath(os.path.join(root, file), GLOBAL_MODELS_DIR)
                        if os.path.dirname(relative_path_to_file) == ".":
                            display_name = os.path.splitext(relative_file_path)[0]
                        else:
                            clean_name = os.path.relpath(root, GLOBAL_MODELS_DIR) 
                        if clean_name not in listed_items:
                            model_options.append(clean_name)
                            listed_items.add(clean_name)

                if root == GLOBAL_MODELS_DIR:
                    for d in dirs:
                        current_dir_path = os.path.join(root, d)

                        if any(f.endswith(('.json', '.safetensors', '.bin')) for f in os.listdir(current_dir_path)):
                            clean_name = os.path.relpath(current_dir_path, GLOBAL_MODELS_DIR)
                            if clean_name not in listed_items:
                                model_options.append(clean_name)
                                listed_items.add(clean_name)

        if not model_options:
            model_options = ["No models found in text_encoders/LLMs"] 

        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "Write your prompt here."}),
                "seed": ("INT", {"default": 1234567890, "min": 0, "max": 0xffffffffffffffff}),
                "model": (model_options, {"tooltip": "Select an LLM. Store your models in text_encoders/LLMs/model_name for clarity."}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "tooltip": "The maximum number of tokens to generate. Higher values mean longer responses while also increasing generation time and VRAM usage."}),
                "context_window": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 256, "tooltip": "The total context window size (input + output tokens). Lower values reduce VRAM."}),
                "apply_system_prompt": ("BOOLEAN", {"default": True, "tooltip": "When enabled your system prompt in 'system_prompt' will be used."}),
				"load_in_8bit": ("BOOLEAN", {"default": True, "label_on": "8-bit", "label_off": "Full Precision", "tooltip": "When enabled to (8-bit), HuggingFace models will load in 8-bit to reduce VRAM usage. GGUF models are not affected, as they have their own quantization."}),
                "stay_on_gpu": ("BOOLEAN", {"default": False, "tooltip": "If enabled, the model will remain on the GPU until you restart ComfyUI or you disable this option. If disabled, the model will offload from the GPU after each run."}), 
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a prompt engineer that excells at creating image descriptions based on \"{prompt}\""}),
            },
            "optional": {
                "settings": ("SETTINGS_FOR_LLM", {"tooltip": "Control settings for the LLM."}),
            }
        }

    CATEGORY = "LLM_Hub"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("generated", "original",)


    def main(self, text, seed, model, max_tokens, context_window, apply_system_prompt, load_in_8bit, stay_on_gpu, system_prompt, settings=None):
        model_full_path = os.path.join(GLOBAL_MODELS_DIR, model)
        if os.path.exists(model_full_path + ".gguf"):
            model_full_path += ".gguf"
        elif os.path.exists(model_full_path + ".safetensors"):
            model_full_path += ".safetensors"


        generate_kwargs = {'max_tokens': max_tokens, 'temperature': 1.0, 'top_p': 0.9, 'top_k': 30,
                           'repeat_penalty': 1.2}

        if settings:
            for option in ['temperature', 'top_p', 'top_k', 'repeat_penalty']:
                if option in settings:
                    generate_kwargs[option] = settings[option]

		
        if apply_system_prompt:
            req = system_prompt.replace("{prompt}", text) if "{prompt}" in system_prompt else f"{system_prompt} {text}"
            messages_for_llm = [
                {"role": "system", "content": f"You are a helpful assistant."},
                {"role": "user", "content": f"An image generation prompt is a single paragraph summary to describe the subject and style of an image. It includes a description of the kind of image, the subject of the image, and some description of the image medium and style in the form of short keyword.\n\nCreate an image generation prompt for the subject \"a creepy creature shadow in the dark in a dimly lit tunnel\" in the style \"digital art illustration with intricate details\"."},
                {"role": "assistant", "content": f"Image Description: A digitally crafted illustration portrays a chilling scene within a dimly lit, cavernous tunnel. The dominant subject of the image is a mysterious creature, its form subtly discernible only as a menacing shadow on the walls of the tunnel. Its elongated silhouette suggests a serpentine creature with sharp protrusions and an ominous aura. The creature's eyes, a pair of glowing orbs, appear eerily human-like yet alien. The tunnel is riddled with intricate details that enhance the eerie atmosphere: dust particles floating lazily in the feeble light, ancient and crumbling stone, water droplets sparkling on the damp walls, and a hauntingly beautiful, bioluminescent fungi growing in the shadows. The dimly lit environment is highlighted by strategically placed light sources that create dramatic chiaroscuro effects, casting an unsettling and atmospheric glow on the scene. Digital Art Illustration with Intricate Details (Dark, Atmospheric, Suspenseful)"},
                {"role": "user", "content": "Now compile the description and style into a single paragraph summary"},
                {"role": "assistant", "content": f"Digital art illustration featuring a dark, atmospheric, and suspenseful scene within a dimly lit, cavernous tunnel. The subject of the image is a mysterious creature, depicted only as a menacing shadow on the walls, with elongated silhouette and sharp protrusions. The creature's eyes, a pair of glowing orbs, appear eerily human-like yet alien. The tunnel is adorned with intricate details, such as dust particles, ancient crumbling stone, water droplets, and a hauntingly beautiful bioluminescent fungi growing in the shadows. Dramatic chiaroscuro effects are created through strategically placed light sources, casting an unsettling and atmospheric glow on the scene."},
                {"role": "user", "content": f"create a detailed summary without the title or style"},
                {"role": "assistant", "content": f"A dimly lit, cavernous tunnel is the setting for this chilling digital illustration. A mysterious creature lurks in the shadows, its elongated silhouette suggestive of a serpentine creature with sharp protrusions and an ominous aura. The creature's eyes, a pair of glowing orbs, appear eerily human-like yet alien. The tunnel is riddled with intricate details that enhance the eerie atmosphere: dust particles floating lazily in the feeble light, ancient and crumbling stone, water droplets sparkling on the damp walls, and a hauntingly beautiful, bioluminescent fungi growing in the shadows. Dramatic chiaroscuro effects are created through strategically placed light sources, casting an unsettling and atmospheric glow on the scene."},
                {"role": "user", "content": f"Generate a prompt from \"magical garden, sitting on a blue_bench, Disney Princess in pink_dress, blonde_hair, portrait, Pixar, Disney style, photorealistic, high details, detailed skin, pores, freckles\""},
                {"role": "assistant", "content": f"In a captivating, Pixar-inspired, Disney-style, photorealistic portrait, a magical garden unfolds as a breathtaking backdrop. The subject of the image is a Disney Princess, dressed elegantly in a pink gown with flowing fabric, sitting serenely on a blue bench. The princess boasts long, blonde hair and a radiant smile. The garden is meticulously detailed, with vibrant, enchanted flora and fauna, as well as whimsical details like sparkling fairy lights and a picturesque waterfall. The princess is highlighted against the lush, detailed foliage, with a focus on the high-definition details of her porcelain skin, visible freckles, and the intricacies of her elegant gown. The image is rendered in the captivating, photorealistic style that exemplifies both the Disney and Pixar brands, capturing the princess's timeless beauty and the magic of her enchanting surroundings."},
                {"role": "user", "content": req},
            ]
        else:
            messages_for_llm = [
                {"role": "system", "content": f"You are a helpful assistant. Try your best to give the best response possible to the user."},
                {"role": "user", "content": f"Create a detailed visually descriptive caption of this description, which will be used as a prompt for a text to image AI system (caption only, no instructions like \"create an image\").Remove any mention of digital artwork or artwork style. Give detailed visual descriptions of the character(s), including ethnicity, skin tone, expression etc. Imagine using keywords for a still for someone who has aphantasia. Describe the image style, e.g. any photographic or art styles / techniques utilized. Make sure to fully describe all aspects of the cinematography, with abundant technical details and visual descriptions. If there is more than one image, combine the elements and characters from all of the images creatively into a single cohesive composition with a single background, inventing an interaction between the characters. Be creative in combining the characters into a single cohesive scene. Focus on two primary characters (or one) and describe an interesting interaction between them, such as a hug, a kiss, a fight, giving an object, an emotional reaction / interaction. If there is more than one background in the images, pick the most appropriate one. Your output is only the caption itself, no comments or extra formatting. The caption is in a single long paragraph. If you feel the images are inappropriate, invent a new scene / characters inspired by these. Additionally, incorporate a specific movie director's visual style and describe the lighting setup in detail, including the type, color, and placement of light sources to create the desired mood and atmosphere. Always frame the scene, including details about the film grain, color grading, and any artifacts or characteristics specific. Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than 250 characters."
                                f"\nDescription : {text}"},
            ]

		
        global _loaded_model, _loaded_tokenizer, _current_model_id, _model_is_on_gpu
        actual_model_path = None
        model_type = None 
        potential_path = os.path.join(GLOBAL_MODELS_DIR, model)


        if os.path.isfile(potential_path + '.gguf'):
            actual_model_path = potential_path + '.gguf'
            model_type = 'gguf'
        elif os.path.isdir(potential_path):
            gguf_files_in_dir = [f for f in os.listdir(potential_path) if f.endswith('.gguf')]
            if gguf_files_in_dir:
                actual_model_path = os.path.join(potential_path, gguf_files_in_dir[0])
                model_type = 'gguf'
            elif any(f.endswith(('.safetensors', '.bin', '.json')) for f in os.listdir(potential_path)):
                actual_model_path = potential_path
                model_type = 'hf_dir'
        
        if not actual_model_path or not model_type:
            return (f"Error: {model} not found or unsupported format. Please check the model file name and directory structure under {GLOBAL_MODELS_DIR}.", text)


        current_model_key = f"{actual_model_path}_offload:{stay_on_gpu}"

        should_reload = False
        if _loaded_model is None or _current_model_id != current_model_key:
            should_reload = True
            if _loaded_model is not None:
                print("Model or GPU setting changed, freeing existing model.")
                free_gpu_memory()
        elif _model_is_on_gpu and stay_on_gpu:

             pass
        elif not _model_is_on_gpu and not stay_on_gpu:

             pass


        if not _loaded_model or _current_model_id != current_model_key:
            print(f"Loading {model}...")

            try:
                if model_type == 'gguf':
                    if not Llama:
                        return ("Error: llama_cpp or llama_cpp_cuda not imported. Cannot load GGUF model.", text)
                    
                    n_gpu_layers_val = 0 
                    if torch.cuda.is_available():
                        n_gpu_layers_val = -1
                    else:
                        print("Warning: No GPU available. GGUF model will run on CPU.")

                    model_to_use = Llama(
                        model_path=actual_model_path,
                        n_gpu_layers=n_gpu_layers_val,
                        seed=int(seed % (2**32 - 1)),
                        verbose=False,
                        n_ctx=context_window,
                    )
                    _loaded_model = model_to_use
                    _model_is_on_gpu = (n_gpu_layers_val != 0)
                    _current_model_id = current_model_key

                elif model_type == 'hf_dir':
                    if not (AutoModelForCausalLM and AutoTokenizer and pipeline):
                        return ("Error: Transformers library not found. Cannot load safetensors model. Please install it: pip install transformers torch sentencepiece", text)
                    
                    hf_model_dir = actual_model_path
                    
                    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
                    
                    device_map_setting = "auto" if torch.cuda.is_available() else "cpu"
                    
                    if torch.cuda.is_available() and load_in_8bit:
                        quantization_status = "8-bit"
                        print(f"Loading HuggingFace model on GPU with {quantization_status}")
                        quant_config = BitsAndBytesConfig(load_in_8bit=True)
                        model_hf = AutoModelForCausalLM.from_pretrained(
                            hf_model_dir,
                            torch_dtype="auto",
                            device_map=device_map_setting,
                            quantization_config=quant_config
                        )
                        _model_is_on_gpu = True
                    elif torch.cuda.is_available() and not load_in_8bit:
                        model_hf = AutoModelForCausalLM.from_pretrained(
                            hf_model_dir,
                            torch_dtype="auto",
                            device_map=device_map_setting
                        )
                        _model_is_on_gpu = True
                        actual_dtype_str = str(model_hf.dtype).replace('torch.', '')
                        print(f"Loading HuggingFace model on GPU in {actual_dtype_str} precision")
                    else:
                        print(f"Loading HuggingFace model on CPU (no GPU available)")
                        model_hf = AutoModelForCausalLM.from_pretrained(hf_model_dir, torch_dtype=torch.float32, device_map="cpu")
                        _model_is_on_gpu = False

                    _loaded_model = model_hf
                    _loaded_tokenizer = tokenizer
                    _current_model_id = current_model_key
                   
                device_str = "GPU" if _model_is_on_gpu else "CPU"
                precision_display_part = "" 

                if model_type == 'hf_dir':
                    if _model_is_on_gpu:
                        if load_in_8bit:
                            precision_display_part = "8-bit"
                        else:
                            precision_display_part = str(model_hf.dtype).replace('torch.', '')
                    else:
                        precision_display_part = str(model_hf.dtype).replace('torch.', '')

                    if precision_display_part:
                        print(f"{model} loaded successfully on {device_str} in {precision_display_part}")
                    else: 
                        print(f"{model} loaded successfully on {device_str}")
                else:
                    print(f"{model} loaded successfully on {device_str}")


            except Exception as e:
                free_gpu_memory()
                return (f"Error loading {model} from {actual_model_path}: {e}", text)

            
        llm_result_content = ""
        try:
            capped_seed = seed % (2**32 - 1)
            set_seed(capped_seed)

            if model_type == 'gguf':
                llm_result = _loaded_model.create_chat_completion(messages_for_llm, **generate_kwargs)
                llm_result_content = llm_result['choices'][0]['message']['content'].strip()
            
            elif model_type == 'hf_dir': 

                if _model_is_on_gpu: 
                    generator = pipeline(
                        "text-generation",
                        model=_loaded_model,
                        tokenizer=_loaded_tokenizer,
                    )
                else:
                    generator = pipeline(
                        "text-generation",
                        model=_loaded_model,
                        tokenizer=_loaded_tokenizer,
                    )


                if hasattr(_loaded_tokenizer, 'apply_chat_template') and _loaded_tokenizer.chat_template:
                    chat_prompt = _loaded_tokenizer.apply_chat_template(messages_for_llm, tokenize=False, add_generation_prompt=True)
                else:
                    chat_prompt = ""
                    for message in messages_for_llm:
                        if message["role"] == "system":
                            chat_prompt += f"<|system|>{message['content']}\n"
                        elif message["role"] == "user":
                            chat_prompt += f"<|user|>{message['content']}\n"
                        elif message["role"] == "assistant":
                            chat_prompt += f"<|assistant|>{message['content']}\n"
                    chat_prompt += "<|assistant|>"

                hf_generate_kwargs = {
                    'max_new_tokens': generate_kwargs.get('max_tokens', 4096),
                    'temperature': generate_kwargs.get('temperature', 1.0),
                    'top_p': generate_kwargs.get('top_p', 0.9),
                    'top_k': generate_kwargs.get('top_k', 30),
                    'do_sample': True if generate_kwargs.get('temperature', 1.0) > 0 else False,
                    'repetition_penalty': generate_kwargs.get('repeat_penalty', 1.2),
                }
                
                result = generator(chat_prompt, **hf_generate_kwargs)
                generated_text = result[0]['generated_text']
                
                if generated_text.startswith(chat_prompt):
                    llm_result_content = generated_text[len(chat_prompt):].strip()
                else:
                    llm_result_content = generated_text.strip()

            
        except Exception as e:
            free_gpu_memory()
            return (f"Error during inference with model '{model}': {e}", text)


        if not stay_on_gpu and _model_is_on_gpu:
            print(f"Offloading {model} from GPU after run as Stay on GPU is disabled.")
            free_gpu_memory()
        elif stay_on_gpu and _model_is_on_gpu:
            print(f"{model} remains on GPU as Stay on GPU is enabled.")
        elif not _model_is_on_gpu:
            print(f"{model} remains on CPU.")
            

        return (llm_result_content, text)

class Generated_Output:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (anytype, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "LLM_Hub"
    FUNCTION = "main"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    
    def main(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None and len(extra_pnginfo) > 0:
            workflow = None
            if "workflow" in extra_pnginfo:
                workflow = extra_pnginfo["workflow"]
            node = None
            if workflow and "nodes" in workflow:
                node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                node["widgets_values"] = [str(text)]
        return {"ui": {"text": (str(text),)}}


class LLM_Settings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.05, "tooltip": "How creative or crazy the text is. High means wild text, low means boring text."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "step": 0.05, "tooltip": "Picks words that are good enough. High means more choices, low means very few choices."}),
                "top_k": ("INT", {"default": 30, "min": 0, "tooltip": "Picks from the best X words. High X means many choices, low X means few choices."}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.1, "step": 0.05, "tooltip": "Stops the text from saying the same thing too many times. High means less repeating."}),
            }
        }

    CATEGORY = "LLM_Hub"
    FUNCTION = "main"
    RETURN_TYPES = ("SETTINGS_FOR_LLM",)
    RETURN_NAMES = ("settings",)

    def main(self, temperature=1.0, top_p=0.9, top_k=30, repetition_penalty=1.2):
        options_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

        return (options_config,)


NODE_CLASS_MAPPINGS = {
    "LLM_Hub": LLM_Hub,
    "LLM_Settings": LLM_Settings,
    "Generated_Output": Generated_Output,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_Hub": "LLM Hub",
    "LLM_Settings": "LLM Settings",
    "Generated_Output": "Generated Output",
}
