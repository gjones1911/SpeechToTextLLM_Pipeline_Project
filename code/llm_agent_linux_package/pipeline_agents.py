import transformers
# Transformers and model loading
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    TextStreamer
)
import gradio as gr
import gradio_client
import numpy
import gc
import torch
from dotenv import dotenv_values
import os
from typing import Optional, Dict, Any, List, Tuple
from .gpu_tools import  get_gpu_with_most_free_memory

print("os wdir: ", os.getcwd())
# Check if a CUDA-compatible GPU is available
gpu_available = torch.cuda.is_available()

# PULL my environment variables (API keys so I can use them)
config = dotenv_values("../env/env_config")

# Set them as environment variables
for key, value in config.items():
    os.environ[key] = value

# list of llms that do not require any auth_token to use
open_llms_no_token_required = [
    {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Instruction-tuned Mistral 7B model with strong reasoning and generation capabilities.",
        "pros": ["Excellent general-purpose LLM", "Performs well in RAG", "Fast inference"],
        "cons": ["May require GPU with ~12GB+ VRAM for best performance"]
    },
    {
        "model": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "description": "DPO fine-tuned Mistral variant for assistant-like dialog and reasoning.",
        "pros": ["Great for conversation and QA", "Performs well in long-context tasks", "No token needed"],
        "cons": ["Heavier than Phi models"]
    },
    {
        "model": "teknium/OpenHermes-2.5-Mistral-7B",
        "description": "Mistral-based chat model tuned for helpful assistant-style responses.",
        "pros": ["Efficient", "Instruction following", "Good with injected context (RAG)"],
        "cons": ["May hallucinate if not grounded"]
    },
    {
        "model": "microsoft/phi-2",
        "description": "Compact transformer trained with curriculum learning, ideal for reasoning and basic chat.",
        "pros": ["Lightweight", "Good accuracy per parameter", "No token required"],
        "cons": ["Limited context window (2k)"]
    },
    {
        "model": "microsoft/phi-3-mini-4k-instruct",
        "description": "Latest 4k context Phi-3 model, small but powerful for structured assistant tasks.",
        "pros": ["Efficient and very fast", "Strong coding and reasoning", "Great for edge devices"],
        "cons": ["Limited generative depth compared to larger models"]
    },
    {
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Very small LLaMA-style model trained for chat and instruction following.",
        "pros": ["Extremely lightweight", "Can run on CPU", "Good for constrained environments"],
        "cons": ["Not as strong in generalization or long reasoning"]
    },
    {
        "model": "OpenAccessAI/MythoMax-L2-13b",
        "description": "A powerful LLaMA-2-based model fine-tuned for rich, open-ended conversation.",
        "pros": ["Powerful and expressive", "Fine-tuned on a diverse instruction dataset", "No token required"],
        "cons": ["Large (13B) - needs ~24GB+ VRAM or GGUF format"]
    },
    {
        "model": "openchat/openchat-3.5-0106",
        "description": "Chat-focused LLaMA-based model for helpful assistant behaviors.",
        "pros": ["Chat-optimized", "Can be used in RAG", "Open access"],
        "cons": ["Some versions require more VRAM"]
    }
]

open_model_names = [d["model"] for d in open_llms_no_token_required]

def show_open_model_names():
    for m in open_model_names:
        print(f"🤖 >> '{m}'")

def show_open_model_descriptions():
    print("📢 Your options for TTS models are: ")
    for d in open_llms_no_token_required:
        mstr = (
            f"Model: '{d['model']}'\n"
            f"Description: {d['description']}"
            f"Pros: {','.join(d['pros'])}\n"
            f"Cons: {','.join(d['cons'])}\n\n"
        )
        print(mstr)

# show_open_model_descriptions()
show_open_model_names()

class PipelineAgent:
    def __init__(self, 
                 model_name: str,
                 auth_token: Optional[str] = None,
                 temperature: float=.7,
                 top_k: int=50,
                 top_p: float=.80,
                 max_new_tokens: int=4000, max_tokens: int=8000,
                 stop_words: list=None,
                 device: str = "cuda",                # auto, cpu, cuda
                 save_context: bool=True,
                 verbose: bool=True,
                 **kwargs,
                ):


        self.model_name = model_name
        self.auth_token = auth_token
        self.temperature = temperature
        self.do_sample = True if temperature else False
        self.top_k = top_k if self.do_sample else None
        self.top_p = top_p if self.do_sample else None
        self.max_new_tokens = max_new_tokens
        self.max_tokens = max_tokens
        self.stop_words = stop_words
        if gpu_available:
            best_gpu, remains =  get_gpu_with_most_free_memory()
            self.device =   best_gpu
        else:
            self.device = "auto"
        print(f"set device: {self.device}")
        self.verbose = verbose
                    

        self._init_agent()
        self.conversation = []
        self.init_sys_dir = ""
        self.save_context = save_context

    def format_input_str(self, input_str, role="user"):
        return {"role": role, "content": input_str}
    
    def _assistant(self, input_str):
        return {"role": "assistant", "content": input_str}

    def _user(self, input_str):
        return {"role": "user", "content": input_str}

    def _system(self, input_str):
        return {"role": "system", "content": input_str}

    def _init_agent(self, ):
        
        self.assistant =  pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                torch_dtype=torch.float16 if gpu_available else torch.float32,
                use_auth_token=self.auth_token,
                # max_tokens=self.max_tokens,
                trust_remote_code=True
            )
        self.tokenizer = self.assistant.tokenizer

    def process_response(self, input_msg_dict: dict, response):
        response_text = response[0]["generated_text"][-1]["content"]
        if self.save_context:
            self.conversation += [input_msg_dict, self._assistant(response_text)]
        return response_text
    
    def generate_response(self, input_str: str, role="user"):
        input_msg_dict = self.format_input_str(input_str, role)
        response = self.assistant(
            self.conversation + [input_msg_dict], 
            max_new_tokens=self.max_new_tokens,
            # max_tokens = self.max_tokens,
            do_sample = self.do_sample,
            top_k = self.top_k,
            top_p = self.top_p,
            stop_words = self.stop_words,
            tokenizer = self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.process_response(input_msg_dict, response)


    def converse(self, system_directive: str="You are a helpful but sarcastic assistant"):
        if system_directive:
            self.init_sys_dir = self._system(system_directive)
            self.conversation += [self.init_sys_dir]
        while True:
            input_str = input(">> ").strip()
            if input_str.lower() in ["stop", "quit", "q"]:
                print("🛑 Recieved stop command, ending conversation...")
                break
            else:
                response = self.generate_response(input_str, role="user")
                print(f"🤖 {response}\n\n")
        
                        