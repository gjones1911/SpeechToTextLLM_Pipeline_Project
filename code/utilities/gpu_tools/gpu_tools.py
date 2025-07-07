"""
    Purpose: This file contains a few tools for optimizing AI processes on GPUs. These include:
             * 'inspect_memory_stats()': viewing GPU resources
             * 'get_gpu_mem_free()': checking available memory for each GPU
             * 'get_gpu_with_most_free_memory()': getting the # of the GPU with the most free memory, and how much
             * 'get_gpu_with_enough_memory()': select GPU with enough memory for a given memory load
             * 'estimate_tokens()': get estimate of how many tokens a current conversation contains
             * 'estimate_memory_usage()': Estimate the GPU memory usage for a given input.
             * 'manage_conversation()':  does a brute force context conversation FiFo truncation based on memory limits
             * 'get_model_hidden_size()': get context window size for an LLM
             * 'get_model_size_in_gib()': get size of a given model in Gib
"""

import torch
import os
import sys
dummy_tensor = torch.tensor([1.0]).to("cuda:0")
stats = torch.cuda.memory_stats(0)

# print(f"Available keys in torch.cuda.memory_stats(): stats{stats}\n\n\n")

############################
## GPU memory checking tools
############################
# used to check structure of torch.memory_stats object
def inspect_memory_stats(gpu_id=0):
    stats = torch.cuda.memory_stats(gpu_id)
    print(f"Available keys in torch.cuda.memory_stats(): stats{stats}\n\n\ngpu:{gpu_id}\n\n\n")
    for key in stats.keys():
        print(key)


def get_gpu_remaining_memory(gpu_id=0):
    
    print(f"thing: {gpu_id}")
    if isinstance(gpu_id, dict) and ":" in gpu_id[""]:
        device = int(gpu_id[""].split(":")[-1].strip())
    elif isinstance(gpu_id, int):
        device = gpu_id
    print(f"gpu int: {gpu_int}")
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available.")
    # Get total memory from device properties (in bytes)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    # Get memory already reserved by PyTorch
    reserved_memory = torch.cuda.memory_reserved(device)
    # Remaining memory is total minus reserved (a rough approximation)
    remaining_memory = total_memory - reserved_memory
    return remaining_memory


def get_gpu_mem_free(gpu_id):
    print(f"thing: {gpu_id}")
    if isinstance(gpu_id, dict) and ":" in gpu_id[""]:
        gpu_int = int(gpu_id[""].split(":")[-1].strip())
    elif isinstance(gpu_id, int):
        gpu_int = gpu_id
    print(f"gpu int: {gpu_int}")
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available.")
    stats = torch.cuda.memory_stats(gpu_id)

    # Use 'allocated_bytes.all.current' and 'reserved_bytes.all.current'
    allocated_mem = stats.get("allocated_bytes.all.current", 0)
    reserved_mem = stats.get("reserved_bytes.all.current", 0)
    
    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
    print(f"allocated_mem: {allocated_mem}")
    print(f"reserved_mem: {reserved_mem}")
    print(f"total_mem: {total_mem}")
    free_mem = total_mem - (allocated_mem + reserved_mem)
    # free_mem = total_mem - reserved_mem
    print(f"free_mem: {free_mem}")
    return free_mem

# used to determine the lowest load GPU
def get_gpu_with_most_free_memory():
    available_gpus = torch.cuda.device_count()
    free_memories = []
    
    for gpu_id in range(available_gpus):
        torch.cuda.empty_cache()  # Clear unallocated memory
        free_mem = get_gpu_mem_free(gpu_id)
        free_memories.append(free_mem)
    
    # Find the GPU with the most free memory
    max_free_mem = max(free_memories)
    best_gpu = free_memories.index(max_free_mem)
    
    return best_gpu, max_free_mem


# Function to find GPU with sufficient free memory
def get_gpu_with_enough_memory(required_memory):
    available_gpus = torch.cuda.device_count()
    for gpu_id in range(available_gpus):
        stats = torch.cuda.memory_stats(gpu_id)

        # Use 'allocated_bytes.all.current' and 'reserved_bytes.all.current'
        allocated_mem = stats.get("allocated_bytes.all.current", 0)
        reserved_mem = stats.get("reserved_bytes.all.current", 0)

        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
        free_mem = total_mem - (allocated_mem + reserved_mem)
        if free_mem >= required_memory:
            return gpu_id
    return None


############################
## Model memory tools
############################
def estimate_tokens(conversation, tokenizer):
    """
    Estimate the total number of tokens for a conversation.
    
    Args:
        conversation (list): List of conversation strings.
        tokenizer: Hugging Face tokenizer.
    
    Returns:
        int: Total token count for the conversation.
    """
    total_tokens = 0
    for message in conversation:
        tokens = tokenizer.encode(message, truncation=False)
        total_tokens += len(tokens)
    return total_tokens

def estimate_tokens(conversation, tokenizer):
    conversation_text = " ".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
    tokenized = tokenizer(conversation_text, return_tensors="pt")
    token_count = tokenized.input_ids.size(1)    
    return token_count

def estimate_memory_usage(token_count, model_hidden_size, dtype_size=4, intermediate_factor=2):
    """
    Estimate the GPU memory usage for a given input.
    
    Args:
        token_count (int): Total number of tokens.
        model_hidden_size (int): Hidden size of the model.
        dtype_size (int): Size of the data type in bytes (default: 4 for float32).
        intermediate_factor (int): Factor for intermediate activations (default: 2).
    
    Returns:
        float: Estimated memory usage in GiB.
    """
    memory_bytes = token_count * model_hidden_size * dtype_size * intermediate_factor
    return memory_bytes / (1024 ** 3)  # Convert to GiB

def get_conversation_memory_usage(conversation, tokenizer, model_hidden_size, dtype_size=4, intermediate_factor=2):
    total_tokens = estimate_tokens(conversation, tokenizer)
    # estimated_memory = estimate_memory_usage(total_tokens, model_hidden_size, dtype_size, intermediate_factor)
    estimated_memory = estimate_memory_requirement(total_tokens, model_hidden_size, dtype_size)
    return estimated_memory


def estimate_memory_requirement(token_count, hidden_size, dtype=torch.float32):
    # Calculate bytes per element (e.g. 4 for float32)
    bytes_per_param = torch.finfo(dtype).bits // 8
    print(f"Bytes: {bytes_per_param}")
    # Rough estimate: For each token, store a hidden vector of length hidden_size
    required_bytes = token_count * hidden_size * bytes_per_param
    return required_bytes

def manage_conversation(conversation, tokenizer, model_hidden_size, max_memory_gib, dtype_size=4, intermediate_factor=2):
    """
    Manage conversation to fit within GPU memory constraints.
    
    Args:
        conversation (list): List of conversation strings.
        tokenizer: Hugging Face tokenizer.
        model_hidden_size (int): Hidden size of the model.
        max_memory_gib (float): Maximum allowable memory in GiB.
        dtype_size (int): Size of the data type in bytes (default: 4 for float32).
        intermediate_factor (int): Factor for intermediate activations (default: 2).
    
    Returns:
        list: Truncated or summarized conversation.
    """
    total_tokens = estimate_tokens(conversation, tokenizer)
    estimated_memory = estimate_memory_usage(total_tokens, model_hidden_size, dtype_size, intermediate_factor)
    
    if estimated_memory > max_memory_gib:
        print(f"Estimated memory {estimated_memory:.2f} GiB exceeds {max_memory_gib:.2f} GiB. Truncating...")
        # Truncate oldest messages to fit within memory
        while estimated_memory > max_memory_gib and len(conversation) > 1:
            conversation.pop(0)  # Remove oldest message
            total_tokens = estimate_tokens(conversation, tokenizer)
            estimated_memory = estimate_memory_usage(total_tokens, model_hidden_size, dtype_size, intermediate_factor)
    
    return conversation

def get_model_hidden_size(model):
    model_hidden_size = model.config.hidden_size  # e.g., 768 for GPT-2
    return model_hidden_size

def get_model_size_in_gib(model):
    """
    Calculate the size of a Hugging Face model in GiB.
    
    Args:
        model: Hugging Face model object.
    
    Returns:
        Model size in GiB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    param_size = torch.tensor([]).element_size()  # Size of a single parameter in bytes (default: float32 = 4 bytes)
    total_size = total_params * param_size  # Total size in bytes
    size_in_gib = total_size / (1024**3)  # Convert bytes to GiB
    return size_in_gib
