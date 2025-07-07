"""
    amas_manager_tools.py:
    - This file contains the main classes and modules to run the amas apps including the singular and multi-persona versions. This can be used to create secondary child classes/modules for other purposes as needed, but the focus is the ML_ACT amas project
"""


############ Assistant Class: the base class for creating and interacting with an assistant

from assistant_gradio_variables import *
from knowledge_nexus_generator import *
from utilities.gpu_tools.gpu_tools import *

class AMAS_Assistant:
    """
        Base assistant type with little to no gradio/ui support
    """
    def __init__(self, model_path: str, load_method: str="pipeline", assistant_name: str="AMAS", 
                 device_map: str="most_free", torch_dtype: float=torch.bfloat16, 
                 use_remote: bool=True, 
                 hf_login: bool=True, hf_uname: str="gjonesQ02", creds_json: str="", 
                 temperature: float=None, do_sample: bool=False, top_k=None, top_p=None, 
                 max_new_tokens=None, max_tokens=None,
                 save_context: bool=True,  truncate_conversation: bool=False,
                 stop_strings: list=None, hidden_size=int,
                 
                 model_mode: str="converse",
                 
                 training_method: str= "peft",
                 output_dir: str="New_MODEL", 
                 hf_path: str="New_Model",
                 get_token: str=None,
                 push_token: str=None, 
                 train_dataset=None, 
                 eval_dataset=None,
                 push_to_hub=False,
                 max_seq_length=2000, epochs=2,  batch_size=20, eval_batch_size=8,
                 gradient_accumulation_steps=10,
                 gradient_checkpointing=True,
                 optim="adamw_torch_fused",
                 logging_steps=10,
                 save_strategy="epoch",
                 learning_rate=2e-4,
                 weight_decay=0.01,
                 bf16=True,
                 tf32=True,
                 max_grad_norm=0.3,
                 warmup_ratio=0.03,
                 lr_scheduler_type="constant",
                 report_to="tensorboard",

                 verbose=True,
                 **kwargs,
                ):
        self.model_path=model_path
        self.load_method=load_method
        self.assistant_name=assistant_name
        self.verbose=verbose

        #####################################
        self.torch_dtype=torch_dtype
        # self.device_map =  "balanced" if tf.config.list_physical_devices('GPU') else 'auto'
        self.device_map = "auto"
        self.device_map = device_map if device_map == "most_free" else self.device_map
        self.assigned_gpu = ""
        
        
        #####################################
        self.use_remote=use_remote
        self.hf_uname=hf_uname
        self.user_name=hf_uname 
        self.get_token=get_token
        self.push_token=push_token

        self.do_sample=do_sample
        self.temp=temperature
        self.top_k=top_k
        self.top_p=top_p
        self.top_k_og = top_k
        self.top_p_og = top_p
        self.max_new_tokens=max_new_tokens
        self.max_tokens=max_tokens
        self.stop_strings=stop_strings
        self.hidden_size = hidden_size
        self.kwargs = kwargs
        self.save_context = save_context
        self.truncate_conversation = truncate_conversation

        self.conversation = []

        if hf_login:
            self.user_name, self.get_token, self.push_token =  self.login_to_hf(json_file=creds_json)

        # load convo model
        if model_mode != "train":
            self.load_model(self.model_path, max_new_tokens=self.max_new_tokens, 
                            device_map=self.device_map, 
                            torch_dtype=self.torch_dtype, use_remote=self.use_remote)
        else:
            self.init_trainer(self, 
                training_method=training_method, 
                output_dir=output_dir, hf_path=hf_path, 
                push_token=self.push_token, 
                train_dataset=train_dataset, eval_dataset=eval_dataset,
                push_to_hub=push_to_hub,
                device_maps=self.device_map,
                max_seq_length=max_seq_length, 
                epochs=epochs,  batch_size=batch_size, 
                eval_batch_size=eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                optim=optim,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                bf16=bf16,
                tf32=tf32,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_scheduler_type,
                report_to=report_to)


        # set up the max_input_tokens using tokenizer
        try:
            self.max_input_tokens = get_max_tokens(self.tokenizer)
            print(f"Max input tokens set to: {self.max_input_tokens}")
        except Exception as ex:
            print(f"ex: {ex}")
            print("Exiting program...")
            sys.exit()

    #### Login and credential tools
    @staticmethod
    def login_hf_from_json(json_file: str, hf_uname: str=None) -> tuple:
        """
        Logs into Hugging Face using credentials from a JSON file.

        :param json_file: Path to the JSON file with credentials.
        :return: Tuple of username, get_token, and push_token.
        """
        with open(json_file, 'r') as file:
            creds = json.load(file)
        hf_uname = hf_uname if hf_uname else "gjonesQ02"
        user_name = creds.get('user_name', hf_uname)
        get_token = creds.get('Get')
        push_token = creds.get('Push')

        if not get_token or not push_token:
            raise ValueError("Credentials file must include 'Get' and 'Push' tokens.")
        login(token=get_token, add_to_git_credential=False, new_session=True)
        login(token=push_token, add_to_git_credential=False, new_session=True)
        return user_name, get_token, push_token
    
    def login_to_hf(self, json_file:str, **kwargs):
        return self.login_hf_from_json(json_file)

    
    ############ conversation input formatting tools
    @staticmethod
    def format_conversation(conversation_log: list[dict]) -> list:
        """
        Formats a set of user/assistant/system inputs. This is really just a place 
        holder and shoule be overwritten by a child class

        :param input_str: list of dictionary entries were each represents a response 
                          from a user, assistant, the system, or a tool
        :return: The original conversation unchanged, this is really a place holder 
                 for more complex operations in a more complex child class
        """
        return conversation_log

    @staticmethod
    def role_prepending_format_conversation(conversation_history) -> str:
        """
        Converts a structured conversation (list of dictionaries) into a formatted string
        suitable for text-generation models, including system prompts.
        
        :param conversation_history: List of dictionaries containing messages.
        :return: Formatted conversation as a string.
        """
    
        formatted_prompt = ""
        
        for turn in conversation_history:
            if turn["role"] == "system":
                # System message (typically appears once at the beginning)
                formatted_prompt += f"System: {turn['content']}\n\n"
            elif turn["role"] == "user":
                # User messages
                formatted_prompt += f"User: {turn['content']}\n"
            elif turn["role"] == "assistant":
                # Assistant responses
                formatted_prompt += f"Assistant: {turn['content']}\n"
        
        return formatted_prompt.strip()  # Remove extra newlines
    
    @staticmethod
    def assistant_input(assistant_input):
        """
        Formats an input string with the associated role for conversation.

        :param input_str: The content of the message.
        :return: A formatted dictionary representing the message.
        """
        return {'role':'assistant', 'content':assistant_input}

    @staticmethod
    def user(user_input, role='user'):
        return {'role':'user', 'content':user_input}

    @staticmethod
    def system(system_input, role='system'):
        return {'role':'system', 'content':system_input}

    @staticmethod
    def python_tool(input_str, role='tool'):
        return {'role':'tool', 'content':input_str}

    
    @staticmethod
    def validate_conversation_input(input_dict: dict):
        """
        Validates that the input dictionary has required keys 'role' and 'content'.

        :param input_dict: Dictionary to validate.
        :raises ValueError: If validation fails.
        """
        if not isinstance(input_dict, dict) or "role" not in input_dict or "content" not in input_dict:
            raise ValueError("Input must be a dictionary with keys 'role' and 'content'.")        

    ######################## model loading tools
    @staticmethod
    def load_autocausal(model_path, load_method, device_map='balanced',
                        torch_dtype=torch.bfloat16):
        if load_method == 'peft':
            model2 = AutoPeftModelForCausalLM.from_pretrained(
              model_path,
              device_map=device_map,
              torch_dtype=torch_dtype
            )
            return model2
        else:
            model2 = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype
            )
            return model2

    def load_model_basic(self, load_method:str="pipeline", **kwargs):
        """
        Loads a text-generation pipeline with a specified model. It can do either a peft or basic pipeline model.
        
        :param model_path: Path to the model.
        :param max_new_tokens: Maximum tokens to generate.
        :param device_map: Device map configuration (default: self.device_map).
        :param torch_dtype: type of torch_dtype to use (default: torch.bfloat16).
        """
        self.load_method = load_method
        if load_method == "pipeline":
            self.load_pipeline_basic(**kwargs)
        else:
            self.load_peft_pipeline(**kwargs)


    def load_pipeline_basic(self, model_path: str, max_new_tokens: int, device_map: str = None,
                            torch_dtype=torch.bfloat16, use_remote: bool = True):
        """
        Loads a text-generation pipeline, allowing the user to choose between remote or local loading.
    
        :param model_path: Path to the model (or Hugging Face model ID).
        :param max_new_tokens: Maximum tokens to generate.
        :param device_map: Device map configuration (default: self.device_map).
        :param torch_dtype: Type of torch_dtype to use (default: torch.bfloat16).
        :param use_remote: If True, attempts remote loading first. If False, loads locally.
        """
        
        # Load tokenizer separately to store it
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        self.torch_dtype=torch_dtype
        self.use_remote=use_remote
        # Fetch model config to get hidden_size
        try:
            config = AutoConfig.from_pretrained(model_path)
            self.hidden_size = getattr(config, "hidden_size", None)
        except Exception as ex:
            print(f"⚠ Warning: Failed to fetch model config for hidden_size. {ex}")
            self.hidden_size = None  # Default to None if unavailable
    
        print(f"Attempting to load model @: '{model_path}'")
        self.assistant = pipeline(
            "text-generation",
            model=model_path,
            device_map=self.device_map,  # Remote API does not support GPUs
            max_new_tokens=self.max_new_tokens,
            tokenizer=self.tokenizer,
            torch_dtype=torch_dtype,
        )
        print("Pipeline Callable Parameters:", inspect.signature(self.assistant.__call__).parameters.keys())
        print("Pipeline Model Configuration:", self.assistant.model.config)
        print(f"✅ Successfully loaded model @: '{model_path}'")
        # return  # Exit function after successful remote load
    
        # # Load model locally if use_remote=False or remote fails
        # print(f"Loading local version of '{model_path}'")
        
        # self.assistant = pipeline(
        #     "text-generation",
        #     model=model_path,
        #     tokenizer=self.tokenizer,
        #     device_map=device_map or self.device_map,
        #     max_new_tokens=max_new_tokens,
        #     torch_dtype=torch_dtype,
        # )
    
        # print(f"✅ Pipeline successfully loaded with model: {model_path}")
        if self.hidden_size:
            print(f"\n\n\t\t\tHidden size: {self.hidden_size}\n\n")
        return


    def load_peft_pipeline(self, model_path: str, max_new_tokens: int, device_map: str = None,
                      torch_dtype=torch.bfloat16, **kwargs,
                     ):
        """
        Loads a text-generation pipeline with the specified model.

        :param model_path: Path to the model.
        :param max_new_tokens: Maximum tokens to generate.
        :param device_map: Device map configuration (default: self.device_map).
        :param torch_dtype: type of torch_dtype to use (default: torch.bfloat16).
        """
        print(f"Loading '{self.load_method}' version of '{model_path}'")
        print(f"device_map: {device_map}")
        model = AutoPeftModelForCausalLM.from_pretrained(
              model_path,
              device_map=device_map,
              torch_dtype=torch_dtype
        )
        self.tokenizer = model.tokenizer
        self.assistant = pipeline(
            "text-generation",
            model=model,
            tokenizer=model.tokenizer,
            device_map=device_map or self.device_map,
            max_new_tokens=max_new_tokens,
            torch_dtype=torch_dtype,
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"
        print(f"Pipeline loaded with model: {model_path}")
    

    def load_model(self, model_path, max_new_tokens: int=None, device_map: str=None, torch_dtype=torch.bfloat16, use_remote: bool=False):
        print(f"Second load model: {device_map}")
        if device_map:
            if device_map == "most_free":
                best_gpu, max_mem = get_gpu_with_most_free_memory()
                print(f"\n\n\n\n\t\t\tAssigning model to GPU {best_gpu} with {max_mem / (1024**3):.2f} GB free memory.\n\n\n\n")
        
                # Define device_map to use the selected GPU
                device_map = {"": f"cuda:{best_gpu}"}  # Assign the entire model to the selected GPU
                self.device_map=device_map
                self.assigned_gpu = device_map
        
        self.load_model_basic(self.load_method, model_path=model_path,
                              max_new_tokens=max_new_tokens if max_new_tokens else self.max_new_tokens,
                              device_map=device_map,
                              torch_dtype=torch_dtype, use_remote=use_remote,
                             )
        # set up the max_input_tokens using tokenizer
        try:
            self.max_input_tokens = get_max_tokens(self.tokenizer)
            print(f"Max input tokens set to: {self.max_input_tokens}")
        except Exception as ex:
            print(f"ex: {ex}")
            print("Exiting program...")
            sys.exit()

    

    ##### Model Fine-tuning tools
    ###################################################
    ##           Training
    ###################################################
    def init_PT_Model(self, base_model="meta-llama/Meta-Llama-3.1-8B-Instruct", device_maps="balanced", 
                      load_mode='base', load_quantize_4bit=True,
                      **kwargs):
        """
        Initializes and configures a model for fine-tuning with optimization settings.

        :param base_model: The name or path of the base model to load.
        :param device_maps: Device map configuration for model loading.
        :param kwargs: Additional arguments for model initialization.
        :return: Tuple containing the model and tokenizer.
        """
        if load_quantize_4bit:
            print("Loading 4-bit")
            # BitsAndBytesConfig for int-4 config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            print("Loading 8-bit")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        

        if load_mode == 'base':
            # Load model and tokenizer
            print(device_maps)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=device_maps,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config
            )
        else: # if not base model will assume peft model
            model = AutoPeftModelForCausalLM.from_pretrained(
              base_model,
              device_map=device_maps,
              torch_dtype=torch.bfloat16,
              quantization_config=bnb_config,
            )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.padding_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        model, tokenizer = setup_chat_format(model, tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        return self.model, self.tokenizer    

    def generate_training_args(self, output_dir, hf_path, 
                            push_token, 
                            train_dataset, eval_dataset=None,
                            push_to_hub=False,
                            device_maps="balanced",
                            max_seq_length=2000, epochs=2,  batch_size=20, eval_batch_size=20,
                            gradient_accumulation_steps=10,
                            gradient_checkpointing=True,
                            optim="adamw_torch_fused",
                            logging_steps=10,
                            save_strategy="epoch",
                            learning_rate=2e-4,
                            bf16=True,
                            tf32=True,
                            max_grad_norm=0.3,
                            warmup_ratio=0.03,
                            lr_scheduler_type="constant",
                            report_to="tensorboard", ):

        if eval_dataset:
            # do the train and eval setup
            training_args = self.setup_training_eval_args(
                output_dir=output_dir,
                hub_model_id=hf_path,
                hub_token=push_token,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                optim=optim,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                learning_rate=learning_rate,
                weight_decay=weight_decay,               # Weight decay for optimization
                bf16=bf16,
                tf32=tf32,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_scheduler_type,
                push_to_hub=push_to_hub,
                report_to=report_to,
            )
        else:
            
            # do the train only setup
            training_args = self.setup_training_args(
                output_dir=output_dir,
                hub_model_id=hf_path,
                hub_token=push_token,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                optim=optim,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                learning_rate=learning_rate,
                weight_decay=weight_decay,               # Weight decay for optimization
                bf16=bf16,
                tf32=tf32,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_scheduler_type,
                push_to_hub=push_to_hub,
                report_to=report_to,
            )
        return training_args
    

    @staticmethod    
    def setup_training_eval_args(output_dir, hf_path, push_token, 
                                push_to_hub=False,
                                device_maps="balanced",
                                max_seq_length=2000, epochs=2,  batch_size=20, eval_batch_size=20,
                                gradient_accumulation_steps=10,
                                gradient_checkpointing=True,
                                optim="adamw_torch_fused",
                                logging_steps=10,
                                save_strategy="epoch",
                                learning_rate=2e-4,
                                weight_decay=0.01,
                                bf16=True,
                                tf32=True,
                                max_grad_norm=0.3,
                                warmup_ratio=0.03,
                                lr_scheduler_type="constant",
                                report_to="tensorboard", ):
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            hub_model_id=hf_path,
            hub_token=push_token,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            optim=optim,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            learning_rate=learning_rate,
            weight_decay=weight_decay,               # Weight decay for optimization
            bf16=bf16,
            tf32=tf32,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            push_to_hub=push_to_hub,
            report_to=report_to,
        )
        return training_args


    
    @staticmethod    
    def setup_training_args(output_dir, hf_path, push_token, 
                                push_to_hub=False,
                                device_maps="balanced",
                                max_seq_length=2000, epochs=2,  batch_size=20, # eval_batch_size=8,
                                gradient_accumulation_steps=10,
                                gradient_checkpointing=True,
                                optim="adamw_torch_fused",
                                logging_steps=10,
                                save_strategy="epoch",
                                learning_rate=2e-4,
                                weight_decay=0.01,
                                bf16=True,
                                tf32=True,
                                max_grad_norm=0.3,
                                warmup_ratio=0.03,
                                lr_scheduler_type="constant",
                                report_to="tensorboard", ):
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            hub_model_id=hf_path,
            hub_token=push_token,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            optim=optim,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            learning_rate=learning_rate,
            weight_decay=weight_decay,               # Weight decay for optimization
            bf16=bf16,
            tf32=tf32,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            push_to_hub=push_to_hub,
            report_to=report_to,
        )
        return training_args

    
    def init_basic_trainer(self, model, tokenizer, 
                            output_dir, hf_path, push_token, 
                            train_dataset, eval_dataset=None,
                            push_to_hub=False,
                            device_maps="balanced",
                            max_seq_length=2000, epochs=2,  batch_size=20, eval_batch_size=8,
                            gradient_accumulation_steps=10,
                            gradient_checkpointing=True,
                            optim="adamw_torch_fused",
                            logging_steps=10,
                            save_strategy="epoch",
                            learning_rate=2e-4,
                            weight_decay=0.01,
                            bf16=True,
                            tf32=True,
                            max_grad_norm=0.3,
                            warmup_ratio=0.03,
                            lr_scheduler_type="constant",
                            report_to="tensorboard", ):
        # set up training arguments
        training_args = self.generate_training_args(
                output_dir=output_dir,
                hub_model_id=hf_path,
                hub_token=push_token,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                optim=optim,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                learning_rate=learning_rate,
                weight_decay=weight_decay,               # Weight decay for optimization
                bf16=bf16,
                tf32=tf32,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_scheduler_type,
                push_to_hub=push_to_hub,
                report_to=report_to,
        )
        
        
        if eval_dataset:
            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
            )
        else:
            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
            )
        return self.trainer

    def init_peft_trainer(self, 
                          output_dir, hf_path, push_token, 
                          dataset, 
                          max_seq_length=2000, epochs=2, push_to_hub=False, 
                          batch_size=20, device_maps="balanced"):
        """
        Initializes the trainer for model fine-tuning.

        :param output_dir: Directory to save the model.
        :param hf_path: Hugging Face path for saving the model.
        :param push_token: Token for pushing to Hugging Face.
        :param dataset: Dataset to use for training.
        :param max_seq_length: Maximum sequence length for the model.
        :param epochs: Number of training epochs.
        :param push_to_hub: Boolean indicating whether to push the model to Hugging Face.
        :return: The initialized trainer.
        """
        # LoRA config based on QLoRA paper & Sebastian Raschka experiment
        peft_config = LoraConfig(
                lora_alpha=128,     # scaling factor
                # lora_alpha=64,     # scaling factor
                lora_dropout=0.05,  # dropout rate during training
                # r=256,              # Indicated Rank of the low-rank matrices, determines the size of the added weight matrices
                r=128,              # Indicated Rank of the low-rank matrices, determines the size of the added weight matrices
                bias="none",
                target_modules="all-linear",   # which networks to target
                # target_modules=["q_proj", "v_proj"],   # which networks to target
                task_type="CAUSAL_LM",         # the task the LLM will perform
        )

        
        args = TrainingArguments(
            output_dir=output_dir,
            hub_model_id=hf_path,
            hub_token=push_token,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=10,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            push_to_hub=push_to_hub,
            report_to="tensorboard",
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            peft_config=peft_config,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            packing=True,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            }
        )
        return self.trainer

    def train(self):
        """
        Starts the training process with the initialized trainer.
        """
        self.trainer.train()

    def init_trainer(self, training_method, 
            output_dir, hf_path, push_token, 
            train_dataset, eval_dataset=None,
            push_to_hub=False,
            device_maps="balanced",
            max_seq_length=2000, epochs=2,  batch_size=20, eval_batch_size=8,
            gradient_accumulation_steps=10,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            weight_decay=0.01,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            report_to="tensorboard",):

        # create trainer
        if training_method == "peft":
            self.init_peft_trainer(
                output_dir=output_dir, hf_path=hf_path, push_token=push_token, 
                dataset=train_dataset, 
                max_seq_length=max_seq_length, epochs=epochs, push_to_hub=push_to_hub, 
                batch_size=batch_size, device_maps=device_maps,
            )
        else:
            self.init_basic_trainer(
                model=self.model, tokenizer=self.tokenizer,
                output_dir=output_dir,
                hub_model_id=hf_path,
                hub_token=push_token,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                optim=optim,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                learning_rate=learning_rate,
                weight_decay=weight_decay,               # Weight decay for optimization
                bf16=bf16,
                tf32=tf32,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_scheduler_type,
                push_to_hub=push_to_hub,
                report_to=report_to,
            )    
        return
        
    def init_trainer_and_train(self, training_method, 
                                output_dir, hf_path, push_token, 
                                train_dataset, eval_dataset=None,
                                push_to_hub=False,
                                device_maps="balanced",
                                max_seq_length=2000, epochs=2,  batch_size=20, eval_batch_size=8,
                                gradient_accumulation_steps=10,
                                gradient_checkpointing=True,
                                optim="adamw_torch_fused",
                                logging_steps=10,
                                save_strategy="epoch",
                                learning_rate=2e-4,
                                weight_decay=0.01,
                                bf16=True,
                                tf32=True,
                                max_grad_norm=0.3,
                                warmup_ratio=0.03,
                                lr_scheduler_type="constant",
                                report_to="tensorboard",
                              ):

        # create trainer
        if training_method == "peft":
            self.init_peft_trainer(
                output_dir=output_dir, hf_path=hf_path, push_token=push_token, 
                dataset=train_dataset, 
                max_seq_length=max_seq_length, epochs=epochs, push_to_hub=push_to_hub, 
                batch_size=batch_size, device_maps=device_maps,
            )
        else:
            self.init_basic_trainer(
                model=self.model, tokenizer=self.tokenizer,
                output_dir=output_dir,
                hub_model_id=hf_path,
                hub_token=push_token,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                optim=optim,
                logging_steps=logging_steps,
                save_strategy=save_strategy,
                learning_rate=learning_rate,
                weight_decay=weight_decay,               # Weight decay for optimization
                bf16=bf16,
                tf32=tf32,
                max_grad_norm=max_grad_norm,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_scheduler_type,
                push_to_hub=push_to_hub,
                report_to=report_to,
            )

        self.train()
        return


    
    ######### Generation parameter setting tools
    def update_temperature(self, new_temp):
        self.temp  = new_temp if new_temp > 0 else None
        self.top_k = self.top_k if self.temp else None
        self.top_p = self.top_p if self.temp else None
        self.do_sample  = True if self.temp else False
        print(f"do sample: {self.do_sample}")
        return
    
    def update_top_k(self, new_top_k):
        new_top_k = int(new_top_k)
        if new_top_k <= 0:
            self.top_k=None
            self.top_p=None
        else:
            self.top_k=new_top_k
    
    def update_top_p(self, new_top_p):
        if new_top_p <= 0:
            self.top_k=None
            self.top_p=None
        else:
            self.top_p=new_top_p
    
    def update_max_new_tokens(self, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        self.max_tokens = max_new_tokens
    
    def update_generation_config(self, **kwargs):
        """
        Updates generation configuration parameters.

        :param kwargs: Dictionary of generation parameters to update.
        """
        for key, value in kwargs.items():
            if key in self.generation_config:
                if key == "temp":
                    self.update_temperature(value)
                elif key == "top_k":
                    self.update_top_k(value)
                elif key == "top_p":
                    self.update_top_p(value)
                elif key == "max_new_tokens":
                    self.update_max_new_tokens(value)

    #### MOdel interaction tools
    def generate_response(self, conversation: list, to_strip=None):
        """
        Generates a response based on the conversation history.

        :param conversation: List of conversation messages.
        :param to_strip: Characters to strip from the response.
        :return: The generated response (str).
        """
        # empty the cache to clean up the gpu
        try:
            torch.cuda.empty_cache()
        except Exception as ex:
            print(ex)
        # Define full argument set (for local inference)
        kwargs1 = {
            "temperature": self.temp,
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_strings": self.stop_strings,  # ❌ Not valid for remote
            "tokenizer": self.assistant.tokenizer,  # ❌ Not valid for remote
        }
        kwargs2 = {
            "temperature": self.temp,
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            # "top_k": self.top_k,
            # "top_p": self.top_p,
            # "stop_strings": self.stop_strings,  # ❌ Not valid for remote
            # "tokenizer": self.assistant.tokenizer,  # ❌ Not valid for remote
        }
        # If using remote, filter out unsupported arguments
        if not self.use_remote:
            if self.verbose:
                print("using local model for inference")
            outputs = self.assistant(conversation, 
                                 temperature=self.temp, 
                                 do_sample=self.do_sample,
                                 max_new_tokens=self.max_new_tokens, 
                                 top_k=self.top_k, top_p=self.top_p,
                                 stop_strings=self.stop_strings,
                                 tokenizer=self.assistant.tokenizer)
        else:
            if self.verbose:
                print("🔹 Using remote model inference, filtering invalid arguments...")
    
            # Get valid arguments for the pipeline's __call__ function
            valid_params = inspect.signature(self.assistant.__call__).parameters.keys()
    
            # Keep only arguments that are valid
            kwargs = {k: v for k, v in kwargs1.items() if k in valid_params}
            
            formatted_input = self.format_conversation(conversation)
            outputs = self.assistant(formatted_input)
                                     # **kwargs2)
        return self.process_response(outputs, to_strip)

    def process_response(self, outputs, to_strip=None):
        """
        Processes the assistant's raw response by stripping unwanted characters.

        :param outputs: Raw output from the assistant.
        :param to_strip: Characters to strip from the response.
        :return: Processed response string.
        """
        to_strip = to_strip or []
        raw_response = outputs[0]["generated_text"][-1]["content"]
        for char in to_strip:
            raw_response = raw_response.strip(char)
        return raw_response


    @staticmethod
    def check_end(input_str, check_list=['q', 'quit', 'stop', 'end']):
        """Checks for any thing in given check list and returns bool based on presence"""
        return input_str in check_list

    @staticmethod
    def check_system_prompt(user_input, system_flag='system:'):
        return system_flag in user_input.lower()
        

    def system_directive_process(self, conversation, user_input, to_strip=[""]):
        conversation += [self.system(user_input)]
        assistant_response = self.generate_response(conversation, to_strip=[""])
        return conversation, assistant_response

    def manage_memory_and_context(self, conversation, user_input):
        if self.save_context and self.truncate_conversation:
            at_max, new_tokens_possible = check_max_tokens(messages=conversation, max_input_tokens=self.max_input_tokens, 
                                             tokenizer=self.tokenizer, user_input=user_input)
            print(f"new_tokens_possible: {new_tokens_possible}")
            estimated_memory_required = get_conversation_memory_usage(
                                            conversation, self.tokenizer, 
                                            self.hidden_size, dtype_size=self.torch_dtype, 
                                            intermediate_factor=2)
            print(f"estimated memory: {estimated_memory_required}")
            if at_max:
                print("\n\n\n\n\t\tabout to overflow!!!!\n\n\n\n")
                summary_string = summarize_conversation(self.assistant, self.tokenizer, messages=conversation, 
                                       stop_strings=self.stop_strings, max_new_tokens=300)
                print(f"Summary of conversation to this point:\n\n{summary_string}\n\n")
                conversation = replace_context_with_summary(summary_string)
            else:
                print(f"\n\n\t\tWe still good--1: {new_tokens_possible} vs. {self.max_input_tokens}\n\n")
    
    
    def response_generation_process(self, conversation, user_input, 
                                   input_role="user"):
        # get and return response from assistant
        # print(f"\n\n\n\t\tSave content 2: {save_context}\n\n\n\n")
        # check for a potential context overflow
        print("\n\n\t\t\tinside: response_generation_process\n------------------------\n\n")
        self.manage_memory_and_context(conversation, user_input)

        
        if input_role.lower() == "system":
            print(f"system input!")
            user_prompt = [self.system(user_input)]
        else:
            user_prompt = [self.user(user_input)]
            
        assistant_response = self.generate_response(conversation + user_prompt, to_strip=[""])
        if self.save_context:
            print("updating conversation")
            conversation += user_prompt + [self.assistant_input(assistant_response)]
        return conversation, assistant_response    

    def assistant_interaction(self, conversation, uname='user'):
        """User input driven method to prompt the set LLM with control logic for quiting, and system prompting"""
        # prompt user for input
        user_input = input(f"{uname}:> ")
        
        # if user wants to end conversation...
        if self.check_end(user_input):
            return None

        # if user wants to give a system directive
        elif self.check_system_prompt(user_input):
            print("\t\t\t\tSystem input given:")
            return self.system_directive_process(conversation, user_input)
        
        # general conversation
        else:
            # get and return response from assistant
            return self.response_generation_process(conversation, user_input)

    def init_conversation(self, system_directive="You are a helpful AI assistant that will help the user with any task they desire"):
        
        if isinstance(system_directive, dict):
            keys_present = 'role' in system_directive and 'content' in system_directive
            if keys_present:
                self.init_system_directive = system_directive
        elif isinstance(system_directive, str):
            self.init_system_directive = self.system(system_directive)
        
        self.conversation = list([self.init_system_directive])


    def base_converse(self, system_directive, uname='user'):
        # self.init_system_directive = self.system(system_directive)
        # self.conversation = list([self.init_system_directive])
        self.init_conversation(system_directive)
        print(f"conversation: {self.conversation}")
        while True:
            # prompt user for input, process and return response from assistant
            # if user has not entered end conversation command
            result = self.assistant_interaction(self.conversation, uname)
            if not result:
                break
            else:
                self.conversation = result[0]
                assistant_response = result[1]
                print(f'{self.assistant_name}:\n{assistant_response}\n')
        return self.conversation



class AMAS_RAG_Assistant(AMAS_Assistant):
    def __init__(
         self, model_path: str, load_method: str="pipeline", assistant_name: str="AMAS", 
         device_map: str="most_free", torch_dtype: float=torch.bfloat16, 
         use_remote: bool=True, 
         hf_login: bool=True, hf_uname: str="gjonesQ02", creds_json: str="", 
         temperature: float=None, do_sample: bool=False, top_k=None, top_p=None, 
         max_new_tokens=None, max_tokens=None,
         save_context: bool=True,  truncate_conversation: bool=False,
         stop_strings: list=None, hidden_size=int,
         nexus_path: str=None, # optional inital knowledge domain to load into manager
         k: int=3, min_score: float=200.0,
         verbose: bool=True,
         **kwargs,
        
    ):
        super().__init__(
            model_path=model_path, load_method=load_method, 
            assistant_name=assistant_name, 
            device_map=device_map, torch_dtype=torch_dtype, 
            use_remote=use_remote, 
            hf_login=hf_login, hf_uname=hf_uname, creds_json=creds_json, 
            temperature=temperature, do_sample=do_sample, top_k=top_k, top_p=top_p, 
            max_new_tokens=max_new_tokens, max_tokens=max_tokens,
            save_context=save_context,  truncate_conversation=truncate_conversation,
            stop_strings=stop_strings, hidden_size=hidden_size,
            model_mode="converse",
            verbose=verbose,
        )
    
        self.knexus_path = nexus_path
        self.knexus_mngr = None
        self.load_knexus_manager(nexus_path)
        self.rag_mode = False
        self.rag_knowledge=""
        self.k=k
        self.min_score=min_score
        print(f"my assistant: {self.assistant}")

    def load_knexus_manager(self, nexus_path):
        self.knexus_path = nexus_path
        self.knexus_mngr = KnowledgeNexusManagerNX(nexus_path)

    def generate_rag_string(self, docs, seperator="\n--------------\n\n"):
        rag_strings = list()
        for doc in docs:
            rag_strings.append(doc.page_content)
        return seperator.join(rag_strings)
    
    def query_knexus(self, query):

        # query your knexus and return a list of documents and corresponding
        # simularity scores
        docs, scores = self.knexus_mngr.query_similarity_search( 
                                query=query, 
                                k=self.k, 
                                min_score=self.min_score, 
                                reverse=False, 
                                verbose=True,
                               )
        rag_string = self.generate_rag_string(docs)
        return rag_string

    def generate_rag_response(self, user_input):
        rag_string = ""
        if self.rag_mode:
            print("\n\nFull RAG mode!!\n\n")
            rag_string = self.query_knexus(user_input)
            rag_prompt="The above is a user input. Please utilize the below documents if they are relevant to the users needs, "\
                       +"otherwise ignore them and user your own general knowledge to respond to the user."
            user_input = "{user_input}\n\n{rag_prompt}\n\nRelated Documents:\n{rag_string}".format(
                user_input=user_input,
                rag_prompt=rag_prompt, rag_string=rag_string)
        else:
            print("\n\nNon-RAG mode!!\n\n")
        self.conversation, assistant_response = self.response_generation_process(self.conversation, user_input, 
                               input_role="user")
        # uname="User"
        # assistant_name="Chatbot"
        # chat_history1.append(user(f"## {uname}\n" + user_input))
        # chat_history1.append(assistant(f"## {assistant_name}:\n" + assistant_response))
        return assistant_response
        
