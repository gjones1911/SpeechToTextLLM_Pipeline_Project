"""
    Purpose: This defines methods and classes to generate gradio based UI's for the AMAs project. 
             The goal is to define all required gradio functions and tools here and pass the 
             parameters of are a pre-created version of some AMA assistant. In this way, the tools 
             and functinality of the Gradio UIs is mapped to these classes and tools allowing the 
             AMAs classes to focus only on those aspects and functions required of the assistant 
             such as Mmdel training/loading, model interaction, and rag based generation. 
"""
import os
import tempfile
import getpass

# 1. Get the current user's home directory
home_dir = os.path.expanduser("~")
print(f"My home directory: {home_dir}")

# 2. Compose the default user cache path
user_gradio_cache = os.path.join(home_dir, "gradio_cache")

# 3. Also allow a local fallback near the script
script_path = os.path.dirname(os.path.abspath(__file__))
local_gradio_tmp = os.path.join(script_path, ".gradio_tmp")

# 4. List of fallback paths
gradio_paths = [user_gradio_cache, local_gradio_tmp]

try_count = 0
while try_count < len(gradio_paths):
    print(f"\t\t--->Try count: {try_count}")
    try:
        # Set GRADIO_CACHE to the candidate path
        os.environ["GRADIO_CACHE"] = gradio_paths[try_count]
        if not os.path.exists(os.environ["GRADIO_CACHE"]):
            print(f"Creating directory: {os.environ['GRADIO_CACHE']}")
            os.makedirs(os.environ["GRADIO_CACHE"])

        # Also set TMPDIR to the same for good measure
        os.environ["TMPDIR"] = gradio_paths[try_count]

        # Monkey-patch tempfile.gettempdir to match
        tempfile.gettempdir = lambda: gradio_paths[try_count]

        # Import gradio and override cache for images
        import gradio as gr
        gr.components.image.Image.GRADIO_CACHE = os.environ["GRADIO_CACHE"]
        print("GRADIO_CACHE for gr.Image is:", gr.components.image.Image.GRADIO_CACHE)
        
        # Success: break the loop
        break

    except Exception as ex:
        print(f"Exception: {ex}")
        print(f"Try count: {try_count}")
        try_count += 1

# import tempfile
# # Override tempfile.gettempdir to return our custom directory
# try:
#     tempfile.gettempdir = lambda: "/home/gerald/gradio_cache"
# except Exception as ex:
#     print(f"ex: {ex}")
    
import gradio as gr
try:
    gr.components.image.Image.GRADIO_CACHE = os.environ["GRADIO_CACHE"]
    print("GRADIO_CACHE for gr.Image is:", gr.components.image.Image.GRADIO_CACHE)
except Exception as ex:
    print(f"ex: {ex}")
    
from functools import partial
from gradio import ChatMessage
import signal
import joblib
import gc

import pymupdf

from assistant_gradio_variables import *
from amas_manager_tools import AMAS_RAG_Assistant
from voice_processing import *


from utilities.gpu_tools.gpu_tools import get_conversation_memory_usage, estimate_memory_requirement, estimate_tokens, estimate_memory_usage, get_gpu_mem_free
import matplotlib.pyplot as plt
import numpy as np
import torch
import tempfile
import json
from pdf2image import convert_from_path

LLM_LIST = [
    f"{home_dir}/shared_space/models/MODELS/meta_llama_Llama_3p2_1B_Instruct",
    f"{home_dir}/shared_space/models/MODELS/Llama_3p1_8B_Instruct/",
    f"{home_dir}/shared_space/models/MODELS/Llama_3p2_3B_Instruct/",
]

# JavaScript for Auto-Scroll
js_code = """
function autoScroll() {
    let chatbox = document.querySelector('.chatbot');
    if (chatbox) {
        chatbox.scrollTop = chatbox.scrollHeight;
    }
}
"""

custom_css_og = """
.markdown-text {
    font-size: 24px;  /* Adjust font size */
    font-family: 'Arial', sans-serif;  /* Change font style */
    font-weight: bold;  /* Make text bold */
    color: #2c3e50;  /* Custom color */
}

.my-file {
    width: 20px !important;
}
.my-button-a {
    width: 50px !important;
}

.chatbot {
    resize: vertical;
    overflow-y: auto;
    min-height: 150px;
    max-height: 600px;
}

submit-button button {
    background-color: #28a745 !important;  /* Green color */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
}

/* Ensure hover effect works */
.submit-button button:hover {
    background-color: #218838 !important;  /* Slightly darker green */
}

/* Fix for Gradio sometimes wrapping buttons in divs */
.submit-button {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

#component-10 {
    background-color: #28a745 !important;  /* Green color */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
}

/* Ensure hover effect works */
#component-10:hover {
    background-color: #218838 !important;  /* Slightly darker green */
}

button.submit-button {
    background-color: #28a745 !important;  /* Green color */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
}

/* Ensure hover effect works */
button.submit-button:hover {
    background-color: #218838 !important;  /* Slightly darker green */
}

/* Force override Gradio's default styling */
button.submit-button.svelte-1ixn6qd {
    background-color: #28a745 !important;  /* Green color */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
}

/* Ensure hover effect works */
button.submit-button.svelte-1ixn6qd:hover {
    background-color: #218838 !important;  /* Slightly darker green */
}

#submit-btn {
    background-color: #28a745 !important;  /* Green color */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
}

/* Ensure hover effect works */
#submit-btn:hover {
    background-color: #218838 !important;  /* Slightly darker green */
}

.gr-row {
    justify-content: center !important;
}


#input-row {
    display: flex !important;
    align-items: center !important;  /* Centers items in the row */
}
"""

ttt= """.gr-chatbot-message .gr-chatbot-avatar > img  {
    width: 80px !important;
    height: 80px !important;
    border-radius: 50%; /* optional: makes it circular */
}"""
custom_css = """


.markdown-text {
    font-size: 24px;  /* Adjust font size */
    font-family: 'Arial', sans-serif;  /* Change font style */
    font-weight: bold;  /* Make text bold */
    color: #2c3e50;  /* Custom color */
}

.my-file {
    width: 20px !important;
}
.my-button-a {
    width: 50px !important;
}

.chatbot {
    resize: vertical;
    overflow-y: auto;
    min-height: 150px;
    max-height: 600px;
}


#submit-btn {
    background-color: #28a745 !important;  /* Green color */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 2px 10px !important;            /* Smaller padding */
    font-size: 16px !important;             /* Smaller text */
    min-width: 0 !important;                /* Prevents wide defaults */
    width: auto !important;                 /* Avoids stretch */
    height: auto !important;                /* Keeps it tidy */
}

/* Ensure hover effect works */
#submit-btn:hover {
    background-color: #218838 !important;  /* Slightly darker green */
}


#btn-small-1a, #btn-small-2a {
    background-color: #008ccc !important;  /* blue color */
    color: white !important;
    border: none !important;
    font-size: 14px !important;
    padding: 4px 10px !important;
    min-width: 0 !important;
    width: 100% !important;  /* Fill the row, but the row is now narrow */
    box-sizing: border-box;
}

#btn-small-1, #btn-small-2 {
    background-color: #008ccc !important;  /* blue color */
    color: white !important;
    border: none !important;
    font-size: 12px !important;
    padding: 4px 5px !important;
    min-width: 0 !important;
    width: 300px !important;  /* Fill the row, but the row is now narrow */
    box-sizing: border-box;
}


/* Ensure hover effect works */
#btn-small-1, #btn-small-2:hover {
    background-color: #0D84C9 !important;  /* Slightly darker blue*/
}

#function-btn {
    background-color: #008ccc !important;  /* blue color */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 2px 10px !important;            /* Smaller padding */
    font-size: 16px !important;             /* Smaller text */
    min-width: 0 !important;                /* Prevents wide defaults */
    width: auto !important;                 /* Avoids stretch */
    height: auto !important;                /* Keeps it tidy */
}

/* Ensure hover effect works */
#function-btn:hover {
    background-color: #0D84C9 !important;  /* Slightly darker green */
}

.gr-row {
    justify-content: center !important;
}


#input-row {
    display: flex !important;
    align-items: center !important;  /* Centers items in the row */
}

#resizable-chatbot {
    resize: both !important;
    overflow: auto !important;
    min-height: 200px;
    min-width: 200px;
    width: 100%;               /* Default width */
    max-width: none !important;
    box-sizing: border-box;
    display: block !important;
}

#resizable-accordion-content {
    resize: vertical !important;       /* Make it resizable by dragging */
    overflow: auto !important;         /* Enable scrollbars if needed */
    min-height: 200px;
    max-height: 1000px;
    height: 300px;                     /* Starting height */
    width: 100%;
    display: block;
    box-sizing: border-box;
}

#accordion-chatbot {
    height: 100%;
    width: 100%;
}


#custom-textbox button {
    background-color: #28a745 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    padding: 2px 10px !important;
    font-size: 16px !important;
    min-width: 0 !important;
    width: auto !important;
    height: auto !important;
}

#custom-textbox button:hover {
    background-color: #218838 !important;
}

/* Assistant (bot) message bubble */
.message.bot {
    background-color: #e0f7fa !important;  /* Light cyan background */
    color: #004d40 !important;            /* Dark green text */
    border-radius: 12px !important;
}

"""

conversational_models = [
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.1-8B-Instruct',
    "nvidia/Nemotron-Mini-4B-Instruct",
    'nvidia/Mistral-NeMo-Minitron-8B-Instruct',
]
##########################################################
###   Chat formating tools
##########################################################
def chat_format(input_string: str=None, role="assistant"):
    return {"role": role, 'content': input_string}

def assistant(input_string: str=None):
    return chat_format(input_string, role="assistant")

def user(input_string: str=None):
    return chat_format(input_string, role="user")

def system(input_string: str=None):
    return chat_format(input_string, role="system")





##########################################################
###   Base Gradio Assistant UI
##########################################################
class BasicGradioAssistant:
    def __init__(self, assistant_bot=None, 
                 uname="user",
                 **kwargs):
        self.assistant_bot = assistant_bot
        self.assistant_app = None
        
        
        self.mode=None

        # user UI basics
        self.input_text=None     # user input for chat 
        self.submit_button=None  # submit button
        self.clear_button=None   # button to clear input field
        self.reset_button=None   # button to clear conversation

        self.input_text2=None     # user input for chat 
        self.submit_button2=None  # submit button
        self.clear_button2=None   # button to clear input field
        self.reset_button2=None   # button to clear conversation
        

        # chatbox(assistant response) UI
        self.assistant_chatbot = None
        
        
        self.assistant_chatbot2 = None
        self.rag_chatbot = None
        
        self.conversation = None

        self.log_process_active = True
        self.logging_path=""

        
        # settings
        self.temp_slider=None
        self.tp_slider=None
        self.tk_slider=None
        self.max_new_token_slider=None
        self.system_dir_text=None
        self.system_dir_button=None
        self.option_dropdown=None
        self.copy_button=None
        self.dropdown_1=None
        self.dropdown_button1=None
        self.dropdown_2=None
        self.dropdown_button2=None
        self.dropdown_3=None
        self.dropdown_button3=None
        self.dropdown_4=None
        self.dropdown_button4=None
        self.init_system_directive=None
        self.directives_given = 0

        # user defined app instance generation tools
        self.rag_knowledge = ""
        self.rag_premable=base_rag_format_string
        
        

        self.uname=uname

        self.k=2
        self.min_score=.0001
        self.reverse=False
        self.mode="min"
        
        self.save_context=False
        self.truncate_conversation=False
        
        self.server = None
        self.verbose=False
        
        # init object with a method to properly closs app upon quiting
        self.set_sig_handler()

    def load_assistant_bot(self, ):
        pass

    
    def set_sig_handler(self, ):
        # set up signal handelers to ensure we close the app correctly and don't block a port
        try:
            # interrupt signal (Ctrl+C)
            signal.signal(signal.SIGINT,  
                          lambda sig, frame: self.signal_handler(sig, frame, self.assistant_app))  
        except Exception as ex:
            print(ex)
        try:
            # stop signal ctrl+Z
            signal.signal(signal.SIGTSTP, 
                          lambda sig, frame: self.signal_handler(sig, frame, self.assistant_app))  
        except Exception as ex:
            print(ex)
        try:
            # termination signal
            signal.signal(signal.SIGTERM, 
                          lambda sig, frame: self.signal_handler(sig, frame, self.assistant_app))  
        except Exception as ex:
            print(ex)

    @staticmethod
    def clear_chat():
        """Used to clear a chat windo by returning None"""
        return None
    
    
    
    @staticmethod
    def create_basic_settings_tab(settings_tab, init_temp, 
                                  max_temp: float=2.8, 
                                  options_list: list=[], 
                                  drop_down1_label="Knowledge-Nexus Options",
                                  **kwargs):
        with settings_tab:
            generation_settings_row1 = gr.Row()  # temp slider
            generation_settings_row2 = gr.Row()  # MODE selector radio (Base, RAG, GRAG)
            generation_settings_row3 = gr.Row()  # top-k, top-p, 
            system_directives_row1 = gr.Row()    # input box, 
            system_directives_row2 = gr.Row()    # submit directive button
            
            with generation_settings_row1:
                temp_slider = gr.Slider(label="Temperature: higher==more 'random'" , 
                                        minimum=0, maximum=max_temp, 
                                        value=init_temp, step=.001)

            # with generation_settings_row2:
            with system_directives_row1:
                system_input = gr.Textbox(label="System Directives:",
                                     value="", )
            # with generation_settings_row2:
            with system_directives_row2:
                system_input_button = gr.Button("System", scale=1)
                
            # ########################################################
            # # Pre-selected questions,
            # self.dropdown = gr.Dropdown(label=drop_down1_label, 
            #                             choices=options_list,
            #                             value=options_list[0],
            #                             interactive=True)
        return temp_slider, system_input, system_input_button

    def query_assistant(self, user_input):
        # print(f"\n\n\n\t\tSave content: {self.save_context}\n\n\n\n")
        # pass last state of conversation and new user input to assistant object
        # the assistant will add the user input to current conversation and use 
        # it to get a response from the assistant. If the save_context flag is set,
        # the user input and assistant response is are added to convesation for context,
        # otherwise the original state of convo is return
        # the method returns a tuple of the form:
        #  --- (conversation: List[dict], assistant_response: str)
        response = self.assistant_bot.response_generation_process(self.conversation, 
                                                                  user_input, 
                                                                  save_context=self.save_context)
        conversation, assistant_response = response[0], response[1]
        return conversation, assistant_response

    def add_rag_knowledge(self, rag_knowledge: str, format_string: str=base_rag_format_string):
        """Use the assistant_bot's built in method for generating a format_string"""
        rag_input = self.assistant_bot.add_rag_knowlege(rag_knowledge=rag_knowledge, 
                                                        format_string=format_string)
        return rag_input
    
    def generate_rag_knowledge(self, user_input, verbose=False, rag_preamble: str=None, 
                               return_rag_string=False,
                              ):
        # use assistant and it's knexus manager to get relevant information
        print(self.k, self.min_score, self.reverse, self.mode)
        rag_knowledge, rag_docs = self.assistant_bot.query_vector_store(
                                                   query=user_input, 
                                                   k=self.k, 
                                                   min_score=self.min_score, 
                                                   reverse=self.reverse,
                                                   mode=self.mode, verbose=False)
        # format_string = (
        #     "If the user asked a question related to the following information, "
        #     "prioritize the details provided here to answer it:\n\t\tPotentially related information:\n\n{rag_knowledge}"
        # )
        format_string = (
            "You are an AI assistant that helps users by analyzing and using system-provided information. "
            "The user may ask about some aspect of a maintenance task such as the scope of work for a given location. "
            "The locations are often given in the form of a numeric code such as 9202, or 9202-E. "
            "A reference to a location in the area may also be included such as 'on the southwest side'. " 
            "If the user's query relates to the following details, prioritize the information provided here "
            "to generate a response. When formulating your response:\n"
            "1. Carefully analyze the content to determine its relevance to the user's question.\n"
            "2. Identify which specific parts of the information are most pertinent to the user's needs.\n"
            "3. Explain how the relevant parts apply to the user's question, if needed.\n"
            "4. Use the relevant information to generate a detailed and helpful response.\n\n"
            "System-provided information:\n\n{rag_knowledge}\n\n"
            "If the provided information is not relevant, respond based on your general knowledge and "
            "clarify that the system-provided information does not apply to the user's question."
        )
        format_string = self.rag_premable
        # if rag_preamble:
        #     format_string = rag_preamble
        if verbose:
            print(f"Rag knowledge returned:\n\n{rag_knowledge}\n\n")
        rag_input = self.add_rag_knowledge(rag_knowledge=rag_knowledge, format_string=format_string)
        rag_prompt = system(rag_input)
        self.rag_knowledge = rag_knowledge
        if return_rag_string:
            return rag_prompt, rag_docs, rag_knowledge
        return rag_prompt, rag_docs
        
    
    
    def query_assistant_rag(self, user_input):
       
        # generate rag statement
        rag_prompt, rag_docs = self.generate_rag_knowledge(user_input)
        user_prompt = user(user_input)
        print(f"save: {self.save_context}, truncate: {self.truncate_conversation}")
        # need to check for truncation need here
        if self.save_context and self.truncate_conversation:
            at_max, new_tokens_possible = check_max_tokens(messages=self.conversation, 
                                             max_input_tokens=self.assistant_bot.max_input_tokens, 
                                             tokenizer=self.assistant_bot.tokenizer, user_input=user_input)
            print(f"new_tokens_possible: {new_tokens_possible}")
            estimated_memory_required = get_conversation_memory_usage(
                                            self.conversation, 
                                            tokenizer=self.assistant_bot.tokenizer, 
                                            model_hidden_size=self.assistant_bot.hidden_size, 
                                            dtype_size=self.assistant_bot.torch_dtype, 
                                            intermediate_factor=2)
            
            print(f"estimated memory: {estimated_memory_required}")
            print(f"device_map: {self.assistant_bot.assigned_gpu}")
            free_memory = get_gpu_mem_free(gpu_id=self.assistant_bot.assigned_gpu)
            hit_mem_limit = int(free_memory*.90) < int(estimated_memory_required) 
            print(f"Free memory: {free_memory}")
            print(f"Hig memory limit?: {hit_mem_limit}")
            if at_max or hit_mem_limit:
                print("\n\n\n\n\t\tabout to overflow!!!!\n\n\n\n")
                summary_string = summarize_conversation(self.assistant_bot.assistant, 
                                       self.assistant_bot.tokenizer, messages=self.conversation, 
                                       stop_strings=self.assistant_bot.stop_strings, max_new_tokens=300)
                print(f"Summary of conversation to this point:\n\n{summary_string}\n\n")
                self.conversation = replace_context_with_summary(summary_string)
            else:
                print(f"\n\n\t\tWe still good>>:\n\t\t\t\t\tTokens: Remaining vs Max:={new_tokens_possible} vs. {self.assistant_bot.max_input_tokens}\n\t\t\t\tMemory (Required vs Remaining){estimated_memory_required} B vs {free_memory} B\n\n\n")            
        
        # add rag and user inputs to convo to generate a response
        # torch.cuda.empty_cache()
        response = self.assistant_bot.generate_response(self.conversation + [user_prompt, rag_prompt], 
                                                        to_strip=[""])
        
        if self.save_context:
            # self.conversation += [user_prompt, assistant(response)]
            self.conversation += [user_prompt, rag_prompt, assistant(response)]
        return self.rag_knowledge, response
    
    def generate_response_dual(self, user_input, chat_history1, chat_history2, chat_history3):
        
        # use user input to retrieve relevant information
        rag_knowledge, response = self.query_assistant_rag(user_input)

        # update histories as needed
        chat_history1.append(user(f"## {self.uname}\n" + user_input))
        chat_history1.append(assistant(f"## Assistant:\n" + response))

        chat_history2.append(user(f"## {self.uname}\n" + user_input))
        chat_history2.append(assistant(f"## Assistant:\n" + response))

        chat_history3.append(user(f"## {self.uname}\n" + user_input))
        chat_history3.append(assistant(f"## Retrieved\n" + rag_knowledge))
        return None, chat_history1, chat_history2, chat_history3

    
    def generate_response(self, user_input, chat_history):
        self.conversation, response = self.query_assistant(user_input)
        chat_history.append({"role": "user", "content": f"# {self.uname}\n" + user_input})
        chat_history.append( {"role": "assistant", "content": f"# {self.assistant.name}\n" + response})
        return None, chat_history

    def reset_conversation(self, ):
        # get the directives from the convo:
        self.conversation = [self.init_system_directive]
        return
    def clear_conversation_dual(self,):
        """used to clear chat windows in dual chatbox scenario"""
        chat_history1 = {}
        chat_history2 = {}
        return chat_history1, chat_history2
    
    def clear_conversation_single(self,):
        """used to clear chat windows in dual chatbox scenario"""
        chat_history1 = {}
        return chat_history1
    
    def give_system_directive(self, new_directive, verbose=True):
        # old_directives = self.conversation[:self.directives_given+1]
        # other_context = self.conversation[self.directives_given+1:]
        # new_directives = old_directives + [self.assistant_bot.system(new_directive)]
        # self.conversation =  new_directives + other_context
        # print("\n\nNew directive")
        # print(self.conversation)
        self.conversation += [self.assistant_bot.system(new_directive)]
        if verbose:
            print("\n\n\t\t\t***New directive***")
            print(self.conversation)
            print("-----------------------\n\n")
        self.directives_given += 1
        return None


    def process_response_and_log_request(self, response_str, sentinel_string = "Your request has been submitted!"):
        # key_dict = {
        #     "Requester Name:":["NA"],
        #     "Badge Number:":["NA"],
        #     "Department:":["NA"],
        #     "Location:":["NA"],
        #     "Maintenance Work Location:":["NA"],
        #     "Reason for Request:":["NA"],
        #     "Specific Maintenance Request:":["NA"],
        #     "Physical Asset (FLOC):":["NA"],
        #     "Maintenance Type:":["NA"],
        # }
        
        
        # vectorize response by new lines, pull out required data points into a dict and store as a dataframe for later review:
        response_vector = response_str.split("\n")
        print(f"Vectorized response:\n{response_vector}")
        # iterate through the key dict adding the information found in the response vector
        for item in response_vector:
            for request_info in self.key_dict:
                if request_info in item:
                    print(f"Found: {request_info}\n\n")
                    self.key_dict[request_info] = [item.split(": ")[-1].strip()]
        name = self.key_dict["Requester Name:"][0].replace(" ", "_")
        badge = self.key_dict["Badge Number:"][0].replace(" ", "_")
        location = self.key_dict["Location:"][0].replace(" ", "_").replace(",", "_").replace(".", "_")
        department = self.key_dict["Department:"][0].replace(" ", "_").replace(",", "_").replace(".", "_")
            
        new_file_name = name +"_"+ badge  + "_" + location + "_" + department + "_" +  ".xlsx"
        
        if sentinel_string in response_str:
            key_df = pd.DataFrame(self.key_dict)
            print(f"saving file to: {new_file_name}")
            key_df.to_excel(self.logging_path +  new_file_name, index=False)
            self.reset_key_dict()
        else:
            print("\n\nNot ready for logging!!\n\n")

    def make_help_ui_3sec(self, 
        theme: gr.themes=gr.themes.Default,
        help_markdown1: str=help_markdown_base,
        help_markdown2: str=help_ui_tab_md1,
        help_markdown3: str=help_ui_tab_md2,
        help_img1: str="./utilities/images/amas_ui_help_img_tab1.png",
        help_img2: str="./utilities/images/amas_ui_help_img_tab2.png",
        help_img3: str="./utilities/images/amas_ui_help_img_tab_settings.png",
        **kwargs, 
    ):
        if help_img1 or help_img2 or help_img3:
            with gr.Blocks(theme=theme, js=js_code, css=custom_css) as help_ui:
                
                with gr.Row():
                    help_markdown = gr.Markdown(help_markdown1, elem_classes=["markdown-text"])
                    
                with gr.Row():
                    
                    ui_tab_1_md_msg = gr.Markdown(help_markdown2, elem_classes=["markdown-text"])
                with gr.Row():
                    img_array = plt.imread(help_img1)
                    gr.Image(img_array)
                    
                if help_img2:
                    with gr.Row():
                        ui_tab_2_md_msg = gr.Markdown(help_markdown2, elem_classes=["markdown-text"])
                    with gr.Row():
                        # img_array2 = plt.imread(help_img2)
                        gr.Image(help_img2)
                    
                if help_img3:
                    with gr.Row():
                        ui_tab_3_md_msg = gr.Markdown(help_markdown3, elem_classes=["markdown-text"])
                    with gr.Row():
                        # img_array3 = plt.imread(help_img3)
                        gr.Image(help_img3)
                    
        return help_ui
    
    def make_comp_chatview_w_dropdown_alpha_1(self, 
            theme: gr.themes=gr.themes.Default,
            view_header: str="Chat View:\nSpeak with the assitant in the user prompt text box below",
            chatbox_header1: str="Chat Log",
            chatbox_header2: str="Retrieved Knowledge",
            chatbox_height: int=450,
            inputbox_label: str="User Prompt:",
            
            button_scale: int=1,
            dropdown_label: str="Example input questions",
            dropdown_options: list=["o1", "o2"],
        ):
        comp_chat_view=None
        with gr.Blocks(theme=theme, js=js_code, css=custom_css) as comp_chat_view:
            with gr.Row():
                # print(markdown_cb_header1)
                # make the two chatboxes.
                # the one on the left is just like the one on 
                # the first tab while the second box is a view 
                # that shows the user query, and the retrieved 
                # and annotated knowledge returned to the assistant
                tab_header = gr.Markdown(view_header, elem_classes=["markdown-text"])
            with gr.Row():
                self.assistant_chatbot2 = gr.Chatbot(scale=1, 
                                                     label=chatbox_header1,
                                                     height=chatbox_height,
                                                     elem_classes=["chatbot"],
                                                     type='messages')   # Chat Dialog box
                self.rag_chatbot = gr.Chatbot(scale=1, 
                                            label=chatbox_header2,
                                            height=chatbox_height,
                                            elem_classes=["chatbot"],
                                            type='messages')   # Chat Dialog box
            with gr.Row():
                # set up user input text box
                self.input_text2 = gr.Textbox(label="User Prompt:",
                                             value="", )
            with gr.Row():
                self.clear_button2 = gr.Button(
                    "Clear Input",
                    scale=button_scale,
                )

                self.submit_button2 = gr.Button(
                    "Submit",
                    scale=button_scale,
                )

                self.reset_button2 = gr.Button(
                    "Clear Conversation",
                    scale=button_scale,
                )
            
            # dropdowns row
            with gr.Row():
                self.dropdown_2 = gr.Dropdown(
                    label=dropdown_label, 
                    choices=dropdown_options, 
                    value=dropdown_options[0], 
                    interactive=True,
                )
            
            with gr.Row():
                self.dropdown_button2=gr.Button(
                    "Copy-Example",
                    scale=button_scale,
                ) 
            # set up the event handlers
            ##### User input handling
            self.input_text2.submit(self.generate_response_dual, 
                                    inputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])
            self.submit_button2.click(self.generate_response_dual, 
                                    inputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])

            #### clear/reset button logic
            self.clear_button2.click(self.clear_chat, outputs=[self.input_text2])
            self.reset_button.click(self.clear_conversation_dual, outputs=[self.assistant_chatbot2, self.rag_chatbot])
            self.dropdown_button2.click(self.copy_text, inputs=self.dropdown_2, outputs=self.input_text2)
        
        return comp_chat_view

    def set_input_event_handlers(self, input_textbox, chatbox1, chatbox2, ):
        pass

    def make_base_chatview_w_dropdown_alpha_2(self,
                            theme: gr.themes=gr.themes.Default,
                            view_header: str="Chat View:\nSpeak with the assitant in the user prompt text box below",
                            chatbox_height: int=450,
                            inputbox_label: str="User Prompt:",
                            
                            button_scale: int=1,
                            dropdown_label: str="Example input questions",
                            dropdown_options: list=["o1", "o2"],
                            rag_mode: str="off",
                           ):
        chatview_basic = None
           
        with gr.Blocks(theme=theme, js=js_code, css=custom_css) as chatview_basic:
            # header for short user instructinos
            with gr.Row():
                markdown_instructions = gr.Markdown(view_header, elem_classes=["markdown-text"])

            # make chatbot chatlog box
            with gr.Row():
                        self.assistant_chatbot = gr.Chatbot(scale=1, 
                                                         # label=markdown_cb_header1,
                                                         height=chatbox_height,
                                                         elem_classes=["chatbot"],
                                                         type='messages')   # Chat Dialog box
            # user input box row
            with gr.Row():
                # set up user input text box
                self.input_text = gr.Textbox(label=inputbox_label,
                                             value="", )
            # submit row
            with gr.Row():
                self.clear_button = gr.Button(
                    "Clear Input",
                    scale=button_scale,
                )

                self.submit_button = gr.Button(
                    "Submit",
                    scale=button_scale,
                )

                self.reset_button = gr.Button(
                    "Clear Conversation",
                    scale=button_scale,
                )
            
            # dropdowns row
            with gr.Row():
                self.dropdown_1 = gr.Dropdown(
                    label=dropdown_label, 
                    choices=dropdown_options, value=dropdown_options[0], interactive=True,
                )
            
            with gr.Row():
                self.dropdown_button1=gr.Button(
                    "Copy-Example",
                    scale=button_scale,
                )
           

            #### clear/reset button logic
            self.clear_button.click(self.clear_chat, outputs=[self.input_text])
            
            self.reset_button.click(self.clear_conversation_single, outputs=[self.assistant_chatbot])
            

            # drop down logic on tab 1
            self.dropdown_button1.click(self.copy_text, inputs=self.dropdown_1, outputs=self.input_text)
            
        return chatview_basic
    
    def make_base_chatview_w_dropdown_alpha_1(self,
                                theme: gr.themes=gr.themes.Default,
                                view_header: str="Chat View:\nSpeak with the assitant in the user prompt text box below",
                                chatbox_height: int=450,
                                inputbox_label: str="User Prompt:",
                                
                                button_scale: int=1,
                                dropdown_label: str="Example input questions",
                                dropdown_options: list=["o1", "o2"],
                                rag_mode: str="off",
                               ):
        chatview_basic = None
           
        with gr.Blocks(theme=theme, js=js_code, css=custom_css) as chatview_basic:
            # header for short user instructinos
            with gr.Row():
                markdown_instructions = gr.Markdown(view_header, elem_classes=["markdown-text"])

            # make chatbot chatlog box
            with gr.Row():
                        self.assistant_chatbot = gr.Chatbot(scale=1, 
                                                         # label=markdown_cb_header1,
                                                         height=chatbox_height,
                                                         elem_classes=["chatbot"],
                                                         type='messages')   # Chat Dialog box
            # user input box row
            with gr.Row():
                # set up user input text box
                self.input_text = gr.Textbox(label=inputbox_label,
                                             value="", )
            # submit row
            with gr.Row():
                self.clear_button = gr.Button(
                    "Clear Input",
                    scale=button_scale,
                )

                self.submit_button = gr.Button(
                    "Submit",
                    scale=button_scale,
                )

                self.reset_button = gr.Button(
                    "Clear Conversation",
                    scale=button_scale,
                )
            
            # dropdowns row
            with gr.Row():
                self.dropdown_1 = gr.Dropdown(
                    label=dropdown_label, 
                    choices=dropdown_options, value=dropdown_options[0], interactive=True,
                )
            
            with gr.Row():
                self.dropdown_button1=gr.Button(
                    "Copy-Example",
                    scale=button_scale,
                )
            # set up the event handlers
            ######################################################################
            #                                       set up event handlers
            ##### User input handling
            # if rag_mode == "off":
            self.input_text.submit(self.generate_response, 
                                   inputs=[self.input_text,
                                           self.assistant_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot])
            self.submit_button.click(self.generate_response, 
                                    inputs=[self.input_text,
                                           self.assistant_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot])
            # else:
            #     print("\n\n\n\t\t\tRAG Mode is on\n\n\n")
            #     self.input_text.submit(self.generate_response_dual, 
            #                        inputs=[self.input_text,
            #                                self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
            #                        outputs=[self.input_text,
            #                                self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])
            #     self.submit_button.click(self.generate_response_dual, 
            #                         inputs=[self.input_text,
            #                                self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
            #                        outputs=[self.input_text,
            #                                self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])

            #### clear/reset button logic
            self.clear_button.click(self.clear_chat, outputs=[self.input_text])
            
            self.reset_button.click(self.clear_conversation_single, outputs=[self.assistant_chatbot])
            

            # drop down logic on tab 1
            self.dropdown_button1.click(self.copy_text, inputs=self.dropdown_1, outputs=self.input_text)
            
        return chatview_basic


    def make_base_chatview_w_dropdown_alpha_3(self,
                                theme: gr.themes=gr.themes.Default,
                                view_header: str="Chat View:\nSpeak with the assitant in the user prompt text box below",
                                chatbox_height: int=450,
                                inputbox_label: str="User Prompt:",
                                
                                button_scale: int=1,
                                dropdown_label: str="Example input questions",
                                dropdown_options: list=["o1", "o2"],
                                rag_mode: str="off",
                                custom_css: str=custom_css,
                               ):
        chatview_basic = None
        rag_tk_slider=None
        rag_sim_lb=None
        reverse_checkbox=None
        convo_text = None
        download_button=None
        file_output=None
        upload_file = None
        output_text = None
        download_button2=None
        file_output2=None
        upload_file2 = None
        output_text2 = None
        load_model_button = None
        model_dropdown = None
        load_model_button2 = None
        model_dropdown2 = None
        max_tokens_box = None
        stop_strings_box = None
        output_model_text = None
        max_tokens_box2 = None
        stop_strings_box2 = None
        output_model_text2 = None
        with gr.Blocks(theme=theme, js=js_code, css=custom_css) as chatview_basic:
            # header for short user instructinos
            with gr.Row():
                markdown_instructions = gr.Markdown(view_header, elem_classes=["markdown-text"])

            # make chatbot chatlog box
            with gr.Row():
                        self.assistant_chatbot = gr.Chatbot(scale=1, 
                                                         # label=markdown_cb_header1,
                                                         height=chatbox_height,
                                                         elem_classes=["chatbot"],
                                                         type='messages')   # Chat Dialog box
            # user input box row
            with gr.Row():
                # set up user input text box
                self.input_text = gr.Textbox(label=inputbox_label,
                                             value="", )
            # submit row
            with gr.Row():
                self.clear_button = gr.Button(
                    "Clear Input",
                    scale=button_scale,
                )

                self.submit_button = gr.Button(
                    "Submit", variant="secondary",
                    scale=button_scale,  elem_classes=["submit-button"], elem_id="submit-btn",
                )

                self.reset_button = gr.Button(
                    "Clear Conversation",
                    scale=button_scale,
                )
            
            # dropdowns row
            with gr.Row():
                self.dropdown_1 = gr.Dropdown(
                    label=dropdown_label, 
                    choices=dropdown_options, value=dropdown_options[0], interactive=True,
                )
            
            with gr.Row():
                self.dropdown_button1=gr.Button(
                    "Copy-Example",
                    scale=button_scale,
                )

            # add conversation logging/loading and model selection
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Accordion("Conversation Historian", open=False):
                        with gr.Accordion("Download Conversation", open=False):
                            with gr.Row():
                                download_button2 = gr.Button("Download Conversation", scale=1, elem_classes=['my-button-a'])
                            with gr.Row():
                                # A File component that will display the downloadable file link
                                file_output2 = gr.File(label="Download your conversation", scale=1, elem_classes=['my-file'])
                        with gr.Accordion("Upload Conversation from JSON", open=False):  
                            with gr.Row():
                                upload_file2 = gr.File(label="Upload Conversation JSON", 
                                                      file_count="single", type="filepath", 
                                                      scale=1, elem_classes=['my-file'])
                        # Model section
                        with gr.Accordion("Model Selection", open=False):  
                            with gr.Row():
                                model_dropdown2 = gr.Dropdown(choices=conversational_models, label="Select Model")
                                max_tokens_box2 = gr.Number(value=4080, label="Max Tokens", precision=0)
                                # stop_strings_box = gr.Textbox(label="Stop Strings (comma-separated)")
                                
                                load_model_button2 = gr.Button("Load Model")
                                output_model_text2 = gr.Textbox(label="Status", interactive=False)
            
            # set up the event handlers
            ######################################################################
            #                                       set up event handlers
            ##### User input handling
            # if rag_mode == "off":
            self.input_text.submit(self.generate_response, 
                                   inputs=[self.input_text,
                                           self.assistant_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot])
            self.submit_button.click(self.generate_response, 
                                    inputs=[self.input_text,
                                           self.assistant_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot])
           
            #### clear/reset button logic
            self.clear_button.click(self.clear_chat, outputs=[self.input_text])
            
            self.reset_button.click(self.clear_conversation_single, outputs=[self.assistant_chatbot])
            

            # drop down logic on tab 1
            self.dropdown_button1.click(self.copy_text, inputs=self.dropdown_1, outputs=self.input_text)

            download_button2.click(fn=self.save_conversation_to_json, inputs=[], outputs=file_output2) 
            
            # load_conversation_from_json
            upload_file2.upload(fn=self.load_conversation_from_json, 
                               inputs=[self.assistant_chatbot, upload_file2], 
                               outputs=[self.assistant_chatbot])
            load_model_button2.click(
                          fn=self.load_model_func, 
                          inputs=[model_dropdown2, max_tokens_box2], 
                          outputs=[output_model_text2])
            
        return chatview_basic
    
    def make_base_generation_settings_ui_alpa(self, 
            theme=gr.themes.Default,
            max_temp: float=2.8,
            temp_label: str="Generation Creativity: higher => more 'creative'",
        ):
        model_dropdown2 = None
        max_tokens_box2 = None
        # stop_strings_box = gr.Textbox(label="Stop Strings (comma-separated)")
        
        load_model_button2 = None
        output_model_text2 = None
        with gr.Blocks(theme=theme) as setting_ui:
            
            #     generation_settings_row1 = gr.Row()  # temp slider
            #     tk_tp_mtkn_row = gr.Row()     # set sliders
            #     # generation_settings_row2 = gr.Row()  # MODE selector radio (Base, RAG, GRAG)
            #     system_directives_row1 = gr.Row()    # input box, 
            #     system_directives_row2 = gr.Row()    # submit directive button
            with gr.Row():
                settings_instructions = """
                This tabs allow you to control how the assistant generated the text in their responses. 
                The temp slider controls how it chooses to put words together. The top % of MLE controls 
                the lowerbound on the probability of a given piece of text be chosen for generation 
                while the 'top # of MLE' controls how many of the top most probable pieces of text will 
                be selected for choosing which text to generate next. Too high a temp will lead to the 
                model generating non-language while at a value of 0 it will just always choose the most 
                likely piece of text making generation more deterministic. 

                The system directive box allows you to give the assistant directives for their behavior. 
                For instance you can tell them to answer the next response with an accent.
                """
                markdown_instructions = gr.Markdown(settings_instructions, elem_classes=["markdown-text"])
            with gr.Row():
                self.temp_slider = gr.Slider(label=temp_label, 
                                        minimum=0, maximum=max_temp, 
                                        value=self.assistant_bot.temp, step=.01)

            with gr.Row():
                self.tk_slider = gr.Slider(label="Top # of MLE" , 
                                        minimum=10, maximum=60, 
                                        value=self.assistant_bot.top_k if self.assistant_bot.top_k else 10, 
                                        step=5)
                self.tp_slider = gr.Slider(label="Top % of MLE" , 
                                        minimum=0.10, maximum=.70, 
                                        value=self.assistant_bot.top_p if self.assistant_bot.top_p else 0.1,
                                        interactive=True,
                                        step=.05)
                self.max_new_token_slider = gr.Slider(label="Max New Tokens", 
                                        minimum=400, maximum=8000, 
                                        value=self.assistant_bot.max_new_tokens if self.assistant_bot.max_new_tokens else 400,
                                        interactive=True,
                                        step=100)
            # with generation_settings_row2:
            with gr.Row():
                self.system_dir_text = gr.Textbox(label="System Directives:",
                                     value="", )
            # with generation_settings_row2:
            with gr.Row():
                self.system_dir_button = gr.Button("System", scale=1)
            with gr.Row():
                # Model section
                with gr.Accordion("Model Selection", open=False):  
                    with gr.Row():
                        model_dropdown2 = gr.Dropdown(choices=conversational_models, label="Select Model")
                        max_tokens_box2 = gr.Number(value=4080, label="Max Tokens", precision=0)
                        # stop_strings_box = gr.Textbox(label="Stop Strings (comma-separated)")
                        
                        load_model_button2 = gr.Button("Load Model")
                        output_model_text2 = gr.Textbox(label="Status", interactive=False)
            
            # add event handlers
            #### generation settings logic
            self.temp_slider.change(self.assistant_bot.update_temperature, inputs=self.temp_slider)
            self.tp_slider.change(self.assistant_bot.update_top_p, inputs=self.tp_slider)
            self.tk_slider.change(self.assistant_bot.update_top_k, inputs=self.tk_slider)
            self.max_new_token_slider.change(self.assistant_bot.update_max_new_tokens, inputs=self.max_new_token_slider)
            
            self.system_dir_text.submit(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])
            self.system_dir_button.click(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])   
            load_model_button2.click(
                          fn=self.load_model_func, 
                          inputs=[model_dropdown2, max_tokens_box2], 
                          outputs=[output_model_text2])
        return setting_ui

    def make_base_generation_settings_and_help_ui_alpa(self, 
            theme=gr.themes.Default,
            max_temp: float=2.8,
            temp_label: str="Generation Creativity: higher => more 'creative'",
            help_markdown1: str=help_markdown_base,
            help_markdown2: str=help_ui_tab_md1,
            help_markdown3: str=help_ui_tab_md2,
            help_img1: str="./utilities/images/amas_ui_help_img_tab1.png",
            help_img2: str="./utilities/images/amas_ui_help_img_tab2.png",
            help_img3: str="./utilities/images/amas_ui_help_img_tab_settings.png",
        ):
        model_dropdown2 = None
        max_tokens_box2 = None
        # stop_strings_box = gr.Textbox(label="Stop Strings (comma-separated)")
        
        load_model_button2 = None
        output_model_text2 = None
        with gr.Blocks(theme=theme) as setting_ui:
            
            #     generation_settings_row1 = gr.Row()  # temp slider
            #     tk_tp_mtkn_row = gr.Row()     # set sliders
            #     # generation_settings_row2 = gr.Row()  # MODE selector radio (Base, RAG, GRAG)
            #     system_directives_row1 = gr.Row()    # input box, 
            #     system_directives_row2 = gr.Row()    # submit directive button
            with gr.Accordion("Generation Settings"):
                with gr.Row():
                    settings_instructions = """
                    This tabs allow you to control how the assistant generated the text in their responses. 
                    The temp slider controls how it chooses to put words together. The top % of MLE controls 
                    the lowerbound on the probability of a given piece of text be chosen for generation 
                    while the 'top # of MLE' controls how many of the top most probable pieces of text will 
                    be selected for choosing which text to generate next. Too high a temp will lead to the 
                    model generating non-language while at a value of 0 it will just always choose the most 
                    likely piece of text making generation more deterministic. 
    
                    The system directive box allows you to give the assistant directives for their behavior. 
                    For instance you can tell them to answer the next response with an accent.
                    """
                    markdown_instructions = gr.Markdown(settings_instructions, elem_classes=["markdown-text"])
                with gr.Row():
                    self.temp_slider = gr.Slider(label=temp_label, 
                                            minimum=0, maximum=max_temp, 
                                            value=self.assistant_bot.temp, step=.01)
    
                with gr.Row():
                    self.tk_slider = gr.Slider(label="Top # of MLE" , 
                                            minimum=10, maximum=60, 
                                            value=self.assistant_bot.top_k if self.assistant_bot.top_k else 10, 
                                            step=5)
                    self.tp_slider = gr.Slider(label="Top % of MLE" , 
                                            minimum=0.10, maximum=.70, 
                                            value=self.assistant_bot.top_p if self.assistant_bot.top_p else 0.1,
                                            interactive=True,
                                            step=.05)
                    self.max_new_token_slider = gr.Slider(label="Max New Tokens", 
                                            minimum=400, maximum=8000, 
                                            value=self.assistant_bot.max_new_tokens if self.assistant_bot.max_new_tokens else 400,
                                            interactive=True,
                                            step=100)
                # with generation_settings_row2:
                with gr.Row():
                    self.system_dir_text = gr.Textbox(label="System Directives:",
                                         value="", )
                # with generation_settings_row2:
                with gr.Row():
                    self.system_dir_button = gr.Button("System", scale=1)
                with gr.Row():
                    # Model section
                    with gr.Accordion("Model Selection", open=False):  
                        with gr.Row():
                            model_dropdown2 = gr.Dropdown(choices=conversational_models, label="Select Model")
                            max_tokens_box2 = gr.Number(value=4080, label="Max Tokens", precision=0)
                            # stop_strings_box = gr.Textbox(label="Stop Strings (comma-separated)")
                            
                            load_model_button2 = gr.Button("Load Model")
                            output_model_text2 = gr.Textbox(label="Status", interactive=False)
            
            if help_img1 or help_img2 or help_img3:
                help_markdown = None
                ui_tab_1_md_msg = None
                ui_tab_2_md_msg = None
                ui_tab_3_md_msg = None
                with gr.Accordion("Assistant-Help"):
                    with gr.Row():
                        help_markdown = gr.Markdown(help_markdown1, elem_classes=["markdown-text"])
                        
                    with gr.Row():
                        
                        ui_tab_1_md_msg = gr.Markdown(help_markdown2, elem_classes=["markdown-text"])
                    with gr.Row():
                        img_array = plt.imread(help_img1)
                        gr.Image(img_array)
                        
                    if help_img2:
                        with gr.Row():
                            ui_tab_2_md_msg = gr.Markdown(help_markdown2, elem_classes=["markdown-text"])
                        with gr.Row():
                            # img_array2 = plt.imread(help_img2)
                            gr.Image(help_img2)
                        
                    if help_img3:
                        with gr.Row():
                            ui_tab_3_md_msg = gr.Markdown(help_markdown3, elem_classes=["markdown-text"])
                        with gr.Row():
                            # img_array3 = plt.imread(help_img3)
                            gr.Image(help_img3)
            
            # add event handlers
            #### generation settings logic
            self.temp_slider.change(self.assistant_bot.update_temperature, inputs=self.temp_slider)
            self.tp_slider.change(self.assistant_bot.update_top_p, inputs=self.tp_slider)
            self.tk_slider.change(self.assistant_bot.update_top_k, inputs=self.tk_slider)
            self.max_new_token_slider.change(self.assistant_bot.update_max_new_tokens, inputs=self.max_new_token_slider)
            
            self.system_dir_text.submit(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])
            self.system_dir_button.click(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])   
            load_model_button2.click(
                          fn=self.load_model_func, 
                          inputs=[model_dropdown2, max_tokens_box2], 
                          outputs=[output_model_text2])
        return setting_ui
    
    def make_basic_chat_ui(self, 
                        theme=gr.themes.Default,
                        view_header: str="Chat View:\nSpeak with the assitant in the user prompt text box below",
                        chatbox_height: int=450,
                        inputbox_label: str="User Prompt:",  
                        button_scale: int=1,
                        dropdown_label: str="Example input questions",
                        dropdown_options: list=[],
                        max_temp=2.8,
                        temp_label: str="Generation Creativity: higher => more 'creative'",
                        **kwargs,   
                        ):
        chat_ui = self.make_base_chatview_w_dropdown_alpha_1(
                        theme=theme, view_header=view_header, 
                        chatbox_height=chatbox_height,
                        inputbox_label=inputbox_label,
                        button_scale=button_scale,
                        dropdown_label=dropdown_label,
                        dropdown_options=dropdown_options
                    )
        
        settings_ui = self.make_base_generation_settings_ui_alpa(
                                            theme=theme,
                                            max_temp=max_temp,
                                            temp_label=temp_label)
        # make the tabbed interface
        self.assistant_app = gr.TabbedInterface([chat_ui, settings_ui], ["Assistant Chat", "Generation Settings"], theme=theme)

    def make_basic_chat_ui_v2(self, 
                        theme=gr.themes.Default,
                        view_header: str="Chat View:\nSpeak with the assitant in the user prompt text box below",
                        chatbox_height: int=450,
                        inputbox_label: str="User Prompt:",  
                        button_scale: int=1,
                        dropdown_label: str="Example input questions",
                        dropdown_options: list=[],
                        max_temp=2.8,
                        temp_label: str="Generation Creativity: higher => more 'creative'",
                        **kwargs,   
                        ):
        chat_ui = self.make_base_chatview_w_dropdown_alpha_3(
                        theme=theme, view_header=view_header, 
                        chatbox_height=chatbox_height,
                        inputbox_label=inputbox_label,
                        button_scale=button_scale,
                        dropdown_label=dropdown_label,
                        dropdown_options=dropdown_options
                    )
        
        settings_ui = self.make_base_generation_settings_ui_alpa(
                                            theme=theme,
                                            max_temp=max_temp,
                                            temp_label=temp_label)
        # make the tabbed interface
        self.assistant_app = gr.TabbedInterface([chat_ui, settings_ui], ["Assistant Chat", "Generation Settings"], theme=theme, css=custom_css)
    
    def make_requester_chat_app_v2(self,
        theme=gr.themes.Default,
        view_header: str=None,
        chatbox_height: int=450,
        inputbox_label: str="Message to MRA:",  
        button_scale: int=1,
        dropdown_label: str="Example input questions",
        dropdown_options: list=maintenance_requests,

        max_temp=2.8,
        temp_label: str="Generation Creativity: higher => more 'creative'",
        k=2,
        min_score=.80,
        reverse=False,
        mode="min",
        rag_format_string=base_instructor_rag_format_string,
        **kwargs,
    ):
        self.k=k
        self.min_score=min_score
        self.reverse=reverse
        self.mode=mode
        print(f"theme3: {theme}")
        if not view_header:
            view_header = """
            This is an initial demo for the AI Maintenance Requester Assistant (AMRA). This app demonstrates the capabilities of the assistant
            to accurate interact with users, guide them through inputing a good maintenace request, and store the request for planner
            processing. This preliminary version attempts to prompt the user for the requester name, badge number, department, location, the
            reason for the request, the desired outcome, the FLOC, and type of maintenace. Below is the main chat log showing your input and
            the assistants outputs. Below the chat log is the area for entering your messages including
            a text box for your messages to the assistant, a submit button, a button to clear the text box, and a button to clear out the chat
            history. 
            
            Below the submit area there is a dropdown box that has sample inputs to get you started. You can select an option and use 
            the copy-button to copy it into the user input area. The settings tab is more of a developer tool that allows you to alter 
            the LLM generation parameters.
    
            You can start chatting with the assistant about some maintenance request and it will attempt to guide you through the process.
            The assistant seeks a name, badge number, reason for request, desired outcome, work center, FLOC
            """
        self.make_basic_chat_ui_v2( 
                    theme=theme,
                    chatbox_height=chatbox_height,
                    inputbox_label=inputbox_label,
                    button_scale=button_scale,
                    dropdown_label=dropdown_label,
                    dropdown_options=dropdown_options,
                    max_temp=max_temp,
                    temp_label=temp_label,
                    )
        return
    
    
    
    def make_requester_chat_app(self,
                theme=gr.themes.Default, 
                user_greeting="Greetings!", 
                chat_bot_label="Chat", 
                max_temp=2.8,
                tab_1_label="Assistant Chat",
                tab_2_label="# Chat & Retrieval View",
                tab_3_label="# System Settings",
                header_md_1=None,
                header_md_2=None,
                header_md_3=None,
                chatbox_height=450,
                button_scale=1,
                k=2,
                min_score=.0001,
                reverse=False,
                mode="min",
                rag_format_string=base_rag_format_string,
                options1=sow_prompt_options_1,
                options2=sow_prompt_options_1,
                options3=sow_prompt_options_1,
                options4=sow_prompt_options_1,
                **kwargs,
        ):
        self.k=k
        self.min_score=min_score
        self.reverse=reverse
        self.mode=mode
        print(f"theme0: {theme}")
        self.rag_premable=rag_format_string
        with gr.Blocks(theme=theme, js=js_code, css=custom_css) as self.assistant_app:
            
            # make basic chat UI Tab
            ################################
            ##       Chatbox UI           ##
            ################################
            ################################
            ##       user input box       ##
            ################################
            ################################
            ##       user input buttons   ##
            ################################
            with gr.Tab(label=tab_1_label):
                # header_row = gr.Row()
                # chatbox_row = gr.Row()
                # input_row = gr.Row()
                # sbutton_row = gr.Row()
                # dropdown_row = gr.Row()
                # dropdown_btn_row = gr.Row()
                # # first row should be the greeting/header
                # with gr.Row():
                #     markdown_header1=(
                #         f"## Chat with {self.name} in the user input area below..."
                #     ) if not header_md_1 else header_md_1
                #     markdown_obj = gr.Markdown()
                # chat box row
                with gr.Row():
                    # make instruction header
                    mra_header = """
                    This is an initial demo for the AI Maintenance Requester Assistant (AMRA). This app demonstrates the capabilities of the assistant
                    to accurate interact with users, guide them through inputing a good maintenace request, and store the request for planner
                    processing. This preliminary version attempts to prompt the user for the requester name, badge number, department, location, the
                    reason for the request, the desired outcome, the FLOC, and type of maintenace. Below is the main chat log showing your input and
                    the assistants outputs. Below the chat log is the area for entering your messages including
                    a text box for your messages to the assistant, a submit button, a button to clear the text box, and a button to clear out the chat
                    history. 
                    
                    Below the submit area there is a dropdown box that has sample inputs to get you started. You can select an option and use 
                    the copy-button to copy it into the user input area. The settings tab is more of a developer tool that allows you to alter 
                    the LLM generation parameters.

                    You can start chatting with the assistant about some maintenance request and it will attempt to guide you through the process.
                    The assistant seeks a name, badge number, reason for request, desired outcome, work center, FLOC
                    """
                    # markdown_header1=(
                    #     f"## Chat with {self.name} in the user input area below..."
                    # ) if not header_md_1 else header_md_1
                    markdown_obj = gr.Markdown(mra_header, elem_classes=["markdown-text"])
                with gr.Row():
                    markdown_cb_header1=(
                            f"## Chat with {self.assistant_bot.name} in the user input area below..."
                        ) if not header_md_1 else header_md_1
                    print(markdown_cb_header1)
                    self.assistant_chatbot = gr.Chatbot(scale=1, 
                                                     label=markdown_cb_header1,
                                                     height=chatbox_height,
                                                     elem_classes=["chatbot"],
                                                     type='messages')   # Chat Dialog box
                # input box row
                with gr.Row():
                    # set up user input text box
                    self.input_text = gr.Textbox(label="User Prompt:",
                                                 value="", )
                # submit row
                with gr.Row():
                    self.clear_button = gr.Button(
                        "Clear Input",
                        scale=button_scale,
                    )

                    self.submit_button = gr.Button(
                        "Submit",
                        scale=button_scale, elem_classes='submit-button',
                    )

                    self.reset_button = gr.Button(
                        "Reset Conversation",
                        scale=button_scale,
                    )
                
                # dropdowns row
                with gr.Row():
                    self.dropdown_1 = gr.Dropdown(
                        label="Example input questions", 
                        choices=options1, value=options1[0], interactive=True,
                    )
                
                with gr.Row():
                    self.dropdown_button1=gr.Button(
                        "Copy-Example",
                        scale=button_scale,
                    )

          
            # Settings Tab
            with gr.Tab(label=tab_3_label):
            #     generation_settings_row1 = gr.Row()  # temp slider
            #     tk_tp_mtkn_row = gr.Row()     # set sliders
            #     # generation_settings_row2 = gr.Row()  # MODE selector radio (Base, RAG, GRAG)
            #     system_directives_row1 = gr.Row()    # input box, 
            #     system_directives_row2 = gr.Row()    # submit directive button
                
                with gr.Row():
                    self.temp_slider = gr.Slider(label="Temperature: higher => more 'RNG'" , 
                                            minimum=0, maximum=max_temp, 
                                            value=self.assistant_bot.temp, step=.01)
    
                with gr.Row():
                    self.tk_slider = gr.Slider(label="Top # of MLE" , 
                                            minimum=10, maximum=60, 
                                            value=self.assistant_bot.top_k if self.assistant_bot.top_k else 10, 
                                            step=5)
                    self.tp_slider = gr.Slider(label="Top % of MLE" , 
                                            minimum=0.10, maximum=.70, 
                                            value=self.assistant_bot.top_p if self.assistant_bot.top_p else 0.1,
                                            interactive=True,
                                            step=.05)
                    self.max_new_token_slider = gr.Slider(label="Max New Tokens", 
                                            minimum=400, maximum=8000, 
                                            value=self.assistant_bot.max_new_tokens if self.assistant_bot.max_new_tokens else 400,
                                            interactive=True,
                                            step=100)
                # with generation_settings_row2:
                with gr.Row():
                    self.system_dir_text = gr.Textbox(label="System Directives:",
                                         value="", )
                # with generation_settings_row2:
                with gr.Row():
                    self.system_dir_button = gr.Button("System", scale=1)
            ######################################################################
            #                                       set up event handlers
            ##### User input handling
            self.input_text.submit(self.generate_response, 
                                   inputs=[self.input_text,
                                           self.assistant_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot])
            self.submit_button.click(self.generate_response, 
                                    inputs=[self.input_text,
                                           self.assistant_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot])


            #### clear/reset button logic
            self.clear_button.click(self.clear_chat, outputs=[self.input_text])
            self.reset_button.click(self.clear_conversation_single, outputs=[self.assistant_chatbot])

            # drop down logic on tab 1
            self.dropdown_button1.click(self.copy_text, inputs=self.dropdown_1, outputs=self.input_text)
            

            #### generation settings logic
            self.temp_slider.change(self.assistant_bot.update_temperature, inputs=self.temp_slider)
            self.tp_slider.change(self.assistant_bot.update_top_p, inputs=self.tp_slider)
            self.tk_slider.change(self.assistant_bot.update_top_k, inputs=self.tk_slider)
            self.max_new_token_slider.change(self.assistant_bot.update_max_new_tokens, inputs=self.max_new_token_slider)
            
            self.system_dir_text.submit(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])
            self.system_dir_button.click(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])


    def make_dual_chat_tabs(self,                          
            theme: gr.themes=gr.themes.Default,
            
            chatbox_height1: int=450,
            chatbox_height2: int=450,
            inputbox_label: str="User Prompt:",
            button_scale: int=1,

            view_header: str="Chat View:\nSpeak with the assitant in the user prompt text box below",
            view_header2: str="Chat View:\nSpeak with the assitant in the user prompt text box below",
            chatbox_header1: str="Chat Log",
            chatbox_header2: str="Retrieved Knowledge",
            
            dropdown_label: str="Example input questions",
            dropdown_options: list=["o1", "o2"],
        ):
        chatview_basic = None
        main_tab=None
        dual_tab=None
        # set up main chat view
        with gr.Blocks(theme=theme, js=js_code, css=custom_css) as chatview_basic:
            with gr.Tab("main") as main_tab:
                # header for short user instructinos
                with gr.Row():
                    markdown_instructions = gr.Markdown(view_header, elem_classes=["markdown-text"])
    
                # make chatbot chatlog box
                with gr.Row():
                            self.assistant_chatbot = gr.Chatbot(scale=1, 
                                                             # label=markdown_cb_header1,
                                                             height=chatbox_height1,
                                                             elem_classes=["chatbot"],
                                                             type='messages')   # Chat Dialog box
                # user input box row
                with gr.Row():
                    # set up user input text box
                    self.input_text = gr.Textbox(label=inputbox_label,
                                                 value="", )
                # submit row
                with gr.Row():
                    self.clear_button = gr.Button(
                        "Clear Input",
                        scale=button_scale,
                    )
    
                    self.submit_button = gr.Button(
                        "Submit",
                        scale=button_scale,
                    )
    
                    self.reset_button = gr.Button(
                        "Clear Conversation",
                        scale=button_scale,
                    )
                
                # dropdowns row
                with gr.Row():
                    self.dropdown_1 = gr.Dropdown(
                        label=dropdown_label, 
                        choices=dropdown_options, value=dropdown_options[0], interactive=True,
                    )
                
                with gr.Row():
                    self.dropdown_button1=gr.Button(
                        "Copy-Example",
                        scale=button_scale,
                    )
        
            with gr.Tab("dual") as dual_tab:
                with gr.Row():
                    # print(markdown_cb_header1)
                    # make the two chatboxes.
                    # the one on the left is just like the one on 
                    # the first tab while the second box is a view 
                    # that shows the user query, and the retrieved 
                    # and annotated knowledge returned to the assistant
                    tab_header = gr.Markdown(view_header2, elem_classes=["markdown-text"])
                with gr.Row():
                    self.assistant_chatbot2 = gr.Chatbot(scale=1, 
                                                         label=chatbox_header1,
                                                         height=chatbox_height2,
                                                         elem_classes=["chatbot"],
                                                         type='messages')   # Chat Dialog box
                    self.rag_chatbot = gr.Chatbot(scale=1, 
                                                label=chatbox_header2,
                                                height=chatbox_height2,
                                                elem_classes=["chatbot"],
                                                type='messages')   # Chat Dialog box
                with gr.Row():
                    # set up user input text box
                    self.input_text2 = gr.Textbox(label=inputbox_label,
                                                 value="", )
                with gr.Row():
                    self.clear_button2 = gr.Button(
                        "Clear Input",
                        scale=button_scale,
                    )
    
                    self.submit_button2 = gr.Button(
                        "Submit",
                        scale=button_scale,
                    )
    
                    self.reset_button2 = gr.Button(
                        "Clear Conversation",
                        scale=button_scale,
                    )
                
                # dropdowns row
                with gr.Row():
                    self.dropdown_2 = gr.Dropdown(
                        label=dropdown_label, 
                        choices=dropdown_options, 
                        value=dropdown_options[0], 
                        interactive=True,
                    )
                
                with gr.Row():
                    self.dropdown_button2=gr.Button(
                        "Copy-Example",
                        scale=button_scale,
                    ) 
            # set up the event handlers 
            self.input_text.submit(self.generate_response_dual, 
                               inputs=[self.input_text,
                                       self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                               outputs=[self.input_text,
                                       self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])
            self.submit_button.click(self.generate_response_dual, 
                                inputs=[self.input_text,
                                       self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                               outputs=[self.input_text,
                                       self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])

            #### clear/reset button logic
            self.clear_button.click(self.clear_chat, outputs=[self.input_text])
            
            self.reset_button.click(self.clear_conversation_single, outputs=[self.assistant_chatbot])
            
            # drop down logic on tab 1
            self.dropdown_button1.click(self.copy_text, inputs=self.dropdown_1, outputs=self.input_text)
            
            ##### User input handling
            self.input_text2.submit(self.generate_response_dual, 
                                    inputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])
            self.submit_button2.click(self.generate_response_dual, 
                                    inputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])

            #### clear/reset button logic
            self.clear_button2.click(self.clear_chat, outputs=[self.input_text2])
            self.reset_button.click(self.clear_conversation_dual, outputs=[self.assistant_chatbot2, self.rag_chatbot])
            self.dropdown_button2.click(self.copy_text, inputs=self.dropdown_2, outputs=self.input_text2)
        return chatview_basic, main_tab, dual_tab
    
    def make_chat_comp_settings_app_v2(self,
                theme=gr.themes.Default, 
                view_header_main=view_header,
                view_header_rag=view_header,
                
                # view_header_help1=view_header,
                # help_img_1: str="",
                # view_header_help2=view_header,
                # help_img_2: str="",
                # view_header_help3=view_header,
                # help_img_3: str="",

                view_header4=view_header,
                chatbox1_header1="",
                chatbox1_height=500,
                chatbox2_height=500,
                inputbox_label="Message to Assistant",
                button_scale=1,
                dropdown_label="# Example User Inputs",
                dropdown_options=['o1', 'o2'],

                help_markdown1: str=help_markdown_base,
                help_markdown2: str=help_ui_tab_md1,
                help_markdown3: str=help_ui_tab_md2,
                help_img1: str="./utilities/images/amas_ui_help_img_tab1.png",
                help_img2: str="./utilities/images/amas_ui_help_img_tab2.png",
                help_img3: str="./utilities/images/amas_ui_help_img_tab2.png",

                tab_1_label="Assistant Chat",
                tab_2_label="Assistant Chat & Retrieval View",
                tab_3_label="Help",
                tab_4_label="System Generation Settings",
                
                max_temp=2.8,
                temp_label="'Creativity': 0 := most probable next word; max:= random text generation",
                js_code=js_code,
                custom_css=custom_css,
                **kwargs,
        ):
        # # make base chat view
        # main_chatview = self.make_base_chatview_w_dropdown_alpha_2(
        #                 theme=theme, view_header=view_header_main, 
        #                 chatbox_height=chatbox1_height,
        #                 inputbox_label=inputbox_label,
        #                 button_scale=button_scale,
        #                 dropdown_label=dropdown_label,
        #                 dropdown_options=dropdown_options,
        #                 rag_mode="on"
        #             )
        
        # # make comparison chat view
        # comparison_chatview = self.make_comp_chatview_w_dropdown_alpha_1(
        #     theme=theme, view_header=view_header_rag,
        #     chatbox_header1=chatbox1_header1,
        #     chatbox_header2=chatbox1_header1,
        #     chatbox_height=chatbox2_height,
        #     inputbox_label=inputbox_label,
            
        #     button_scale=button_scale,
        #     dropdown_label=dropdown_label,
        #     dropdown_options=dropdown_options,
        # )
        
        chatview_basic, main_chatview, comparison_chatview = self.make_dual_chat_tabs(                         
                theme=theme,
                view_header=view_header_main,
                chatbox_height1=chatbox1_height,
                inputbox_label=inputbox_label,
        
                view_header2=view_header_rag,
                chatbox_header1=chatbox1_header1,
                chatbox_header2=chatbox1_header1,
                chatbox_height2=chatbox2_height,
                
                button_scale=button_scale,
                dropdown_label=dropdown_label,
                dropdown_options=dropdown_options,
            )
        # make help tab
        help_ui = self.make_help_ui_3sec(
                help_markdown1=help_markdown1,
                help_markdown2=help_markdown2,
                help_markdown3=help_markdown3,
                help_img1=help_img1,
                help_img2=help_img2,
                help_img3=help_img2,
                **kwargs, 
            )
        
        # make settings tab
        settings_ui = self.make_base_generation_settings_ui_alpa(
                                                theme=theme,
                                                max_temp=max_temp,
                                                temp_label=temp_label)
        
           
        
        # print(dir(chatview_basic))
        # print(chatview_basic[0])
        # print(chatview_basic)
        # make the tabbed interface
        self.assistant_app = gr.TabbedInterface(
            # [main_chatview, comparison_chatview, help_ui, settings_ui], 
            # [chatview_basic, help_ui, settings_ui],
            [tab_1_label, tab_2_label, tab_3_label, tab_4_label], 
            theme=theme, js=js_code, css=custom_css,
        )
        return


    def update_rag_k(self, k_value: int):
        self.k = k_value
        return
    
    def update_cosine_threshold(self, value: float):
        self.min_score=value
        return
    
    def update_rag_ranking_reverse(self, is_checked: bool):
        self.reverse=is_checked
        return

    # self.save_conversation_to_json
    def save_conversation_to_json(self,):
        """
        Saves the conversation list to a JSON file and returns the file name.
        This file will be offered for download in the Gradio interface.
        """
        filename = "conversation.json"
        print(f"Saving conversation to {filename}")
        # Dump the conversation into a JSON file with indentation for readability
        with open(filename, "w") as f:
            json.dump(self.conversation, f, indent=2)
        return filename

    def convert_conversation_to_chat_history(self, chat_history, conv):
        """
        Converts the conversation list of dictionaries into a chat history 
        formatted as a list of tuples for the Gradio Chatbot.
        
        This function assumes that a "user" message is followed by an "assistant" message.
        If a user message is left without a corresponding response, an empty string is used.
        """
        
        temp = None
        for message in conv:
            chat_history.append(message)
        return chat_history
    
    def load_conversation_from_json_dual(self, chat_history1, chat_history2, uploaded_file):
        """
        Loads a conversation from an uploaded JSON file.
        
        This function does the following:
          1. Reads the uploaded JSON file.
          2. Saves a copy to the local directory (as "uploaded_conversation.json").
          3. Updates the global conversation variable with the loaded data.
          4. Returns a JSON-formatted string of the loaded conversation.
        """
        if uploaded_file is None:
            return "No file uploaded."
        
        try:
            # Read and load the JSON file contents from the temporary file path
            with open(uploaded_file.name, "r") as f:
                loaded_conv = json.load(f)
        except Exception as e:
            return f"Error loading JSON file: {str(e)}"
        
        # Save the uploaded JSON permanently in the local directory
        destination = "tmp/uploaded_conversations/uploaded_conversation.json"
        with open(destination, "w") as dest:
            json.dump(loaded_conv, dest, indent=2)
        
        # Update the conversation with the newly loaded data
        self.conversation = loaded_conv
        chat_history1 = self.convert_conversation_to_chat_history( chat_history1, self.conversation)
        chat_history2 = self.convert_conversation_to_chat_history( chat_history2, self.conversation)
        # Return a pretty-printed JSON string to display the updated conversation in the UI
        # return json.dumps(self.conversation, indent=2)
        return chat_history1, chat_history2

    def load_conversation_from_json(self, chat_history1, uploaded_file):
        """
        Loads a conversation from an uploaded JSON file.
        
        This function does the following:
          1. Reads the uploaded JSON file.
          2. Saves a copy to the local directory (as "uploaded_conversation.json").
          3. Updates the global conversation variable with the loaded data.
          4. Returns a JSON-formatted string of the loaded conversation.
        """
        if uploaded_file is None:
            return "No file uploaded."
        
        try:
            # Read and load the JSON file contents from the temporary file path
            with open(uploaded_file.name, "r") as f:
                loaded_conv = json.load(f)
        except Exception as e:
            return f"Error loading JSON file: {str(e)}"
        
        # Save the uploaded JSON permanently in the local directory
        destination = "tmp/uploaded_conversations/uploaded_conversation.json"
        with open(destination, "w") as dest:
            json.dump(loaded_conv, dest, indent=2)
        
        # Update the conversation with the newly loaded data
        self.conversation = loaded_conv
        chat_history1 = self.convert_conversation_to_chat_history( chat_history1, self.conversation)
        # Return a pretty-printed JSON string to display the updated conversation in the UI
        # return json.dumps(self.conversation, indent=2)
        return chat_history1
    
    # def load_model_func(self, model_name, max_tokens, stop_strings=""):
    def load_model_func(self, model_name, max_tokens):
        try:
            # tokenizer = AutoTokenizer.from_pretrained(model_name)
            # model = AutoModelForCausalLM.from_pretrained(model_name)
            del self.assistant_bot.assistant
            try:
                torch.cuda.empty_cache()
            except Exception as ex:
                print(ex)
            self.assistant_bot.max_new_tokens = max_tokens
            # stop_list = stop_strings.split(",") if stop_strings else []
            self.assistant_bot.stop_strings=None if "nvidia" not in model_name else ["<extra_id_1>"]
            self.assistant_bot.load_model(
                        model_name,
                        max_new_tokens=max_tokens, 
                        device_map=self.assistant_bot.device_map, 
                        torch_dtype=self.assistant_bot.torch_dtype, 
                        use_remote=self.assistant_bot.use_remote)
                                                   
                                                   # )
            # stop_list = stop_strings.split(",") if stop_strings else []
            outstr = (f"Model '{model_name}' loaded successfully!\n"
                    f"Max Tokens: {max_tokens}\n"
                    f"Stop Strings: {self.assistant_bot.stop_strings}")
            return outstr
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def make_base_amas_mpga_app_v2(self,
                theme=gr.themes.Default, 
                view_header_main=view_header,
                view_header_rag=view_header,

                view_header4=view_header,
                chatbox1_header1="",
                chatbox1_height=500,
                chatbox2_height=500,
                inputbox_label="Message to Assistant",
                button_scale=1,
                dropdown_label="# Example User Inputs",
                dropdown_options=['o1', 'o2'],
                
                rag_format_string=base_rag_format_string,


                help_markdown1: str=help_markdown_base,
                help_markdown2: str=help_ui_tab_md1,
                help_markdown3: str=help_ui_tab_md2,
                help_img1: str="./utilities/images/amas_ui_help_img_tab1.png",
                help_img2: str="./utilities/images/amas_ui_help_img_tab2.png",
                help_img3: str="./utilities/images/amas_ui_help_img_tab_settings.png",

                tab_1_label="Assistant Chat",
                tab_2_label="Assistant Chat & Retrieval View",
                tab_3_label="Help",
                tab_4_label="System Generation Settings",
                
                max_temp=2.8,
                temp_label="'Creativity': 0 := most probable next word; max:= random text generation",
                js_code=js_code,
                custom_css=custom_css,
                k=2,
                min_score=.8,
                reverse=False,
                mode="min",
                                   
                **kwargs,
        ):
        self.k=k
        self.min_score=min_score
        self.reverse=reverse
        self.mode=mode
        self.rag_premable=rag_format_string
        rag_tk_slider=None
        rag_sim_lb=None
        reverse_checkbox=None
        convo_text = None
        download_button=None
        file_output=None
        upload_file = None
        output_text = None
        download_button2=None
        file_output2=None
        upload_file2 = None
        output_text2 = None
        load_model_button = None
        model_dropdown = None
        load_model_button2 = None
        model_dropdown2 = None
        max_tokens_box = None
        stop_strings_box = None
        output_model_text = None
        max_tokens_box2 = None
        stop_strings_box2 = None
        output_model_text2 = None
        with gr.Blocks(theme=theme, js=js_code, css=custom_css) as self.assistant_app:
            # main tab
            with gr.Tab(label=tab_1_label):
                with gr.Row():
                    # with gr.Column(scale=1):
                    #     with gr.Accordion("Conversation Historian", open=False):
                    #         with gr.Accordion("Download Conversation", open=False):
                    #             with gr.Row():
                    #                 download_button = gr.Button("Download Conversation", scale=1, elem_classes=['my-button-a'])
                    #             with gr.Row():
                    #                 # A File component that will display the downloadable file link
                    #                 file_output = gr.File(label="Download your conversation", scale=1, elem_classes=['my-file'])
                    #         with gr.Accordion("Upload Conversation from JSON", open=False):  
                    #             with gr.Row():
                    #                 upload_file = gr.File(label="Upload Conversation JSON", 
                    #                                       file_count="single", type="filepath", 
                    #                                       scale=1, elem_classes=['my-file'])
                    #         with gr.Accordion("Model Selection", open=False):  
                    #             with gr.Row():
                    #                 model_dropdown = gr.Dropdown(choices=conversational_models, label="Select Model")
                    #                 max_tokens_box = gr.Number(value=4080, label="Max Tokens", precision=0)
                    #                 # stop_strings_box = gr.Textbox(label="Stop Strings (comma-separated)")
                                    
                    #                 load_model_button = gr.Button("Load Model")
                    #                 output_model_text = gr.Textbox(label="Status", interactive=False)
                    # main column
                    with gr.Column(scale=10):
                        with gr.Row():
                            markdown_obj = gr.Markdown(view_header_main)
                        # chat box row
                        with gr.Row():
                            
                            self.assistant_chatbot = gr.Chatbot(scale=10, 
                                                             label="",
                                                             height=chatbox1_height,
                                                             elem_classes=["chatbot"],
                                                             type='messages')   # Chat Dialog box
                        # input box row
                        with gr.Row():
                            # set up user input text box
                            self.input_text = gr.Textbox(label=inputbox_label,
                                                         value="", )
                        # submit row
                        with gr.Row():
                            self.clear_button = gr.Button(
                                "Clear Input",
                                scale=button_scale,
                            )
        
                            self.submit_button = gr.Button(
                                "Submit", variant="secondary",
                                scale=button_scale,  elem_classes=["submit-button"], elem_id="submit-btn", 
                            )
        
                            self.reset_button = gr.Button(
                                "Clear Conversation",
                                scale=button_scale,
                            )
                        
                        
                        # dropdowns row
                        with gr.Row():
                            self.dropdown_1 = gr.Dropdown(
                                label=dropdown_label, 
                                choices=dropdown_options, value=dropdown_options[0], interactive=True,
                            )
                        
                        with gr.Row():
                            self.dropdown_button1=gr.Button(
                                "Copy-Example",
                                scale=button_scale,
                            )
                # converation history logging/loading 
                with gr.Row():
                     with gr.Column(scale=1):
                        with gr.Accordion("Conversation Historian", open=False):
                            with gr.Accordion("Download Conversation", open=False):
                                with gr.Row():
                                    download_button = gr.Button("Download Conversation", scale=1, elem_classes=['my-button-a'])
                                with gr.Row():
                                    # A File component that will display the downloadable file link
                                    file_output = gr.File(label="Download your conversation", scale=1, elem_classes=['my-file'])
                            with gr.Accordion("Upload Conversation from JSON", open=False):  
                                with gr.Row():
                                    upload_file = gr.File(label="Upload Conversation JSON", 
                                                          file_count="single", type="filepath", 
                                                          scale=1, elem_classes=['my-file'])
                            # Model section
                            with gr.Accordion("Model Selection", open=False):  
                                with gr.Row():
                                    model_dropdown = gr.Dropdown(choices=conversational_models, label="Select Model")
                                    max_tokens_box = gr.Number(value=4080, label="Max Tokens", precision=0)
                                    # stop_strings_box = gr.Textbox(label="Stop Strings (comma-separated)")
                                    
                                    load_model_button = gr.Button("Load Model")
                                    output_model_text = gr.Textbox(label="Status", interactive=False)        
            
            
            
            ##################################################
            # Chat, RAG view tab
            with gr.Tab(label=tab_2_label):
                with gr.Row():
                    markdown_obj2 = gr.Markdown(view_header_rag)
                with gr.Row():
                    self.assistant_chatbot2 = gr.Chatbot(scale=1, 
                                                         label="Chat-log",
                                                         height=chatbox2_height,
                                                         elem_classes=["chatbot"],
                                                         type='messages')   # Chat Dialog box
                    self.rag_chatbot = gr.Chatbot(scale=1, 
                                                label="Input vs Retrieved Knowlege",
                                                height=chatbox2_height,
                                                elem_classes=["chatbot"],
                                                type='messages')   # Chat Dialog box
                with gr.Row(elem_id="input-row", equal_height=False):
                    # set up user input text box
                    # with gr.Column(scale=9, elem_id="input-row"):
                    self.input_text2 = gr.Textbox(label=inputbox_label,
                                                 value="", scale=10)
                
                    self.submit_button2 = gr.Button(
                        "Submit",variant="secondary", size='md',
                        scale=1,  elem_classes=["submit-button"], #elem_id="submit-btn",
                    )
                with gr.Row():
                    # with gr.Column(scale=1, elem_id="input-row"):
                        # with gr.Row():
                        #     self.submit_button2 = gr.Button(
                        #         "Submit",variant="secondary",
                        #         scale=1,  elem_classes=["submit-button"], #elem_id="submit-btn",
                        #     )   
                        # with gr.Row():
                    self.clear_button2 = gr.Button(
                        "Clear Input", size='sm',
                        scale=1,
                    )

                    # self.submit_button2 = gr.Button(
                    #     "Submit",variant="secondary",
                    #             scale=button_scale,  elem_classes=["submit-button"], #elem_id="submit-btn",
                    # )
                        # with gr.Row():
                    self.reset_button2 = gr.Button(
                        "Clear Conversation",  size='sm',
                        scale=1,
                    )
                
                # dropdowns row
                with gr.Row():
                    self.dropdown_2 = gr.Dropdown(
                        label=dropdown_label, 
                        choices=dropdown_options, value=dropdown_options[0], interactive=True,
                    )
                
                with gr.Row():
                    self.dropdown_button2=gr.Button(
                        "Copy-Example",
                        scale=button_scale,
                    )            
                # converation history logging/loading 
                with gr.Row():
                     with gr.Column(scale=1):
                        with gr.Accordion("Conversation Historian", open=False):
                            with gr.Accordion("Download Conversation", open=False):
                                with gr.Row():
                                    download_button2 = gr.Button("Download Conversation", scale=1, elem_classes=['my-button-a'])
                                with gr.Row():
                                    # A File component that will display the downloadable file link
                                    file_output2 = gr.File(label="Download your conversation", scale=1, elem_classes=['my-file'])
                            with gr.Accordion("Upload Conversation from JSON", open=False):  
                                with gr.Row():
                                    upload_file2 = gr.File(label="Upload Conversation JSON", 
                                                          file_count="single", type="filepath", 
                                                          scale=1, elem_classes=['my-file'])
                            # Model section
                            with gr.Accordion("Model Selection", open=False):  
                                with gr.Row():
                                    model_dropdown2 = gr.Dropdown(choices=conversational_models, label="Select Model")
                                    max_tokens_box2 = gr.Number(value=4080, label="Max Tokens", precision=0)
                                    # stop_strings_box = gr.Textbox(label="Stop Strings (comma-separated)")
                                    
                                    load_model_button2 = gr.Button("Load Model")
                                    output_model_text2 = gr.Textbox(label="Status", interactive=False)
            
            
            
            # make help tab to inform user of what to do if they get confused
            with gr.Tab(label=tab_3_label):
                if help_img1:
                    with gr.Row():
                        help_markdown = gr.Markdown(help_markdown1, elem_classes=["markdown-text"])
                    with gr.Row():
                        img_array = plt.imread(help_img1)
                        gr.Image(img_array)
                    
                if help_img2:
                    with gr.Row():
                        ui_tab_2_md_msg = gr.Markdown(help_markdown2, elem_classes=["markdown-text"])
                    with gr.Row():
                        # img_array2 = plt.imread(help_img2)
                        gr.Image(help_img2)
                    
                if help_img3:
                    with gr.Row():
                        ui_tab_3_md_msg = gr.Markdown(help_markdown3, elem_classes=["markdown-text"])
                    with gr.Row():
                        # img_array3 = plt.imread(help_img3)
                        gr.Image(help_img3)
                else:
                    print("\n\n\n\t\t\tno image 3 to set\n\n\n")
            
            # Settings Tab
            with gr.Tab(label=tab_4_label):
            #     generation_settings_row1 = gr.Row()  # temp slider
            #     tk_tp_mtkn_row = gr.Row()     # set sliders
            #     # generation_settings_row2 = gr.Row()  # MODE selector radio (Base, RAG, GRAG)
            #     system_directives_row1 = gr.Row()    # input box, 
            #     system_directives_row2 = gr.Row()    # submit directive button
                
                with gr.Row():
                    self.temp_slider = gr.Slider(label=temp_label , 
                                            minimum=0, maximum=max_temp, 
                                            value=self.assistant_bot.temp, step=.01)
    
                with gr.Row():
                    self.tk_slider = gr.Slider(label="Top # of MLE" , 
                                            minimum=10, maximum=60, 
                                            value=self.assistant_bot.top_k if self.assistant_bot.top_k else 10, 
                                            step=5)
                    self.tp_slider = gr.Slider(label="Top % of MLE" , 
                                            minimum=0.10, maximum=.70, 
                                            value=self.assistant_bot.top_p if self.assistant_bot.top_p else 0.1,
                                            interactive=True,
                                            step=.05)
                    self.max_new_token_slider = gr.Slider(label="Max New Tokens", 
                                            minimum=400, maximum=8000, 
                                            value=self.assistant_bot.max_new_tokens if self.assistant_bot.max_new_tokens else 400,
                                            interactive=True,
                                            step=100)
                # with generation_settings_row2:
                with gr.Row():
                    self.system_dir_text = gr.Textbox(label="System Directives:",
                                         value="", )
                # with generation_settings_row2:
                with gr.Row():
                    self.system_dir_button = gr.Button("System", scale=1)
                with gr.Row():
                    rag_markdown_header_obj = gr.Markdown("RAG Settings", elem_classes=["markdown-text"])
                with gr.Row():
                    # top K
                    rag_tk_slider = gr.Slider(label="Top K Similiar Docs" , 
                                            minimum=1, maximum=5, 
                                            value=self.k if self.k else 2, 
                                            step=1)
                with gr.Row():
                    # top K
                    rag_sim_lb = gr.Slider(label="Cosine Similiar Threshold" , 
                                            minimum=.4, maximum=.9, 
                                            value=self.min_score, 
                                            step=.1)
                with gr.Row():
                    # reverse of listings
                    reverse_checkbox = gr.Checkbox(label="Reverse Rag Rankings", value=self.reverse)
            
            
            
            
            
            ######################################################################
            #                                       set up event handlers
            ##### User input handling
            self.input_text.submit(self.generate_response_dual, 
                                   inputs=[self.input_text,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])
            self.submit_button.click(self.generate_response_dual, 
                                    inputs=[self.input_text,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])


            #### clear/reset button logic
            self.clear_button.click(self.clear_chat, outputs=[self.input_text])
            # self.reset_button.click(self.reset_conversation)
            self.reset_button.click(self.clear_conversation_dual, outputs=[self.assistant_chatbot, self.assistant_chatbot2])

            # drop down logic on tab 1
            self.dropdown_button1.click(self.copy_text, inputs=self.dropdown_1, outputs=self.input_text)
            
            download_button.click(fn=self.save_conversation_to_json, inputs=[], outputs=file_output) 
            
            # load_conversation_from_json
            upload_file.upload(fn=self.load_conversation_from_json_dual, 
                               inputs=[self.assistant_chatbot, self.assistant_chatbot2, upload_file], 
                               outputs=[self.assistant_chatbot, self.assistant_chatbot2])
            load_model_button.click(
                          fn=self.load_model_func, 
                          inputs=[model_dropdown, max_tokens_box], 
                          outputs=[output_model_text])
            ##### User input handling
            self.input_text2.submit(self.generate_response_dual, 
                                    inputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])
            self.submit_button2.click(self.generate_response_dual, 
                                    inputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])

            #### clear/reset button logic
            self.clear_button2.click(self.clear_chat, outputs=[self.input_text2])
            # self.reset_button2.click(self.reset_conversation)
            self.reset_button2.click(self.clear_conversation_dual, outputs=[self.assistant_chatbot, self.assistant_chatbot2])
            
            self.dropdown_button2.click(self.copy_text, inputs=self.dropdown_2, outputs=self.input_text2)
            
            download_button2.click(fn=self.save_conversation_to_json, inputs=[], outputs=file_output2) 
            # load_conversation_from_json
            upload_file2.upload(fn=self.load_conversation_from_json_dual, 
                               inputs=[self.assistant_chatbot, self.assistant_chatbot2, upload_file2], 
                               outputs=[self.assistant_chatbot, self.assistant_chatbot2])
            load_model_button2.click(
                          fn=self.load_model_func, 
                          inputs=[model_dropdown2, max_tokens_box2], 
                          outputs=[output_model_text2])
            #### generation settings logic
            self.temp_slider.change(self.assistant_bot.update_temperature, inputs=self.temp_slider)
            self.tp_slider.change(self.assistant_bot.update_top_p, inputs=self.tp_slider)
            self.tk_slider.change(self.assistant_bot.update_top_k, inputs=self.tk_slider)
            self.max_new_token_slider.change(self.assistant_bot.update_max_new_tokens, inputs=self.max_new_token_slider)
            
            self.system_dir_text.submit(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])
            self.system_dir_button.click(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])

            rag_tk_slider.change(self.update_rag_k, inputs=[rag_tk_slider])
            rag_sim_lb.change(self.update_cosine_threshold, inputs=[rag_sim_lb])
            reverse_checkbox.change(self.update_rag_ranking_reverse, inputs=[reverse_checkbox])
        
        return
    
    def make_chat_comp_settings_app(self,
                theme=gr.themes.Default, 
                user_greeting="Greetings!", 
                chat_bot_label="Chat", 
                max_temp=2.8,
                tab_1_label="# Chat",
                tab_2_label="# Chat & Retrieval View",
                tab_3_label="# System Settings",
                header_md_1=None,
                header_md_2=None,
                header_md_3=None,
                chat_md_header_1="Chat with assistant {} below", 
                help_markdown_message=None,
                chatbox_height=450,
                button_scale=1,
                k=2,
                min_score=.0001,
                reverse=False,
                mode="min",
                
                rag_format_string=base_rag_format_string,
                options1=sow_prompt_options_1,
                options2=sow_prompt_options_1,
                options3=sow_prompt_options_1,
                options4=sow_prompt_options_1,
                **kwargs,
        ):
        self.k=k
        self.min_score=min_score
        self.reverse=reverse
        self.mode=mode
        self.rag_premable=rag_format_string
        with gr.Blocks(theme=theme, js=js_code, css=custom_css) as self.assistant_app:
            
            # make basic chat UI Tab
            ################################
            ##       Chatbox UI           ##
            ################################
            ################################
            ##       user input box       ##
            ################################
            ################################
            ##       user input buttons   ##
            ################################
            with gr.Tab(label=tab_1_label):
                # header_row = gr.Row()
                # chatbox_row = gr.Row()
                # input_row = gr.Row()
                # sbutton_row = gr.Row()
                # dropdown_row = gr.Row()
                # dropdown_btn_row = gr.Row()
                # # first row should be the greeting/header
                with gr.Row():
                    markdown_header1=(
                        f"## Chat with {self.name} in the user input area below..."
                    ) if not header_md_1 else header_md_1
                    markdown_obj = gr.Markdown(markdown_header1)
                # chat box row
                with gr.Row():
                    markdown_cb_header1=(
                            f"## Chat with {self.assistant_bot.name} in the user input area below..."
                        ) if not chat_md_header_1 else chat_md_header_1
                    print(markdown_cb_header1)
                    self.assistant_chatbot = gr.Chatbot(scale=1, 
                                                     label=markdown_cb_header1,
                                                     height=chatbox_height,
                                                     elem_classes=["chatbot"],
                                                     type='messages')   # Chat Dialog box
                # input box row
                with gr.Row():
                    # set up user input text box
                    self.input_text = gr.Textbox(label="User Prompt:",
                                                 value="", )
                # submit row
                with gr.Row():
                    self.clear_button = gr.Button(
                        "Clear Input",
                        scale=button_scale,
                    )

                    self.submit_button = gr.Button(
                        "Submit",
                        scale=button_scale,
                    )

                    self.reset_button = gr.Button(
                        "Clear Conversation",
                        scale=button_scale,
                    )
                
                # dropdowns row
                with gr.Row():
                    self.dropdown_1 = gr.Dropdown(
                        label="Example input questions", 
                        choices=options1, value=options1[0], interactive=True,
                    )
                
                with gr.Row():
                    self.dropdown_button1=gr.Button(
                        "Copy-Example",
                        scale=button_scale,
                    )
            # Chat, RAG view tab
            # make tab that has the chatbox, and a view of the 
            # document knowledge retrieved
            with gr.Tab(label=tab_2_label):
                chatbox_row_t2 = gr.Row()
                input_row_t2 = gr.Row()
                sbuttons_row2 = gr.Row()
                markdown_cb_header1=(
                            f"## Chat Log with {self.assistant_bot.name}"
                        ) 
                # print(markdown_cb_header1)
                # make the two chatboxes.
                # the one on the left is just like the one on 
                # the first tab while the second box is a view 
                # that shows the user query, and the retrieved 
                # and annotated knowledge returned to the assistant
                with gr.Row():
                    self.assistant_chatbot2 = gr.Chatbot(scale=1, 
                                                         label=markdown_cb_header1 + "1",
                                                         height=chatbox_height,
                                                         elem_classes=["chatbot"],
                                                         type='messages')   # Chat Dialog box
                    self.rag_chatbot = gr.Chatbot(scale=1, 
                                                label="Input vs Retrieved Knowlege",
                                                height=chatbox_height,
                                                elem_classes=["chatbot"],
                                                type='messages')   # Chat Dialog box
                with gr.Row():
                    # set up user input text box
                    self.input_text2 = gr.Textbox(label="User Prompt:",
                                                 value="", )
                with gr.Row():
                    self.clear_button2 = gr.Button(
                        "Clear Input",
                        scale=button_scale,
                    )

                    self.submit_button2 = gr.Button(
                        "Submit",
                        scale=button_scale,
                    )

                    self.reset_button2 = gr.Button(
                        "Clear Conversation",
                        scale=button_scale,
                    )
                
                # dropdowns row
                with gr.Row():
                    self.dropdown_2 = gr.Dropdown(
                        label="Example input questions", 
                        choices=options1, value=options1[0], interactive=True,
                    )
                
                with gr.Row():
                    self.dropdown_button2=gr.Button(
                        "Copy-Example",
                        scale=button_scale,
                    )            
            
            # make help tab to inform user of what to do if they get confused
            with gr.Tab(label="Help"):
                with gr.Row():
                    help_markdown_base = """
                    > This chat assistant is intended to demonstrate some basic rag capabilities for testing of the MLACT project. 
                    > There are three main tabs. The first is the main chat area where you can enter your messages to the assistant
                    > in the text box below the main chat window and either hit enter when ready or click the submit button. 
                    > The clear-chat button will erase your current user input, and the clear conversation button will clear the chat window. 
                    >      The second tab is a dual view of the chat history as seen in the first tab, along with a view of the information 
                    > the RAG system is retrieving for the assitant to use on the right most chat window. The interface for user input is the
                    > same, but this view allows you to see what the assistant is seeing and even instruct them to utilize different information 
                    > that they have negelected to use. 
                    >      The third tab is this very tab and is intended to serve as a user guide below are annotated images of the different
                    > tab UIs and what each component does. 
                    >      The fourth and final tab is a tab for generation parameter settings. The temperature slider controls the model 
                    > generation temperature or "randomness". The higher the value the more "creative" the assistant will get when generating 
                    > text. The top_k box will inform the assistant what the top K next tokens to choose from should be, and the top_p box will
                    > control what the lowest probability of a token being the next one will be. Different K values lead to the assistant 
                    > choosing from the top K tokens while the P values will see the assistant chose from only those assistants above a 
                    > probability of P being the next correct token. 
                    """
                    
                    help_markdown_text = help_markdown_base if not help_markdown_message else help_markdown_message
                    help_markdown = gr.Markdown(help_markdown_text, elem_classes=["markdown-text"])
                with gr.Row():
                    ui_tab_md = """
                    # Main Chat Tab Help
                    
                    The below image is an annotated image of the first (main) chat tab. It highlights the different components on the page and their function.
                    The main chat window will display the chat log of messages between you (user) and the assistant (Assistant Response/Chat History). You can enter you messages in the "User Input Area" in the User prompt box. You can clear any input in the box with the clear-input button, submit the input by pressing enter or the submit button, and clear the chat log with the clear conversation buttons. Below the user input area, the "example Input Options" area provides a dropdown list of all of the various tasks stored in the vector store used to retrieve any input related text. You can look through these for examples of the types of tasks the assistant can use to inform new tasks based on old ones. You can also just make things up and see how well it can make reasonable hazards and controls based on it's domain knowledge. 
                    """
                    ui_tab_1_md_msg = gr.Markdown(ui_tab_md, elem_classes=["markdown-text"])
                with gr.Row():
                    
                    image_tab1_path = "./utilities/images/amas_ui_help_img_tab1.png"
                    
                    img_array = plt.imread(image_tab1_path)
                    print(img_array)
                    gr.Image(img_array)
                with gr.Row():
                    ui_tab_md2="""
                    # Retrieval Chat Tab
                    The second tab allows you to see chat history between you and the assistant on the left chat
                    log, and the view of your input and the information pulled from the RAG system for comparison
                    on the right chat log. This allow you to see the original informaiton and even suggest changes
                    to the assistant based on that. The rest of the tab functions just like the first tab. 
                    """
                    ui_tab_2_md_msg = gr.Markdown(ui_tab_md2, elem_classes=["markdown-text"])
                with gr.Row():
                    image_tab2_path = "./utilities/images/amas_ui_help_img_tab2.png"
                    img_array2 = plt.imread(image_tab2_path)
                    gr.Image(img_array2)
            
            # Settings Tab
            with gr.Tab(label=tab_3_label):
            #     generation_settings_row1 = gr.Row()  # temp slider
            #     tk_tp_mtkn_row = gr.Row()     # set sliders
            #     # generation_settings_row2 = gr.Row()  # MODE selector radio (Base, RAG, GRAG)
            #     system_directives_row1 = gr.Row()    # input box, 
            #     system_directives_row2 = gr.Row()    # submit directive button
                
                with gr.Row():
                    self.temp_slider = gr.Slider(label="Temperature: higher => more 'RNG'" , 
                                            minimum=0, maximum=max_temp, 
                                            value=self.assistant_bot.temp, step=.01)
    
                with gr.Row():
                    self.tk_slider = gr.Slider(label="Top # of MLE" , 
                                            minimum=10, maximum=60, 
                                            value=self.assistant_bot.top_k if self.assistant_bot.top_k else 10, 
                                            step=5)
                    self.tp_slider = gr.Slider(label="Top % of MLE" , 
                                            minimum=0.10, maximum=.70, 
                                            value=self.assistant_bot.top_p if self.assistant_bot.top_p else 0.1,
                                            interactive=True,
                                            step=.05)
                    self.max_new_token_slider = gr.Slider(label="Max New Tokens", 
                                            minimum=400, maximum=8000, 
                                            value=self.assistant_bot.max_new_tokens if self.assistant_bot.max_new_tokens else 400,
                                            interactive=True,
                                            step=100)
                # with generation_settings_row2:
                with gr.Row():
                    self.system_dir_text = gr.Textbox(label="System Directives:",
                                         value="", )
                # with generation_settings_row2:
                with gr.Row():
                    self.system_dir_button = gr.Button("System", scale=1)
            ######################################################################
            #                                       set up event handlers
            ##### User input handling
            self.input_text.submit(self.generate_response_dual, 
                                   inputs=[self.input_text,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])
            self.submit_button.click(self.generate_response_dual, 
                                    inputs=[self.input_text,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])


            #### clear/reset button logic
            self.clear_button.click(self.clear_chat, outputs=[self.input_text])
            # self.reset_button.click(self.reset_conversation)
            self.reset_button.click(self.clear_conversation_dual, outputs=[self.assistant_chatbot, self.assistant_chatbot2])

            # drop down logic on tab 1
            self.dropdown_button1.click(self.copy_text, inputs=self.dropdown_1, outputs=self.input_text)
            
            
            ##### User input handling
            self.input_text2.submit(self.generate_response_dual, 
                                    inputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])
            self.submit_button2.click(self.generate_response_dual, 
                                    inputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
                                   outputs=[self.input_text2,
                                           self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])

            #### clear/reset button logic
            self.clear_button2.click(self.clear_chat, outputs=[self.input_text2])
            # self.reset_button2.click(self.reset_conversation)
            self.reset_button.click(self.clear_conversation_dual, outputs=[self.assistant_chatbot, self.assistant_chatbot2])
            
            self.dropdown_button2.click(self.copy_text, inputs=self.dropdown_2, outputs=self.input_text2)
            

            #### generation settings logic
            self.temp_slider.change(self.assistant_bot.update_temperature, inputs=self.temp_slider)
            self.tp_slider.change(self.assistant_bot.update_top_p, inputs=self.tp_slider)
            self.tk_slider.change(self.assistant_bot.update_top_k, inputs=self.tk_slider)
            self.max_new_token_slider.change(self.assistant_bot.update_max_new_tokens, inputs=self.max_new_token_slider)
            
            self.system_dir_text.submit(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])
            self.system_dir_button.click(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])

   
        
    
    def make_basic_chatapp(self, 
                           theme=gr.themes.Default, 
                           user_greeting="Greetings!", 
                           chat_bot_label="Chat", 
                           max_temp=2.8,
                           tab_1_label="Chat",
                           tab_2_label="Settings",
                           chatbox_height=450,
                           button_scale=1,
                           **kwargs,
                           ):

        tab_chat = gr.Tab(label=tab_1_label)
        tab_settings =  gr.Tab(label=tab_2_label)
        
        with gr.Blocks(theme=theme) as self.assistant_app:
            

            with  gr.Tab(label=tab_1_label):
                greetings_row = gr.Row()
                chat_row = gr.Row()
                input_row = gr.Row()
                buttons_row = gr.Row()
                with greetings_row:
                    greetings_preamble = gr.Markdown(user_greeting)
                
                with chat_row:
                    self.assistant_chatbot = gr.Chatbot(scale=1, 
                                                     label=chat_bot_label,
                                                     height=chatbox_height,
                                                     type='messages')   # Chat Dialog box
    
                with input_row:
                    self.input_text = gr.Textbox(label="User Prompt:",
                                                 value="Start chatting here...", )
                with buttons_row:
                    self.submit_button = gr.Button("Submit", scale=1)        # user input submission
                    self.clear_button = gr.Button("Clear", scale=1)          # clear user input
                    self.reset_button = gr.Button("Reset-Memory", scale=1)   # reset conversation history
            
            with gr.Tab(label=tab_2_label):
                generation_settings_row1 = gr.Row()  # temp slider
                # generation_settings_row2 = gr.Row()  # MODE selector radio (Base, RAG, GRAG)
                # generation_settings_row3 = gr.Row()  # top-k, top-p, 
                system_directives_row1 = gr.Row()    # input box, 
                system_directives_row2 = gr.Row()    # submit directive button
                
                with generation_settings_row1:
                    self.temp_slider = gr.Slider(label="Temperature: higher==more 'random'" , 
                                            minimum=0, maximum=max_temp, 
                                            value=self.assistant_bot.temp, step=.001)
    
                # with generation_settings_row2:
                with system_directives_row1:
                    self.system_dir_text = gr.Textbox(label="System Directives:",
                                         value="", )
                # with generation_settings_row2:
                with system_directives_row2:
                    self.system_dir_button = gr.Button("System", scale=1)
            # # make the settings tab for generatino and adding system inputs
            # temp_system_dir_objs = self.create_basic_settings_tab(
            #     tab_settings, init_temp=self.assistant_bot.temp, 
            #     max_temp=max_temp, 
            #     options_list=["A"], 
            #     drop_down1_label="Knowledge-Nexus Options",
            # )
            # with tab_settings:
                
            # self.temp_slider = temp_system_dir_objs[0]
            # self.system_dir_text = temp_system_dir_objs[1]
            # self.system_dir_button = temp_system_dir_objs[2]

            #                                       set up event handlers
            ##### User input handling
            self.input_text.submit(self.generate_response, 
                                   inputs=[self.input_text, self.assistant_chatbot], 
                                   outputs=[self.input_text, self.assistant_chatbot])
            self.submit_button.click(self.generate_response, 
                                    inputs=[self.input_text, self.assistant_chatbot], 
                                    outputs=[self.input_text, self.assistant_chatbot])


            #### clear/reset button logic
            self.clear_button.click(self.clear_chat, outputs=[self.input_text])
            self.reset_button.click(self.reset_conversation)

            #### generation settings logic
            self.temp_slider.change(self.assistant_bot.update_temperature, inputs=self.temp_slider)
            self.system_dir_text.submit(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])
            self.system_dir_button.click(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])
            
    
    def launch_app(self, share=False, debug=False, server_port=7860, server_name=None, system_directive: str=None, 
                   save_context: bool = False, use_system_role: bool=True,
                  ):
        
        # print(f"2----> save context: {save_context}")
        self.save_context=save_context
        
        # Give assistant initial directive
        if system_directive:
            if use_system_role:
                self.init_system_directive = self.assistant_bot.system(system_directive)
            else:
                print("Using user role instead!!!!")
                self.init_system_directive = self.assistant_bot.user(system_directive)
            print(f"Initial Directive:\n{self.init_system_directive}\n")
        self.conversation = [self.init_system_directive]
        self.directives_given = 0
        
        if not server_name:
            return self.assistant_app.launch(share=share, debug=debug, server_port=server_port)
        else:
            return self.assistant_app.launch(share=share, debug=debug, server_port=server_port, server_name=server_name)
    
    def start_app(self, share=False, debug=False, server_port=7860, server_name=None, system_directive: str=None,
                  save_context: bool=False,
                  ):
        self.save_context=save_context
        self.server = self.launch_app(share=share, debug=debug, 
                        server_port=server_port, 
                        server_name=server_name, system_directive=system_directive, save_context=save_context)
        return self.server
    
    
    def kill_app(self):
        """Stop currently running app referenced by the assistant_app variable"""
        self.assistant_app.close()
        sys.exit()
    
    @staticmethod
    def copy_text(selected_text):
        """Used by drop down button to copy text into user input box"""
        return selected_text
    
    # Function to handle signals
    def signal_handler(self, sig, frame, demo):
        """Function to handle signals intended to stop a running script so 
           that the app can close the port it is using correctly.
        """
        print(f"\n\nReceived signal: {sig}\nClosing Gradio app.")
        self.kill_app()  

class RAGWorkshopUI(BasicGradioAssistant):
    def __init__(self, assistant_bot, 
                 uname="user",
                 embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2",
                 current_store = "your domain",
                 **kwargs):
        super().__init__(assistant_bot=assistant_bot, 
                         uname=uname,
                         **kwargs)
        self.VECTOR_STORE_LIST = list()
        self.embedding_model_name = embedding_model_name
        self.assitant_model_name = ""
        self.chunk_size = 500000
        self.chunk_overlap = 200
        self.k=3
        self.min_score=20
        self.VECTOR_STORE_LIST = [current_store,]
        

    def load_llm(self, llm_name, **kwargs):
        self.load_assistant_bot(llm_name, **kwargs)
        return f"LLM loaded: {llm_name}"
    
    def load_assistant_bot(self, llm_name, **kwargs):
        """
            Can be used to load an new LLM assistant based on the provided type
        """
        knexus_mngr = self.assistant_bot.knexus_mngr
        self.assistant_bot = AMAS_RAG_Assistant(llm_name, 
                                                hf_login=False, 
                                                top_k=self.assistant_bot.top_k,
                                                top_p=self.assistant_bot.top_p,
                                                temperature=self.assistant_bot.temp,
                                                max_new_tokens=self.assistant_bot.max_new_tokens,
                                                max_tokens=self.assistant_bot.max_tokens,
                                                **kwargs)
        self.assistant_bot.knexus_mngr = knexus_mngr

    def load_vectorstore(self, name):
        self.assistant_bot.knexus_mngr.load_nexus_manager(name)
        return f"Loaded Vector Store: {name}"

    
    
    
    def create_vectorstore_from_pdfs(self, file_name, 
                                    embedding_model_name=None,
                                    chunk_size=None, chunk_overlap=None):
        pdf_paths = [file_name]
        if chunk_size:
            self.chunk_size=chunk_size
        if chunk_overlap:
            self.chunk_overlap=chunk_overlap
        if embedding_model_name:
            self.embedding_model_name = embedding_model_name
        self.assistant_bot.knexus_mngr.process_pdfs_to_vector_store_and_embeddings( 
                            pdf_paths, 
                            embedding_model_name=self.embedding_model_name, 
                            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, 
                        )

    def query_similarity_search(self, query, k=2, min_simuliarity=200):
        docs, scores = self.assistant_bot.knexus_mngr.query_similarity_search(
                                query=query, 
                                k=k, 
                                min_score=min_score, 
                                reverse=False, 
                                # verbose=True,
                                verbose=False,
                               )
        return docs, scores

    def show_docs_scores(self, docs, scores):
        for d,s in zip(docs, scores):
            print(f"Similarity Score: {s}\n")
            print(f"Document Chunk:\n{d}\n>>>>>>>>>>>>>>>>>>>>>>\n\n")
    
    def upload_and_create_vectorstore(self, file):
        new_vs_name = self.create_vectorstore_from_pdfs(file.name)
        if new_vs_name not in self.VECTOR_STORE_LIST:
            self.VECTOR_STORE_LIST.append(new_vs_name)
        return gr.update(choices=self.VECTOR_STORE_LIST, value=new_vs_name), f"Created new store: {new_vs_name}"

    def update_params(self, chunk_size, k, min_similarity, temperature, top_k, top_p, max_tokens):
        self.chunk_size=chunk_size
        self.assistant_bot.update_top_k(top_k)
        self.assistant_bot.update_top_p(top_p)
        self.assistant_bot.update_temperature(temperature)
        self.k=k
        self.min_score=min_similarity
        self.temp=temperature
        self.top_k=top_k
        self.top_p=top_p
        self.max_tokens=max_tokens
    
    def handle_chat(self, history, user_input, rag_mode, chunk_size, k, min_sim, temperature, top_k, top_p, max_tokens):
        self.update_params(
            chunk_size=chunk_size,
            k=k,
            min_similarity=min_sim,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens
        )

        self.assistant_bot.rag_mode = True if rag_mode == "RAG Enabled" else False
        
        response = self.assistant_bot.generate_rag_response(user_input)
        # if rag_mode == "RAG Enabled":
        #     answer, debug = self.query(user_input)
        # else:
        #     answer = rag_workshop.base_llm_query(user_input)  # you need to implement this
        #     debug = "RAG Disabled: Answer generated using base LLM only."
        history.append(user(f"## User:\n" + user_input))
        history.append(assistant(f"## Assistant:\n" + response))
        debug = "OK!"
        # history = history + [(user_input, answer)]
        return history, "", debug
    
    def create_base_rag_ui(self,):
        with gr.Blocks() as demo:
            gr.Markdown("# 🤖 RAG Chatbot Playground")
        
            with gr.Row():
                # side panel for loading new LLMs, vector stores, and uploading new files
                with gr.Column(scale=1):
                    with gr.Row():
                        llm_dropdown = gr.Dropdown(label="LLM", choices=LLM_LIST, value=LLM_LIST[0])
                    with gr.Row():
                        rag_mode_radio = gr.Radio(   # ✅ Fixed indent!
                            choices=["RAG Enabled", "RAG Disabled"],
                            value="RAG Disabled",
                            label="RAG Mode"
                        )

                    vs_dropdown = gr.Dropdown(label="Vector Store", choices=self.VECTOR_STORE_LIST, value=self.VECTOR_STORE_LIST[0])
                    upload_pdf = gr.File(label="Upload PDF (Create Vector Store)", file_count="single", type="filepath")
        
                    gr.Markdown("### RAG Parameters")
                    chunk_size = gr.Slider(100, 2000, value=self.chunk_size, step=50, label="Chunk Size")
                    k = gr.Slider(1, 20, value=4, step=1, label="K (Neighbors)")
                    min_sim = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Min Similarity")
                    temperature = gr.Slider(0.0, 1.5, value=self.assistant_bot.temp, step=0.05, label="Temperature")
                    top_k = gr.Slider(1, 50, value=self.assistant_bot.top_k, step=1, label="Top K")
                    top_p = gr.Slider(0.0, 1.0, value=self.assistant_bot.top_p, step=0.01, label="Top P")
                    max_tokens = gr.Slider(50, 1024, value=self.assistant_bot.max_tokens, step=10, label="Max Tokens")
        
                    llm_status = gr.Textbox(label="LLM Status")
                    vs_status = gr.Textbox(label="Vector Store Status")
        
                with gr.Column(scale=2):
                    self.assistant_chatbot = gr.Chatbot(height=450, label="RAG Assistant", type='messages')
                    user_input = gr.Textbox(
                        placeholder="Type your question here & press Enter...",
                        label="Your Prompt",
                        submit_btn=True,
                        show_label=False
                    )
                    debug_output = gr.Textbox(label="Debug Info", interactive=False)
        
            # ---- Events ----
            llm_dropdown.change(self.load_llm, inputs=llm_dropdown, outputs=llm_status)
            vs_dropdown.change(self.load_vectorstore, inputs=vs_dropdown, outputs=vs_status)
            upload_pdf.upload(self.upload_and_create_vectorstore, inputs=upload_pdf, outputs=[vs_dropdown, vs_status])
        
            user_input.submit(
                self.handle_chat,
                inputs=[self.assistant_chatbot, user_input, rag_mode_radio, chunk_size, k, min_sim, temperature, top_k, top_p, max_tokens],
                outputs=[self.assistant_chatbot, user_input, debug_output]
            )
        return demo
    
    def create_base_settings_ui(self, ):
        return

    def create_base_knexus_gen_ui(self,):
        return
        
    def create_app_tabs(self,
                        theme=gr.themes.Default(),
                        css=None,
                        ):
        chat_rag_ui = self.create_base_rag_ui()
        # chat_settings_ui = self.create_base_settings_ui()
        # rag_generation_ui = self.create_base_knexus_gen_ui()
        
        self.assistant_app = gr.TabbedInterface(
            [chat_rag_ui],
            ["chat_rag_ui"],
            # [chat_rag_ui, chat_settings_ui, rag_generation_ui], 
            # ["chat_rag_ui", "chat_settings_ui", "rag_generation_ui"],
            theme=theme, css=custom_css)



class DialogBotBasic(BasicGradioAssistant):
    def __init__(self, assistant_bot, 
                 uname="user",
                 stt_model: str="google",
                 tts_model: str="whisper",
                 **kwargs):
        super().__init__(assistant_bot=assistant_bot, 
                         uname=uname,
                         **kwargs)
        self.tts_model=tts_model
        self.stt_model=stt_model
        
    

class GradioSowAssistant(BasicGradioAssistant):
    def __init__(self, assistant_bot, 
                 uname="user",
                 **kwargs):
        super().__init__(assistant_bot=assistant_bot, 
                         uname=uname,
                         **kwargs)

    def load_assistant_bot(self, assistant_type, **kwargs):
        """
            Can be used to load an new LLM assistant based on the provided type
        """
        self.assistant_bot = assistant_type(**kwargs)




class GradioChatAssistant(BasicGradioAssistant):
    def __init__(self, assistant_bot, 
                 uname: str="user",
                 save_context: bool=False, 
                 truncate_conversation:bool=False,
                 **kwargs):
        super().__init__(assistant_bot=assistant_bot, 
                         uname=uname,
                         **kwargs)
        self.save_context=save_context
        print(f"\n\n\n\t\t\tsave context?: {self.save_context}\n\n\n")
        self.truncate_conversation=truncate_conversation
        self.key_dict = {
            "Requester Name:":["NA"],
            "Badge Number:":["NA"],
            "Department:":["NA"],
            "Location:":["NA"],
            "Maintenance Work Location:":["NA"],
            "Reason for Request:":["NA"],
            "Specific Maintenance Request:":["NA"],
            "Physical Asset (FLOC):":["NA"],
            "Maintenance Type:":["NA"],
        }

    def load_assistant_bot(self, assistant_type, **kwargs):
        self.assistant_bot = assistant_type(**kwargs)

    def reset_key_dict(self, ):
        self.key_dict = {
            "Requester Name:":["NA"],
            "Badge Number:":["NA"],
            "Department:":["NA"],
            "Location:":["NA"],
            "Maintenance Work Location:":["NA"],
            "Reason for Request:":["NA"],
            "Specific Maintenance Request:":["NA"],
            "Physical Asset (FLOC):":["NA"],
            "Maintenance Type:":["NA"],
        }

class GradioSimAssistant(GradioChatAssistant):
    def __init__(self, 
                 assistant_bot, 
                 uname: str="user",
                 save_context: bool=False, 
                 truncate_conversation:bool=False,
                 logging_path="", 
                 **kwargs):
        super().__init__(assistant_bot=assistant_bot, 
                         uname=uname,
                         **kwargs)
        self.save_context=save_context
        print(f"\n\n\n\t\t\tsave context?: {self.save_context}\n\n\n")
        self.truncate_conversation=truncate_conversation
    
        

    def generate_response(self, user_input, chat_history):
        self.conversation, response = self.query_assistant(user_input)
        # print(self.conversation)
        print(self.log_process_active)
        if self.log_process_active:
            self.process_response_and_log_request(response, sentinel_string = "Your request has been submitted!")
        chat_history.append({"role": "user", "content": f"# {self.uname}\n" + user_input})
        chat_history.append( {"role": "assistant", "content": f"# {self.assistant_bot.name}\n" + response})
        return None, chat_history

    
    def generate_flood_sim_ui(self, 
            theme: gr.themes=gr.themes.Default,
            view_header: str="Chat View:\nSpeak with the assitant in the user prompt text box below",
            chatbox_header1: str="Chat Log",
            chatbox_header2: str="Retrieved Knowledge",
            chatbox_height: int=450,
            inputbox_label: str="User Prompt:",
            init_temp: float=.7,
            max_temp: float=2.5,
            button_scale: int=1,
            dropdown_label: str="Example input questions",
            dropdown_options: list=["o1", "o2"],
        ):
        with gr.Blocks(theme=theme, js=js_code, css=custom_css) as self.assistant_app:
            with gr.Row():
                # print(markdown_cb_header1)
                # make the two chatboxes.
                # the one on the left is just like the one on 
                # the first tab while the second box is a view 
                # that shows the user query, and the retrieved 
                # and annotated knowledge returned to the assistant
                tab_header = gr.Markdown(view_header, elem_classes=["markdown-text"])
            with gr.Row():
                self.assistant_chatbot2 = gr.Chatbot(scale=1, 
                                                     label=chatbox_header1,
                                                     height=chatbox_height,
                                                     elem_classes=["chatbot"],
                                                     type='messages')   # Chat Dialog box
            with gr.Row():
                # set up user input text box
                self.input_text2 = gr.Textbox(label="User Prompt:",
                                             value="", )
            with gr.Row():
                temp_slider = gr.Slider(label="Temperature: higher==more 'random'" , 
                                        minimum=0, maximum=max_temp, 
                                        value=init_temp, step=.001)

            # with generation_settings_row2:
            with gr.Row():
                system_input = gr.Textbox(label="System Directives:",
                                     value="", )

            # set up the event handlers
            ##### User input handling
            self.input_text2.submit(self.generate_response, 
                                    inputs=[self.input_text2,
                                           self.assistant_chatbot2], 
                                   outputs=[self.input_text2,
                                           self.assistant_chatbot2])
            temp_slider.change(self.assistant_bot.update_temperature, inputs=self.temp_slider)
            system_input.submit(self.give_system_directive, 
                                        inputs=[system_input], 
                                        outputs=[system_input])
            # self.submit_button2.click(self.generate_response_dual, 
            #                         inputs=[self.input_text2,
            #                                self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot], 
            #                        outputs=[self.input_text2,
            #                                self.assistant_chatbot, self.assistant_chatbot2, self.rag_chatbot])

            #### clear/reset button logic
            # self.clear_button2.click(self.clear_chat, outputs=[self.input_text2])
            # self.reset_button.click(self.clear_conversation_dual, outputs=[self.assistant_chatbot2, self.rag_chatbot])
            # self.dropdown_button2.click(self.copy_text, inputs=self.dropdown_2, outputs=self.input_text2)
        
        return

class GradioMraAssistant(GradioChatAssistant):
    def __init__(self, assistant_bot, log_process_active:bool=True, 
                 logging_path="./logged_work_requests/",  # unique to class
                 save_context: bool=False, 
                 truncate_conversation:bool=False,
                 **kwargs):
        super().__init__(assistant_bot=assistant_bot, 
                 save_context=save_context, 
                 truncate_conversation=truncate_conversation,
                 **kwargs)
        self.log_process_active=log_process_active
        self.logging_path=logging_path
        

    def generate_response(self, user_input, chat_history):
        self.conversation, response = self.query_assistant(user_input)
        # print(self.conversation)
        print(self.log_process_active)
        if self.log_process_active:
            self.process_response_and_log_request(response, sentinel_string = "Your request has been submitted!")
        chat_history.append({"role": "user", "content": f"# {self.uname}\n" + user_input})
        chat_history.append( {"role": "assistant", "content": f"# {self.assistant_bot.name}\n" + response})
        return None, chat_history

class GradioMraAssistant2(GradioMraAssistant):
    def __init__(self, assistant_bot, log_process_active:bool=True, 
                 logging_path="./logged_work_requests/",  # unique to class
                 save_context: bool=False, 
                 truncate_conversation:bool=False,
                 **kwargs):
        super().__init__(
            assistant_bot=assistant_bot, 
            log_process_active=log_process_active,
            save_context=save_context, 
            truncate_conversation=truncate_conversation,
            **kwargs,
        )


    def make_requester_chat_app(self,
                theme=gr.themes.Default, 
                user_greeting="Greetings!", 
                chat_bot_label="Chat", 
                max_temp=2.8,
                tab_1_label="Assistant Chat",
                tab_2_label="# Chat & Retrieval View",
                tab_3_label="# System Settings",
                header_md_1=None,
                header_md_2=None,
                header_md_3=None,
                chatbox_height=450,
                button_scale=1,
                k=2,
                min_score=.0001,
                reverse=False,
                mode="min",
                rag_format_string=base_rag_format_string,
                options1=sow_prompt_options_1,
                options2=sow_prompt_options_1,
                options3=sow_prompt_options_1,
                options4=sow_prompt_options_1,
                **kwargs,
        ):
        self.k=k
        self.min_score=min_score
        self.reverse=reverse
        self.mode=mode
        
        self.rag_premable=rag_format_string
        with gr.Blocks(theme=theme) as self.assistant_app:
            
            # make basic chat UI Tab
            ################################
            ##       Chatbox UI           ##
            ################################
            ################################
            ##       user input box       ##
            ################################
            ################################
            ##       user input buttons   ##
            ################################
            with gr.Tab(label=tab_1_label):
                with gr.Row():
                    # make instruction header
                    mra_header = """
                    Hi! This is an initial demo for the Maintenance Requester Assistant (MRA). This app demonstrates the capabilities of the assistant
                    to accurate interact with users, guide them through inputing a good maintenace request, and store the request for planner
                    processing. This preliminary version attempts to prompt the user for the requester name, badge number, the reason for the request,
                    the desired outcome, the FLOC, and type of maintenace.

                    Below is the main chat log showing your input and the assistants outputs. Below the chat log is the area for entering your messages
                    including the text box, a submit button, a button to clear the text box, and a button to clear out the chat history. Below the 
                    submit area there is a dropdown box that has sample inputs to get you started. You can select an option and use the copy-button to
                    copy it into the user input area. The settings tab is more of a developer tool that allows you to alter the LLM generation 
                    parameters.
                    """
                    markdown_header1=(
                        f"## Chat with {self.name} in the user input area below..."
                    ) if not header_md_1 else header_md_1
                    markdown_obj = gr.Markdown(mra_header)
                    
                with gr.Row():
                    markdown_cb_header1=(
                            f"## Chat with {self.assistant_bot.name} in the user input area below..."
                        ) if not header_md_1 else header_md_1
                    print(markdown_cb_header1)
                    self.assistant_chatbot = gr.Chatbot(scale=1, 
                                                     label=markdown_cb_header1,
                                                     height=chatbox_height,
                                                     type='messages')   # Chat Dialog box
                # input box row
                with gr.Row():
                    # set up user input text box
                    self.input_text = gr.Textbox(label="User Prompt:",
                                                 value="", )
                # submit row
                with gr.Row():
                    self.clear_button = gr.Button(
                        "Clear Input",
                        scale=button_scale,
                    )

                    self.submit_button = gr.Button(
                        "Submit",
                        scale=button_scale,
                    )

                    self.reset_button = gr.Button(
                        "Reset Conversation",
                        scale=button_scale,
                    )
                
                # dropdowns row
                with gr.Row():
                    self.dropdown_1 = gr.Dropdown(
                        label="Example input questions", 
                        choices=options1, value=options1[0], interactive=True,
                    )
                
                with gr.Row():
                    self.dropdown_button1=gr.Button(
                        "Copy-Example",
                        scale=button_scale,
                    )

          
            # Settings Tab
            with gr.Tab(label=tab_3_label):
            #     generation_settings_row1 = gr.Row()  # temp slider
            #     tk_tp_mtkn_row = gr.Row()     # set sliders
            #     # generation_settings_row2 = gr.Row()  # MODE selector radio (Base, RAG, GRAG)
            #     system_directives_row1 = gr.Row()    # input box, 
            #     system_directives_row2 = gr.Row()    # submit directive button
                
                with gr.Row():
                    self.temp_slider = gr.Slider(label="Temperature: higher => more 'RNG'" , 
                                            minimum=0, maximum=max_temp, 
                                            value=self.assistant_bot.temp, step=.01)
    
                with gr.Row():
                    self.tk_slider = gr.Slider(label="Top # of MLE" , 
                                            minimum=10, maximum=60, 
                                            value=self.assistant_bot.top_k if self.assistant_bot.top_k else 10, 
                                            step=5)
                    self.tp_slider = gr.Slider(label="Top % of MLE" , 
                                            minimum=0.10, maximum=.70, 
                                            value=self.assistant_bot.top_p if self.assistant_bot.top_p else 0.1,
                                            interactive=True,
                                            step=.05)
                    self.max_new_token_slider = gr.Slider(label="Max New Tokens", 
                                            minimum=400, maximum=8000, 
                                            value=self.assistant_bot.max_new_tokens if self.assistant_bot.max_new_tokens else 400,
                                            interactive=True,
                                            step=100)
                # with generation_settings_row2:
                with gr.Row():
                    self.system_dir_text = gr.Textbox(label="System Directives:",
                                         value="", )
                # with generation_settings_row2:
                with gr.Row():
                    self.system_dir_button = gr.Button("System", scale=1)
            ######################################################################
            #                                       set up event handlers
            ##### User input handling
            self.input_text.submit(self.generate_response, 
                                   inputs=[self.input_text,
                                           self.assistant_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot])
            self.submit_button.click(self.generate_response, 
                                    inputs=[self.input_text,
                                           self.assistant_chatbot], 
                                   outputs=[self.input_text,
                                           self.assistant_chatbot])


            #### clear/reset button logic
            self.clear_button.click(self.clear_chat, outputs=[self.input_text])
            self.reset_button.click(self.reset_conversation)

            # drop down logic on tab 1
            self.dropdown_button1.click(self.copy_text, inputs=self.dropdown_1, outputs=self.input_text)
            

            #### generation settings logic
            self.temp_slider.change(self.assistant_bot.update_temperature, inputs=self.temp_slider)
            self.tp_slider.change(self.assistant_bot.update_top_p, inputs=self.tp_slider)
            self.tk_slider.change(self.assistant_bot.update_top_k, inputs=self.tk_slider)
            self.max_new_token_slider.change(self.assistant_bot.update_max_new_tokens, inputs=self.max_new_token_slider)
            
            self.system_dir_text.submit(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])
            self.system_dir_button.click(self.give_system_directive, inputs=[self.system_dir_text], outputs=[self.system_dir_text])

class GradioChatAssistantDualScreenTab(GradioChatAssistant):
    def __init__(self, assistant_bot, **kwargs):
        super().__init__(assistant_bot=assistant_bot, **kwargs)

    def load_assistant_bot(self, assistant_type, **kwargs):
        self.assistant_bot = assistant_type(**kwargs)
