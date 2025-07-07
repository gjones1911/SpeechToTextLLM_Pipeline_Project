"""
    Purpose: This file contains variables and methods to support demo-scripts.
"""

import time
import joblib
import os
import sys
import argparse

help_msg_base = """
    Usage: python {script_name} [options]

    Optional Arguments:
    --mode ->       Specify the initial mode of the LLM assistant. Options are "BASE", "RAG", "GRAG". Default is "BASE" which is just the model with no RAG/GRAG support.
    
    --knexus_path -> PATH, Specify the path to the knowledge state directory with the name of the base knowledge model stored there. For instance, if your base knowledge
                    model was saved with JHA_DOCS, each part of the knowledge store will have this as the base name with other text appended to the name to indicate 
                    what is stored there. 
    
    --help: -->     Shows a listing of the available document stores. You can copy one of these and use it with the --state_path option to load the knowledge
    --model: -->    path to or huggingface name of a conversational model 
    """
knexus_options_base = [
        "../DomainNexus/HazardAnalysis_Knowledge/JHA_PLN_LOTO", 
        "../DomainNexus/Regulations/Classified_Information", 
        "../DomainNexus/Maintenance_Knowledge/PLN_PRSNL_LOTO", 
        "../DomainNexus/Maintenance_Knowledge/HV_MAINTENANCE", 
    ]



move_trivia_directive = (
    "You are a helpful movie cinema assistant. "
    "To the best of your ability answer any user posed questions to the best of your ability. "
    "Provide sources for any information provided. "
)

class AMAS_DEMO_MANAGER:
    def __init__(self):
        pass
    
    # can be used to print all of the contents of the locla KnowledgeStore documents (pdfs)
    @staticmethod
    def print_options(path=r"../data/Documents/KnowledgeStores"):
        try:
            # go through all contents of dir inpath and if it is a file add it to the list
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_dir() and not entry.name.startswith("."):
                        print("* ", entry.name)
        except FileNotFoundError:
            print(f"Error: The directory {path} does not exist")
        except PermissionError:
            print(f"Error: Permission denied to access '{path}'.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}") 
        sys.exit()
    
    @staticmethod    
    def print_help_msg(help_msg):
        print(help_msg)
        sys.exit()
    
    @staticmethod    
    def print_usage_and_exit(ers, help_msg):
        print(ers)
        print_help_msg(help_msg)
    
    # used to show the user the potential options to use on the command line
    @staticmethod
    def print_knowledge_stores(knexus_options_base=knexus_options_base):
        
        print("\n\n\t\t\t ***** The current available options are: ***** ")
        for k in knexus_options_base:
            print(k)
            print("----------------")
        sys.exit()
    
    @staticmethod    
    def generate_arg_parser(desc, mode, knexus_path, model, **kwargs):
        parser = argparse.ArgumentParser(description=desc)
        
        # Add optional arguments
        parser.add_argument("--mode", type=str, default=mode, 
                            help="Specify the mode to run the application in (default: 'BASE').")
        parser.add_argument("--knexus_path", type=str, default=knexus_path, 
                            help="Specify the path to the state file (default: 'default_state_path').")
        parser.add_argument("--options", action="store_true", help="Show available Knowledge document paths")
        parser.add_argument("--model", type=str, default=model, 
                            help="Path to or name of a hugging face or comparable Huggingface model")
        return parser
    
    @staticmethod
    def parse_demo_args(parser, **kwargs):
        
        try: 
            args = parser.parse_args()
            show_options = args.options     # boolean for if user just wants to see the options string
            mode = args.mode                # RAG/GRAG/BASE  
            knexus_path = args.knexus_path  # local path to stored knexus
            model_path = args.model         # path to local or HF model
    
            if show_options:
                print_knowledge_stores()
                sys.exit()
            return mode, knexus_path, model_path
        except Exception as ex:
            print(ex)
            sys.exit()
    
    @staticmethod    
    def start_app_ui(gradio_ui, share=True, start_port=7850, attempt_limit=10, 
                     save_context=False, system_directive=move_trivia_directive,
                      **kwargs,
                    ):
        attempts = 0
        server = None
        print(f"----> save_content: {save_context}")
        while True:
            try:
                server = gradio_ui.start_app(share=share, debug=False, server_port=start_port, server_name=None, system_directive=system_directive, 
                           save_context=save_context)
                print(f"Gradio app started successfully on port {start_port}.")
                # break
            except Exception as ex:
                print(f"Exception while starting server on port {start_port}:\n{ex}")
                start_port += 1
                attempts += 1
                time.sleep(2)
                # gradio_ui.kill_app()
                if server:
                    gradio_ui.kill_app()
                if attempts > attempt_limit:
                    print(f"Failed to start server after {attempts} attempts.")
                    break
            finally:
                if server:
                    try:
                        gradio_ui.kill_app()
                        print(f"Server on port {start_port} has been closed.")
                    except Exception as cleanup_ex:
                        print(f"Error during server cleanup: {cleanup_ex}")
                    server = None   

    @staticmethod    
    def launch_app_ui_mp_amas(mp_gradio_ui, 
                    share=True, start_port=7850, debug=False,
                    server_name: str=None, attempt_limit=10, 
                    save_context=True, verbose=True, start_directives: dict=None,
                    **kwargs,
                    ):
        attempts = 0
        server = None
        if verbose:
            print(f"----> save_content: {save_context}")
        while True:
            try:
                server = mp_gradio_ui.launch_app(
                            share=share, debug=debug, 
                            server_port=start_port, 
                            server_name=server_name, 
                            system_directives=start_directives, 
                            save_context=save_context, **kwargs,
                  )
                if verbose:
                    print(f"Gradio app started successfully on port {start_port}.")
                # break
            except Exception as ex:
                print(f"Exception while starting server on port {start_port}:\n{ex}")
                start_port += 1
                attempts += 1
                time.sleep(2)
                # gradio_ui.kill_app()
                if server:
                    gradio_ui.kill_app()
                if attempts > attempt_limit:
                    print(f"Failed to start server after {attempts} attempts.")
                    break
            finally:
                if server:
                    try:
                        gradio_ui.kill_app()
                        print(f"Server on port {start_port} has been closed.")
                    except Exception as cleanup_ex:
                        print(f"Error during server cleanup: {cleanup_ex}")
                    server = None        
            
class ObjectFormatJsonFileGenerator:
    def __init__(self, **kwargs):
        """
        Initialize the custom class with the provided variables as keyword arguments.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save_to_json(self, file_name):
        """
        Save the current variables of the instance to a JSON file.
        """
        data = {key: value for key, value in self.__dict__.items()}
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    @classmethod
    def load_from_json(cls, file_name):
        """
        Load variables from a JSON file and create an instance of the class.
        """
        with open(file_name, 'r') as json_file:
            data = json.load(json_file)
        return cls(**data)
        
        
    