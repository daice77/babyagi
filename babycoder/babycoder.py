"""
# AI-based Programming Assistant

This script is an AI-based programming assistant that automates coding tasks. It imports necessary libraries, sets environment variables and API keys, and reads the objective from a file or command-line argument. It uses embeddings, file management, and several agents to complete tasks based on the given objective. At the end, it saves and applies the generated code and displays the progress throughout the process.

## Dependencies:

- openai library
- embeddings.py module
- agents.py module
- utils.py module

## Environment Variables (to be defined in .env file):

- OPENAI_API_KEY
- OPENAI_API_MODEL
- OPENAI_API_MODEL_SIMPLE
- RESET_EMBEDDINGS_INDEX (optional)

## Main tasks performed by the script:

1. Load objective (from file or command line argument)
2. Generate initial tasks, refactor the tasks, add relevant details, and necessary context
3. Loop through the generated tasks
4. Print task details
5. Determine which agent to delegate the task to (command executor or code writer/refactor agent)
6. Execute the task using the assigned agent
7. Update the file with generated code or apply refactoring

"""

import os
import sys
from dotenv import load_dotenv
import json
import argparse

import shutil
from embeddings import Embeddings

from utils import (
    execute_command_json,
    execute_command_string,
    save_code_to_file,
    split_code_into_chunks,
)
from utils import print_colored_text, print_char_by_char
from utils import OPENAI_API_MODEL

from agents import (
    code_writer_agent,
    code_refactor_agent,
    code_tasks_context_agent,
    code_relevance_agent,
)
from agents import task_assigner_agent, task_assigner_recommendation_agent
from agents import command_executor_agent, file_management_agent
from agents import (
    code_tasks_refactor_agent,
    code_tasks_context_agent,
    code_tasks_initializer_agent,
    code_tasks_details_agent,
)

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[{asctime}] [{levelname}] [{filename}:{lineno}] {funcName}: {message}",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="debug.openai.log",
    filemode="a",
    style="{",
)

logger = logging.getLogger(__name__)

# Set Variables
load_dotenv()
current_directory = os.getcwd()


# clear embeddings index and re-index all playground files during startup
RESET_EMBEDDINGS_INDEX = os.getenv("RESET_EMBEDDINGS_INDEX", False)
if RESET_EMBEDDINGS_INDEX:
    print(
        f"\033[91m\033[1m"
        + "\n*****RESET AND RECREATE EMBEDDINGS INDEX*****"
        + "\033[0m\033[0m"
    )
    try:
        playground_data_path = os.path.join(current_directory, "playground_data")

        # Delete the contents of the playground_data directory but not the directory itself
        # This is to ensure that we don't have any old data lying around
        for filename in os.listdir(playground_data_path):
            file_path = os.path.join(playground_data_path, filename)

            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

    embeddings = Embeddings(current_directory)
    embeddings.compute_repository_embeddings()

# gpt-4 cost warning
if "gpt-4" in OPENAI_API_MODEL.lower():
    print(
        f"\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

# get OBJECTIVE
if len(sys.argv) > 1:
    OBJECTIVE = sys.argv[1]
elif os.path.exists(os.path.join(current_directory, "objective.txt")):
    with open(os.path.join(current_directory, "objective.txt")) as f:
        OBJECTIVE = f.read()
assert OBJECTIVE, "OBJECTIVE missing"


def main_function(_args):
    """
    The main function executes the AI-based Programming Assistant script that simplifies the coding process.
    It automates imports, sets environment variables, and reads the objective from a file or command-line argument.
    The AI-assistant then proceeds to complete tasks based on the targeted result. It saves the outcome as a generated code
    and displays progress throughout the process.

    :return: None
    """
    print_colored_text(f"****Objective****", color="green")
    print_char_by_char(OBJECTIVE, 0.00001, 10)

    # Create the tasks
    print_colored_text("*****Working on tasks*****", "red")
    print_colored_text(" - Creating initial tasks", "yellow")
    task_agent_output = code_tasks_initializer_agent(OBJECTIVE)
    print_colored_text(" - Reviewing and refactoring tasks to fit agents", "yellow")
    task_agent_output = code_tasks_refactor_agent(OBJECTIVE, task_agent_output)
    print_colored_text(" - Adding relevant technical details to the tasks", "yellow")
    task_agent_output = code_tasks_details_agent(OBJECTIVE, task_agent_output)
    print_colored_text(" - Adding necessary context to the tasks", "yellow")
    task_agent_output = code_tasks_context_agent(OBJECTIVE, task_agent_output)
    print()

    print_colored_text("*****TASKS*****", "green")
    print_char_by_char(task_agent_output, 0.00000001, 10)

    # Task list
    task_json = json.loads(task_agent_output)

    embeddings = Embeddings(current_directory)

    for task in task_json["tasks"]:
        task_description = task["description"]
        task_isolated_context = task["isolated_context"]

        print_colored_text(
            f'*****TASK {task["id"]}/{len(task_json["tasks"])}*****', "yellow"
        )
        print_char_by_char(task_description)
        print_colored_text("*****TASK CONTEXT*****", "yellow")
        print_char_by_char(task_isolated_context)

        # HUMAN FEEDBACK
        # Uncomment below to enable human feedback before each task. This can be used to improve the quality of the tasks,
        # skip tasks, etc. I believe it may be very relevant in future versions that may have more complex tasks and could
        # allow a ton of automation when working on large projects.
        #
        # Get user input as a feedback to the task_description
        # print_colored_text("*****TASK FEEDBACK*****", "yellow")
        # user_input = input("\n>:")
        # task_description = task_human_input_agent(task_description, user_input)
        # if task_description == "<IGNORE_TASK>":
        #     continue
        # print_colored_text("*****IMPROVED TASK*****", "green")
        # print_char_by_char(task_description)

        # Assign the task to an agent
        task_assigner_recommendation = task_assigner_recommendation_agent(
            OBJECTIVE, task_description
        )
        task_agent_output = task_assigner_agent(
            OBJECTIVE, task_description, task_assigner_recommendation
        )

        print_colored_text("*****ASSIGN*****", "yellow")
        print_char_by_char(task_agent_output)

        chosen_agent = json.loads(task_agent_output)["agent"]

        if chosen_agent == "command_executor_agent":
            command_executor_output = command_executor_agent(
                task_description, task["file_path"]
            )
            print_colored_text("*****COMMAND*****", "green")
            print_char_by_char(command_executor_output)

            command_execution_output = execute_command_json(command_executor_output)
        else:
            # CODE AGENTS
            if chosen_agent == "code_writer_agent":
                # Compute embeddings for the codebase
                # This will recompute embeddings for all files in the 'playground' directory
                print_colored_text(
                    "*****RETRIEVING RELEVANT CODE CONTEXT*****", "yellow"
                )
                embeddings.compute_repository_embeddings()
                relevant_chunks = embeddings.get_relevant_code_chunks(
                    task_description, task_isolated_context
                )

                current_directory_files = execute_command_string("find . -type f")
                current_directory_files = current_directory_files.split("\n")
                file_management_output = file_management_agent(
                    OBJECTIVE,
                    task_description,
                    current_directory_files,
                    task["file_path"],
                )
                print_colored_text("*****FILE MANAGEMENT*****", "yellow")
                print_char_by_char(file_management_output)
                file_path = json.loads(file_management_output)["file_path"]

                code_writer_output = code_writer_agent(
                    task_description, task_isolated_context, relevant_chunks
                )

                print_colored_text("*****CODE*****", "green")
                print_char_by_char(code_writer_output)

                # Save the generated code to the file the agent selected
                save_code_to_file(code_writer_output, file_path)

            elif chosen_agent == "code_refactor_agent":
                # The code refactor agent works with multiple agents:
                # For each task, the file_management_agent is used to select the file to edit.Then, the
                # code_relevance_agent is used to select the relevant code chunks from that filewith the
                # goal of finding the code chunk that is most relevant to the task description. This is
                # the code chunk that will be edited. Finally, the code_refactor_agent is used to edit
                # the code chunk.

                current_directory_files = execute_command_string("find . -type f")
                current_directory_files = current_directory_files.split("\n")
                file_management_output = file_management_agent(
                    OBJECTIVE,
                    task_description,
                    current_directory_files,
                    task["file_path"],
                )
                file_path = json.loads(file_management_output)["file_path"]

                print_colored_text("*****FILE MANAGEMENT*****", "yellow")
                print_char_by_char(file_management_output)

                # Split the code into chunks and get the relevance scores for each chunk
                code_chunks = split_code_into_chunks(file_path, 80)
                print_colored_text("*****ANALYZING EXISTING CODE*****", "yellow")
                relevance_scores = []
                for chunk in code_chunks:
                    score = code_relevance_agent(OBJECTIVE, task_description, chunk)
                    relevance_scores.append(score)

                # Select the most relevant chunk and the context chunks (similar high score)
                relevance_scores_dict = [json.loads(r) for r in relevance_scores]
                relevance_scores_int = [
                    {"relevance_score": int(x["relevance_score"])}
                    for x in relevance_scores_dict
                ]
                results = [
                    {**score, **chunk}
                    for score, chunk in zip(relevance_scores_int, code_chunks)
                ]
                ranked_results = sorted(
                    results, key=lambda x: x["relevance_score"], reverse=True
                )
                selected_chunk = ranked_results[0]
                top_score = selected_chunk["relevance_score"]
                relevant_chunks = [
                    r for r in ranked_results if r["relevance_score"] >= 0.7 * top_score
                ]
                code_snipplet = {
                    "code": selected_chunk["code"],
                    "from_line": selected_chunk["start_line"],
                    "to_line": selected_chunk["end_line"],
                    "file_path": file_path,
                }

                # Refactor the code
                modified_code_output = code_refactor_agent(
                    task_description,
                    code_snipplet,
                    context_chunks=relevant_chunks,
                    isolated_context=task_isolated_context,
                )

                try:
                    patch = json.loads(modified_code_output)["patch"]

                except json.JSONDecodeError as e:
                    # try to remove ``` at beginning and end
                    modified_code_output = modified_code_output[
                        modified_code_output.find("{") : modified_code_output.rfind(
                            "```"
                        )
                    ]
                    patch = json.loads(modified_code_output)["patch"]

                print_colored_text("*****PATCHING CODE*****", "green")
                print(patch)
                response = []
                response += execute_command_string(f"echo '{patch}' > .tmp.patch")
                response += execute_command_string(
                    f"sed 's/\\\\n/\\\n/g' .tmp.patch > .tmp.sed"
                )
                response += execute_command_string(f"patch --backup < .tmp.sed")
                response += execute_command_string(f"rm .tmp.*")
                logger.debug(f"Patching command returned:{response}")

        print_colored_text("*****TASK COMPLETED*****", "yellow")

    print_colored_text("*****ALL TASKS COMPLETED*****", "yellow")
    pass


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="AI-based Programming Assistant", add_help=False
    )
    parser.add_argument(
        "--help", action="store_true", help="Display short usage information"
    )
    parser.add_argument(
        "objective",
        nargs="?",
        help="Objective to complete by the assistant",
    )
    args = parser.parse_args(argv)
    return args


def show_usage():
    usage = """ 
    Usage: python babycoder.py [objective]
    objective: Objective to complete by the assistant

    Example: python babycoder.py "Create a function to add two numbers"
    """
    print(usage)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.help:
        show_usage()
    else:
        main_function(args)
