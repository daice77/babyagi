from utils import openai_call, remaining_tokens
import os
from dotenv import load_dotenv
import platform
import openai
import json
from utils import MAX_TOKENS_SIMPLE, OPENAI_API_MODEL_SIMPLE, OPENAI_API_MODEL


# Set Variables
load_dotenv()
current_directory = os.getcwd()
os_version = platform.release()


def code_tasks_initializer_agent(objective: str) -> str:
    prompt = f"""You are an AGI agent responsible for creating a detailed JSON checklist of tasks that will guide other AGI agents to complete a given programming objective. Your task is to analyze the provided objective and generate a well-structured checklist with a clear starting point and end point, as well as tasks broken down to be very specific, clear, and executable by other agents without the context of other tasks.

    The current agents work as follows:
    - code_writer_agent: Writes code snippets or functions and saves them to the appropriate files. This agent can also append code to existing files if required.
    - code_refactor_agent: Responsible for modifying and refactoring existing code to meet the requirements of the task.
    - command_executor_agent: Executes terminal commands for tasks such as creating directories, installing dependencies, etc.

    Keep in mind that the agents cannot open files in text editors, and tasks should be designed to work within these agent capabilities.

    Here is the programming objective you need to create a checklist for: \n```objective\n{objective}\n```

    To generate the checklist, follow these steps:

    1. Analyze the objective to identify the high-level requirements and goals of the project. This will help you understand the scope and create a comprehensive checklist.

    2. Break down the objective into smaller, highly specific tasks that can be worked on independently by other agents. Ensure that the tasks are designed to be executed by the available agents (code_writer_agent, code_refactor and command_executor_agent) without requiring opening files in text editors.

    3. Assign a unique ID to each task for easy tracking and organization. This will help the agents to identify and refer to specific tasks in the checklist.

    4. Organize the tasks in a logical order, with a clear starting point and end point. The starting point should represent the initial setup or groundwork necessary for the project, while the end point should signify the completion of the objective and any finalization steps.

    5. Provide the current context for each task, which should be sufficient for the agents to understand and execute the task without referring to other tasks in the checklist. This will help agents avoid task duplication.

    6. Pay close attention to the objective and make sure the tasks implement all necessary pieces needed to make the program work.
    
    7. Compile the tasks into a well-structured JSON format, ensuring that it is easy to read and parse by other AGI agents. The JSON should include fields such as task ID, description and file_path.

    IMPORTANT: BE VERY CAREFUL WITH IMPORTS AND MANAGING MULTIPLE FILES. REMEMBER EACH AGENT WILL ONLY SEE A SINGLE TASK. ASK YOURSELF WHAT INFORMATION YOU NEED TO INCLUDE IN THE CONTEXT OF EACH TASK TO MAKE SURE THE AGENT CAN EXECUTE THE TASK WITHOUT SEEING THE OTHER TASKS OR WHAT WAS ACCOMPLISHED IN OTHER TASKS.

    Pay attention to the way files are passed in the tasks, always use full paths. For example 'project/main.py'.

    Make sure tasks are not duplicated.

    Do not take long and complex routes, minimize tasks and steps as much as possible.

    Here is a sample JSON output for a checklist:

            {{
                "tasks": [
                    {{
                    "id": 1,
                    "description": "Run a command to create the project directory named 'project'",
                    "file_path": "./project",
                    }},
                    {{
                    "id": 2,
                    "description": "Run a command to Install the following dependencies: 'numpy', 'pandas', 'scikit-learn', 'matplotlib'",
                    "file_path": "null",
                    }},
                    {{
                    "id": 3,
                    "description": "Write code to create a function named 'parser' that takes an input named 'input' of type str, [perform a specific task on it], and returns a specific output",
                    "file_path": "./project/main.py",
                    }},
                    ...
                    {{
                    "id": N,
                    "description": "...",
                    }}
                ],
            }}

    The tasks will be executed by either of the three agents: command_executor, code_writer or code_refactor. They can't interact with programs. They can either run terminal commands or write code snippets. Their output is controlled by other functions to run the commands or save their output to code files. Make sure the tasks are compatible with the current agents. ALL tasks MUST start either with the following phrases: 'Run a command to...', 'Write code to...', 'Edit existing code to...' depending on the agent that will execute the task. Let's think step by step. RETURN JSON ONLY:"""

    return openai_call(prompt, temperature=0.8, max_tokens=remaining_tokens(prompt))


def code_tasks_refactor_agent(objective: str, task_list_json) -> str:
    prompt = f"""You are an AGI tasks_refactor_agent responsible for adapting a task list generated by another agent to ensure the tasks are compatible with the current AGI agents. Your goal is to analyze the task list and make necessary modifications so that the tasks can be executed by the agents listed below

    YOU SHOULD OUTPUT THE MODIFIED TASK LIST IN THE SAME JSON FORMAT AS THE INITIAL TASK LIST. DO NOT CHANGE THE FORMAT OF THE JSON OUTPUT. DO NOT WRITE ANYTHING OTHER THAN THE MODIFIED TASK LIST IN THE JSON FORMAT.
    
    The current agents work as follows:
    - code_writer_agent: Writes code snippets or functions and saves them to the appropriate files. This agent can also append code to existing files if required.
    - code_refactor_agent: Responsible for editing current existing code/files.
    - command_executor_agent: Executes terminal commands for tasks such as creating directories, installing dependencies, etc.

    Here is the overall objective you need to refactor the tasks for: \n```objective\n{objective}\n```
    Here is the JSON task list you need to refactor for compatibility with the current agents: {task_list_json}.

    To refactor the task list, follow these steps:
    1. Modify the task descriptions to make them compatible with the current agents, ensuring that the tasks are self-contained, clear, and executable by the agents without additional context. You don't need to mention the agents in the task descriptions, but the tasks should be compatible with the current agents.
    2. If necessary, add new tasks or remove irrelevant tasks to make the task list more suitable for the current agents.
    3. Keep the JSON structure of the task list intact, maintaining the "id", "description" and "file_path" fields for each task.
    4. Pay close attention to the objective and make sure the tasks implement all necessary pieces needed to make the program work.

    Always specify file paths to files. Make sure tasks are not duplicated. Never write code to create files. If needed, use commands to create files and folders.
    Return the updated JSON task list with the following format:

            {{
                "tasks": [
                    {{
                    "id": 1,
                    "description": "Run a commmand to create a folder named 'project' in the current directory",
                    "file_path": "./project",
                    }},
                    {{
                    "id": 2,
                    "description": "Write code to print 'Hello World!' with Python",
                    "file_path": "./project/main.py",
                    }},
                    {{
                    "id": 3,
                    "description": "Write code to create a function named 'parser' that takes an input named 'input' of type str, [perform a specific task on it], and returns a specific output",
                    "file_path": "./project/main.py",
                    }}
                    {{
                    "id": 3,
                    "description": "Run a command calling the script in ./project/main.py",
                    "file_path": "./project/main.py",
                    }}
                    ...
                ],
            }}

    IMPORTANT: All tasks should start either with the following phrases: 'Run a command to...', 'Write a code to...', 'Edit the code to...' depending on the agent that will execute the task:

    Let's think step by step. ALWAYS ENSURE ALL TASKS HAVE RELEVANT CONTEXT ABOUT THE CODE TO BE WRITTEN, INCLUDE DETAILS ON HOW TO CALL FUNCTIONS, CLASSES, IMPORTS, ETC. AGENTS HAVE NO VIEW OF OTHER TASKS, SO THEY NEED TO BE SELF-CONTAINED. RETURN THE JSON:"""

    return openai_call(prompt, model=OPENAI_API_MODEL, temperature=0, max_tokens=remaining_tokens(prompt))


def code_tasks_details_agent(objective: str, task_list_json) -> str:
    prompt = f"""You are an AGI agent responsible for improving a list of tasks in JSON format and adding ALL the necessary details to each task. These tasks will be executed individually by agents that have no idea about other tasks or what code exists in the codebase. It is FUNDAMENTAL that each task has enough details so that an individual isolated agent can execute. The metadata of the task is the only information the agents will have.

    Each task should contain the details necessary to execute it. For example, if it creates a function, it needs to contain the details about the arguments to be used in that function and this needs to be consistent across all tasks.

    Look at all tasks at once, and update the task description adding details to it for each task so that it can be executed by an agent without seeing the other tasks and to ensure consistency across all tasks. DETAILS ARE CRUCIAL. For example, if one task creates a class, it should have all the details about the class, including the arguments to be used in the constructor. If another task creates a function that uses the class, it should have the details about the class and the arguments to be used in the constructor.

    RETURN JSON OUTPUTS ONLY.
    
    Here is the overall objective you need to refactor the tasks for: \n```objective\n{objective}\n```.
    Here is the task list you need to improve: {task_list_json}
    
    RETURN THE SAME TASK LIST but with the description improved to contain the details you is adding for each task in the list. DO NOT MAKE OTHER MODIFICATIONS TO THE LIST. Your input should go in the 'description' field of each task.
    
    Let's think step by step. RETURN JSON ONLY:"""
    return openai_call(prompt, model=OPENAI_API_MODEL, temperature=0.2, max_tokens=remaining_tokens(prompt))


def code_tasks_context_agent(objective: str, task_list_json):
    prompt = f"""You are an AGI agent responsible for improving a list of tasks in JSON format and adding ALL the necessary context to it. These tasks will be executed individually by agents that have no idea about other tasks or what code exists in the codebase. It is FUNDAMENTAL that each task has enough context so that an individual isolated agent can execute. The metadata of the task is the only information the agents will have.

    Look at all tasks at once, and add the necessary context to each task so that it can be executed by an agent without seeing the other tasks. Remember, one agent can only see one task and has no idea about what happened in other tasks. CONTEXT IS CRUCIAL. For example, if one task creates one folder and the other tasks creates a file in that folder. The second tasks should contain the name of the folder that already exists and the information that it already exists.

    This is even more important for tasks that require importing functions, classes, etc. If a task needs to call a function or initialize a Class, it needs to have the detailed arguments, etc.

    Note that you should identify when imports need to happen and specify this in the context. Also, you should identify when functions/classes/etc already exist and specify this very clearly because the agents sometimes duplicate things not knowing.

    Always use imports with the file name. For example, 'from my_script import MyScript'. 
    
    RETURN JSON OUTPUTS ONLY.
    
    Here is the overall objective you need to refactor the tasks for: \n```objective\n{objective}\n```.
    Here is the task list you need to improve: {task_list_json}
    
    RETURN THE SAME TASK LIST but with a new field called 'isolated_context' for each task in the list. This field should be a string with the context you are adding. DO NOT MAKE OTHER MODIFICATIONS TO THE LIST.
    
    Let's think step by step. RETURN JSON ONLY:"""
    return openai_call(prompt, model=OPENAI_API_MODEL, temperature=0.2, max_tokens=remaining_tokens(prompt))


def task_assigner_recommendation_agent(objective: str, task: str) -> str:
    prompt = f"""You are an AGI agent responsible for providing recommendations on which agent should be used to handle a specific task. Analyze the provided major objective of the project and a single task from the JSON checklist generated by the previous agent, and suggest the most appropriate agent to work on the task.

    The overall objective is: \n```objective\n{objective}\n```
    The current task is: \n```task\n{task}\n```
    
    The available agents are:
    1. code_writer_agent: Responsible for writing code based on the task description.
    2. code_refactor_agent: Responsible for editing existing code.
    3. command_executor_agent: Responsible for executing commands and handling file operations, such as creating, moving, or deleting files.

    When analyzing the task, consider the following tips:
    - Pay attention to keywords in the task description that indicate the type of action required, such as "write", "edit", "run", "create", "move", or "delete".
    - Keep the overall objective in mind, as it can help you understand the context of the task and guide your choice of agent.
    - If the task involves writing new code or adding new functionality, consider using the code_writer_agent.
    - If the task involves modifying or optimizing existing code, consider using the code_refactor_agent.
    - If the task involves file operations, command execution, or running a script, consider using the command_executor_agent.

    Based on the task and overall objective, suggest the most appropriate agent to work on the task. """
    return openai_call(prompt, model=OPENAI_API_MODEL_SIMPLE, temperature=0.2, max_tokens=remaining_tokens(prompt, max_tokens=MAX_TOKENS_SIMPLE))


def task_assigner_agent(objective: str, task: str, recommendation: str) -> str:
    prompt = f"""You are an AGI agent responsible for choosing the best agent to work on a given task. Your goal is to analyze the provided major objective of the project and a single task from the JSON checklist generated by the previous agent, and choose the best agent to work on the task.

    The overall objective is: \n```objective\n{objective}\n```
    The current task is: \n```task\n{task}\n```

    Use this recommendation to guide you: {recommendation}
        
    The available agents are:
    1. code_writer_agent: Responsible for writing code based on the task description.
    2. code_refactor_agent: Responsible for editing existing code.
    2. command_executor_agent: Responsible for executing commands and handling file operations, such as creating, moving, or deleting files.

    Please consider the task description and the overall objective when choosing the most appropriate agent. Keep in mind that creating a file and writing code are different tasks. If the task involves creating a file, like "calculator.py" but does not mention writing any code inside it, the command_executor_agent should be used for this purpose. The code_writer_agent should only be used when the task requires writing or adding code to a file. The code_refactor_agent should only be used when the task requires modifying existing code.
    
    TLDR: To create files, use command_executor_agent; to write text/code to files, use code_writer_agent; to modify existing code, use code_refactor_agent.

    Choose the most appropriate agent to work on the task and return a JSON output with the following format: {{"agent": "agent_name"}}. ONLY return JSON output:"""
    return openai_call(prompt, model=OPENAI_API_MODEL_SIMPLE, temperature=0, max_tokens=remaining_tokens(prompt, max_tokens=MAX_TOKENS_SIMPLE))


def command_executor_agent(task: str, file_path: str) -> str:
    prompt = f"""You are an AGI agent responsible for executing a given command on the {os_version} OS. Your goal is to analyze the provided major objective of the project and a single task from the JSON checklist generated by the previous agent, and execute the command on the {os_version} OS. 

    The current task is: \n```task\n{task}\n```
    File or folder name referenced in the task (relative file path): {file_path} 
    
    Based on the task, write the appropriate command to execute on the {os_version} OS. Make sure the command is relevant to the task and objective. For example, if the task is to create a new folder, the command should be 'mkdir new_folder_name'. Return the command as a JSON output with the following format: {{"command": "command_to_execute"}}. ONLY return JSON output:"""
    return openai_call(prompt, model=OPENAI_API_MODEL_SIMPLE, temperature=0, max_tokens=remaining_tokens(prompt, max_tokens=MAX_TOKENS_SIMPLE))


def code_writer_agent(task: str, isolated_context: str, context_code_chunks: list | str) -> str:
    prompt = f"""You are an AGI agent responsible for writing code to accomplish a given task. Your goal is to analyze the provided major objective of the project and a single task from the JSON checklist generated by the previous agent, and write the necessary code to complete the task.

    The current task is: \n```task\n{task}\n```

    To help you make the code useful in this codebase, use this context as reference of the other pieces of the codebase that are relevant to your task. PAY ATTENTION TO THIS: {isolated_context}
    
    The following code chunks were found to be relevant to the task (JSON with fileName, from/to line, code snippet). You can use them as reference to write the code if they are useful. PAY CLOSE ATTENTION TO THIS: 
    {context_code_chunks}

    Note: Always use 'encoding='utf-8'' when opening files with open().
    
    Based on the task and objective, write the appropriate code to achieve the task. Make sure the code is relevant to the task and objective, and follows best practices. Return the code as a plain text output and NOTHING ELSE. Use identation and line breaks in the in the code. Make sure to only write the code and nothing else as your output will be saved directly to the file by other agent. 
    IMPORTANT: If the task is asking you to write code to write files, this is a mistake! Interpret it and either do nothing or return  the plain code, not a code to write file, not a code to write code, etc.
    Let's think step by step."""
    return openai_call(prompt, model=OPENAI_API_MODEL, temperature=0.2, max_tokens=remaining_tokens(prompt))


def code_refactor_agent(task_description: str, existing_code_snippet: dict, context_chunks, isolated_context: str) -> str:

    prompt = f"""You are an AGI agent responsible for refactoring code to accomplish a given task. Your goal is to analyze the provided major objective of the project, the task descriptionm and refactor the code accordingly.

    The current task description is: \n```task description\n{task_description}\n```
    To help you make the code useful in this codebase, use this context as reference of the other pieces of the codebase that are relevant to your task: {isolated_context}

    Here are some code context chunks that might also be relevant to the task:
    {context_chunks}
    
    This is the existing code you should refactor (this is a snippet from the file {existing_code_snippet["file_path"]}, line {existing_code_snippet["from_line"]} to line {existing_code_snippet["to_line"]}):
    \n```code\n{existing_code_snippet["code"]}\n```
    
    Now refactor above code to achieve the task (based on the task description, objective). Make sure the refactored code is relevant to the task and objective, follows best practices, verified. Reflect on your response before replying.

    Next prepare the modifications to the original code. Use the output format from "diff -u", which next agent can then apply as a patch to the original source code file.
    
    The source code is provided as an snippet from original source code file, including line numbers in original source code file. Pretend you know the whole file and based on that, generate the patch (output of diff, sent back as JSON).

    Respond only with JSON! Use following format: 
    {{"file_path": "{{file_to_be_modified}}", "patch":{{"output_from_diff"}}}}. 
    
    Let's think step by step.

    IMPORTANT: RESPOND WITH JSON ONLY, NOTHING ELSE! YOUR OUTPUT WILL BE ADDED DIRECTLY TO THE FILE BY OTHER AGENT. BE MINDFUL OF THIS:"""

    return openai_call(prompt, model=OPENAI_API_MODEL, temperature=0.2, max_tokens=remaining_tokens(prompt))


def file_management_agent(objective: str, task: str, current_directory_files: str, file_path: str) -> str:
    prompt = f"""You are an AGI agent responsible for managing files in a software project. Your goal is to analyze the provided major objective of the project and a single task from the JSON checklist generated by the previous agent, and determine the appropriate file path and name for the generated code.

    The overall objective is: \n```objective\n{objective}\n```
    The current task is: \n```task\n{task}\n```
    Specified file path (relative path from the current dir): {file_path}

    Make the file path adapted for the current directory files. The current directory files are: {current_directory_files}. Assume this file_path will be interpreted from the root path of the directory.

    Do not use '.' or './' in the file path.

    BE VERY SPECIFIC WITH THE FILES, AVOID FILE DUPLICATION, AVOID SPECIFYING THE SAME FILE NAME UNDER DIFFERENT FOLDERS, ETC.

    Based on the task, determine the file path and name for the generated code. Return the file path and name as a JSON output with the following format: {{"file_path": "file_path_and_name"}}. ONLY return JSON output:"""
    return openai_call(prompt, model=OPENAI_API_MODEL_SIMPLE, temperature=0, max_tokens=remaining_tokens(prompt, max_tokens=MAX_TOKENS_SIMPLE))


def code_relevance_agent(objective: str, task_description: str, code_chunk: dict) -> str:
    prompt = f"""You are an AGI agent responsible for evaluating the relevance of a code chunk in relation to a given task. Your goal is to analyze the provided major objective of the project, the task description, and the code chunk, and assign a relevance score from 0 to 1000, where 0 is completely irrelevant and 1000 is highly relevant.

    The overall objective is: \n```objective\n{objective}\n```
    The current task description is: \n```task description\n{task_description}\n```
    The code chunk is as follows (line numbers included):
    {str(code_chunk)}

    Based on the task description, objective, and code chunk, assign a relevance score between 0 and 1000 (inclusive) for the code chunk. DO NOT OUTPUT ANYTHING OTHER THAN THE RELEVANCE SCORE AS A NUMBER."""

    relevance_score = openai_call(
        prompt, model=OPENAI_API_MODEL, temperature=0.1, max_tokens=50)

    return json.dumps({"relevance_score": relevance_score.strip()})


def task_human_input_agent(task: str, human_feedback: str) -> str:
    prompt = f"""You are an AGI agent responsible for getting human input to improve the quality of tasks in a software project. Your goal is to analyze the provided task and adapt it based on the human's suggestions. The tasks should  start with either 'Run a command to...', 'Write code to...', or 'Edit existing code to...' depending on the agent that will execute the task.

    For context, this task will be executed by other AGI agents with the following characteristics:
    - code_writer_agent: Writes code snippets or functions and saves them to the appropriate files. This agent can also append code to existing files if required.
    - code_refactor_agent: Responsible for modifying and refactoring existing code to meet the requirements of the task.
    - command_executor_agent: Executes terminal commands for tasks such as creating directories, installing dependencies, etc.

    The current task is:\n```task\n{task}\n```

    The human feedback is:\n```feedback\n{human_feedback}\n```

    If the human feedback is empty, return the task as is. If the human feedback is saying to ignore the task, return the following string: <IGNORE_TASK>

    Note that your output will replace the existing task, so make sure that your output is a valid task that starts with one of the required phrases ('Run a command to...', 'Write code to...', 'Edit existing code to...').
    
    Please adjust the task based on the human feedback while ensuring it starts with one of the required phrases ('Run a command to...', 'Write code to...', 'Edit existing code to...'). Return the improved task as a plain text output and nothing else. Write only the new task. Let's think step by step."""

    return openai_call(prompt, model=OPENAI_API_MODEL, temperature=0.3, max_tokens=remaining_tokens(prompt))


if __name__ == "__main__":
    pass
