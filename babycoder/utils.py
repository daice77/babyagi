"""
# Code Documentation
This code module is designed for connecting to the OpenAI API, executing user commands in a safe environment, and handling TTS (Text-to-Speech) functionality.

## Imported Libraries & Modules:

- tiktoken: Used for counting tokens
- numpy: Used for numeric operations
- os, platform: For handling file paths and system-related tasks
- dotenv: Used for managing environment variables
- gTTS: Google Text-to-Speech library
- pydub, simpleaudio: Libraries for audio manipulation and playback
- time, tempfile, logging, json, subprocess, typing: Additional utility libraries for various tasks

## Setup & Initialization:

Logger and environment variables are initialized, and various important variables and settings are defined at the top of the module.

## Functions:

- available_tokens(): Calculate the available tokens for an OpenAI API call
- count_tokens(): Count the number of tokens in a string
- print_colored_text(): Print colored text on the terminal and optionally play TTS audio
- print_char_by_char(): Print characters one by one with an optional delay
- text_to_speech(): Convert text to speech and play it
- openai_call(): Call the OpenAI API with the given prompt and parameters, handling retries in case of errors
- execute_command_json(): Execute a command specified in a JSON string and return the result
- execute_command_string(): Execute a command specified in a string and return the result
- save_code_to_file(): Save code to a file, either in "playground" subdirectory
- refactor_code(): Refactor a code file based on a list of modifications
- split_code_into_chunks(): Split a code file into chunks of lines
"""
import tiktoken
import numpy as np
import os
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa
import numpy as np
import time
import tiktoken
import tempfile
import logging
import openai
import json
import subprocess
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

# Set Variables
load_dotenv()
current_directory = os.getcwd()

openai_calls_retried = 0
max_openai_calls_retries = 3

# enable/disable text to speech
TTS = os.getenv("TTS", False)

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", None)
OPENAI_API_MODEL_SIMPLE = os.getenv("OPENAI_API_MODEL_SIMPLE", "gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

OPENAI_API_MODEL_SIMPLE = os.getenv("OPENAI_API_MODEL_SIMPLE", "gpt-3.5-turbo")
assert (
    OPENAI_API_MODEL_SIMPLE
), "OPENAI_API_MODEL_SIMPLE environment variable is missing from .env"

MAX_TOKENS = 8196
MAX_TOKENS_SIMPLE = 4096


def remaining_tokens(
    prompt, max_tokens: int = MAX_TOKENS, model: str = "cl100k_base"
) -> int:
    """
    Calculate the remaining tokens that can be used in an OpenAI API call after considering the input text.

    Parameters
    ----------
    prompt : str
        The input text for which the remaining tokens need to be calculated.
    max_tokens : int, optional
        The maximum tokens allowed by the API. Default is MAX_TOKENS.
    model : str, optional
        The model to be used for token counting. Default is "cl100k_base".

    Returns
    -------
    int
        The number of remaining tokens after considering the input text.
    """
    return int(0.9 * (max_tokens - count_tokens(prompt, model)))


def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    This function returns the number of tokens in a given string based on a specified encoding.

    Parameters
    ----------
    string : str
        The input string to be tokenized.
    encoding_name :str, optional
        The name of the encoding model to be used. Defaults to 'cl100k_base'.

    Returns
    -------
    int
        The number of tokens in the input string based on the specified encoding.
    """
    # no support for gpt-4 yet but number of tokens is the same as this model
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def print_colored_text(text, color) -> None:
    """
    Print the given text in the specified color and style.

    Parameters
    ----------
    text : str
        The text to print.
    color : str, optional
        The color to print the text in. Default is 'red'.
    style : str, optional
        The text style. Default is 'bold'.
    """
    color_mapping = {
        "blue": "\033[34m",
        "red": "\033[31m",
        "yellow": "\033[33m",
        "green": "\033[32m",
    }
    color_code = color_mapping.get(color.lower(), "")
    reset_code = "\033[0m"
    print(color_code + text + reset_code)
    if TTS == True:
        text_to_speech(text)


def print_char_by_char(text: str, delay=0.00001, chars_at_once=3) -> None:
    """
    Print the given text one character at a time with an optional delay and indentation.

    Parameters
    ----------
    text : str
        The text to print.
    delay : float, optional
        The amount of time (in seconds) to wait before printing the next character. Default is 0.02.
    spaces : int, optional
        The number of spaces to indent each line. Default is 0.
    """
    if text is not None:
        for i in range(0, len(text), chars_at_once):
            chunk = text[i : i + chars_at_once]
            print(chunk, end="", flush=True)
            time.sleep(delay)
        print()


def __play_audio(audio) -> None:
    samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
    sa.play_buffer(samples, audio.channels, 2, audio.frame_rate).wait_done()


def text_to_speech(text, lang="en") -> None:
    """
    Convert the given text to speech and play it.

    Parameters
    ----------
    text : str
        The input text that needs to be converted to speech.
    lang : str, optional
        The language to use for TTS. Default is 'en'.
    """
    tts = gTTS(text=text.replace("*", ""), lang=lang)

    with tempfile.NamedTemporaryFile(delete=True) as fp:
        tts.save(fp.name)
        audio = AudioSegment.from_mp3(fp.name)
        __play_audio(audio)


def openai_call(
    prompt: str,
    model: str = OPENAI_API_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 100,
) -> str:
    """
    Call the OpenAI API with the given prompt and parameters, handling retries in case of errors.

    Parameters
    ----------
    prompt : str
        The text prompt to send to the OpenAI API.
    model : str, optional
        The model to use for the API call. Default is set by the OPENAI_API_MODEL environment variable.
    temperature : float, optional
        Controls the randomness of the generated text. Default is 0.5.
    max_tokens : int, optional
        The maximum number of tokens allowed in the generated text. Default is 100.

    Returns
    -------
    str
        The text generated by the OpenAI API.
    """
    global openai_calls_retried
    if not model.startswith("gpt-"):
        # Use completion API
        logger.info(f"Request(completion): {prompt}")

        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        logger.info(f"Response: {response.choices[0].text.strip()}")
        return response.choices[0].text.strip()
    else:
        # Use chat completion API
        messages = [{"role": "user", "content": prompt}]
        try:
            logger.info(f"Request(chat): {prompt}")
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
            )
            openai_calls_retried = 0
            logger.info(f"Response: {response.choices[0].message.content.strip()}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("Error calling OpenAI:")
            if openai_calls_retried < max_openai_calls_retries:
                # try again
                openai_calls_retried += 1
                print(
                    f"Error calling OpenAI. Retrying {openai_calls_retried} of {max_openai_calls_retries}..."
                )
                return openai_call(prompt, model, temperature, max_tokens)
            else:
                # re-raise the exception
                raise e


def execute_command_json(json_string: str) -> str:
    """
    Execute a command and return stdout output (or stderr if there was an error)

    Parameters
    ----------
    command : str (JSON)
        The shell command to execute, eg '{"command": "cp . backup"}'

    Returns
    -------
    str
        The stdout (or stderr) outputs as a single string.
    """
    try:
        command_data = json.loads(json_string)
        full_command = command_data.get("command")

        return execute_command_string(full_command)

    except json.JSONDecodeError as e:
        return f"Error: Unable to decode JSON string: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


def execute_command_string(command_string: str) -> str:
    """
    Execute a command and return stdout output (or stderr if there was an error)

    Parameters
    ----------
    command_string : str  # Changed the parameter name to match the docstring and function signature
        The shell command to execute.

    Returns
    -------
    str
        The stdout (or stderr) outputs as a single string.
    """
    try:
        process = subprocess.Popen(
            command_string,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            cwd="playground",
        )
        stdout, stderr = process.communicate(timeout=60)

        return_code = process.returncode

        if return_code == 0:
            return stdout
        else:
            return stderr

    except subprocess.TimeoutExpired:
        process.terminate()
        return "Error: Timeout reached (60 seconds)"
    except Exception as e:
        return f"Error: {str(e)}"


def save_code_to_file(code: str, file_path: str) -> None | str:
    """
    Save the given code to a file with the given filename.

    Parameters
    ----------
    code : str
        The code to save.
    filename : str, optional
        The name of the file to save the code to. Default is 'output.py'.
    """
    full_path = os.path.join(current_directory, "playground", file_path)
    try:
        mode = "a" if os.path.exists(full_path) else "w"
        with open(full_path, mode, encoding="utf-8") as f:
            f.write(code + "\n\n")
    except Exception as e:
        return f"Error: {str(e)}"


def refactor_code(
    modified_code: List[Dict[str, Union[int, str]]], file_path: str
) -> None:
    """
    Stores the refactored code in the given file.

    Parameters
    ----------
    modified_code : List[Dict[str, Union[int, str]]]
        A list of dictionaries containing start line, end line, and modified code.
    file_path : str
        The path to the file that needs to be refactored.

    """
    full_path = os.path.join(current_directory, "playground", file_path)

    with open(full_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for modification in modified_code:
        start_line = modification["start_line"]
        end_line = modification["end_line"]
        modified_chunk = modification["modified_code"].splitlines()

        # Remove original lines within the range
        del lines[start_line - 1 : end_line]

        # Insert the new modified_chunk lines
        for i, line in enumerate(modified_chunk):
            lines.insert(start_line - 1 + i, line + "\n")

    with open(full_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def split_code_into_chunks(
    file_path: str, chunk_size: int = 50
) -> List[Dict[str, Union[int, str]]]:
    """
    Split the given code into chunks, each containing at most max_lines lines of code.

    Parameters
    ----------
    code : str
        The code to split.
    max_lines : int, optional
        The maximum number of lines of code per chunk. Default is 20.

    Returns
    -------
    List[str]
        A list of code chunks.
    """
    full_path = os.path.join(current_directory, "playground", file_path)

    with open(full_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    for i in range(0, len(lines), chunk_size):
        start_line = i + 1
        end_line = min(i + chunk_size, len(lines))
        chunk = {
            "start_line": start_line,
            "end_line": end_line,
            "code": "".join(lines[i:end_line]),
        }
        chunks.append(chunk)
    return chunks


if __name__ == "__main__":
    pass
