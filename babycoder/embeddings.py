import os
import csv
import openai
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast
from dotenv import load_dotenv
import time
from checksums import (
    calculate_checksums,
    compare_checksums,
    persist_checksums,
    load_checksums,
)

import logging

logger = logging.getLogger(__name__)

# Heavily derived from OpenAi's cookbook example
load_dotenv()

# the dir is the ./playground directory
REPOSITORY_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "playground"
)

EMBEDDINGS_MODEL = f"text-embedding-ada-002"
MAX_TOKENS_EMBEDDINGS_MODEL = 8192


class Embeddings:
    def __init__(self, workspace_path: str) -> None:
        """
        Initialize the Embeddings class.

        Parameters
        ----------
        workspace_path : str
            Path to the workspace.
        """
        self.workspace_path = workspace_path
        openai.api_key = os.getenv("OPENAI_API_KEY", "")

        self.DOC_EMBEDDINGS_MODEL = EMBEDDINGS_MODEL
        self.QUERY_EMBEDDINGS_MODEL = EMBEDDINGS_MODEL
        self.SEPARATOR = "\n* "
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.model_max_length = MAX_TOKENS_EMBEDDINGS_MODEL
        self.separator_len = len(self.tokenizer.tokenize(self.SEPARATOR))
        self.max_openai_calls_retries = 3

    def _remove_changed_or_deleted(self, compare: dict, workspace_path: str) -> None:
        """
        Remove stored embeddings for files that have been changed or deleted.

        Parameters
        ----------
        compare : dict
            Dictionary containing 'changed' files.
        workspace_path : str
            Path to the workspace.
        """
        # Load CSV files
        repo_info_file = os.path.join(
            workspace_path, "playground_data", "repository_info.csv"
        )
        embeddings_file = os.path.join(
            workspace_path, "playground_data", "doc_embeddings.csv"
        )

        try:
            df_repo_info = pd.read_csv(repo_info_file)
            df_repo_info = df_repo_info.set_index(["filePath", "lineCoverage"])
            for changed_file in compare["changed"]:
                # Drop rows from repository_info.csv if present
                if changed_file in df_repo_info.index:
                    df_repo_info = df_repo_info.sort_index()
                    df_repo_info = df_repo_info.drop(changed_file)
            df_repo_info.to_csv(repo_info_file)
        except FileNotFoundError:
            pass
        try:
            df_embeddings = pd.read_csv(embeddings_file)
            df_embeddings = df_embeddings.set_index(["filePath", "lineCoverage"])
            for changed_file in compare["changed"]:
                # Drop rows from doc_embeddings.csv if present
                if changed_file in df_embeddings.index:
                    df_embeddings = df_embeddings.sort_index()
                    df_embeddings = df_embeddings.drop(changed_file)
            df_embeddings.to_csv(embeddings_file)
        except FileNotFoundError:
            pass

    def compute_repository_embeddings(self):
        """
        Load stored checksums and calculate the repository's embeddings (update only for changed files).
        """
        old_checksums = load_checksums(
            os.path.join(self.workspace_path, "playground_data", "checksums.json")
        )
        current_checksums = calculate_checksums(REPOSITORY_PATH)

        # check which files have changed (or deleted)
        compare = compare_checksums(REPOSITORY_PATH, old_checksums)

        # remove stored embeddings for files that have changed or are deleted
        self._remove_changed_or_deleted(compare, self.workspace_path)

        # and now calculate embeddings for updated or new files
        info_and_last_file = {"info": None, "last_file_processed": None}
        while True:
            info_and_last_file = self.extract_info(
                REPOSITORY_PATH,
                ignore_files=compare["unchanged"],
                continue_from=info_and_last_file["last_file_processed"],
            )
            self.save_info_to_csv(info_and_last_file["info"])
            self.save_info_to_csv(
                info_and_last_file["info"],
                filename=os.path.join(
                    self.workspace_path,
                    "playground_data",
                    "repository_info_current_chunk.csv",
                ),
            )


            if info_and_last_file["last_file_processed"] is None:
                # we are done - all files processed
                # now update and generate doc embeddings
                df = pd.read_csv(
                    os.path.join(
                        self.workspace_path,
                        "playground_data",
                        "repository_info.csv",
                    )
                )
                df_changed = df[df["filePath"].isin(compare["changed"])]
                df_changed = df_changed.set_index(["filePath", "lineCoverage"])
                df = df.set_index(["filePath", "lineCoverage"])
                self.df = df
                context_embeddings = self.compute_doc_embeddings(df_changed)
                self.save_doc_embeddings_to_csv(
                    context_embeddings,
                    df_changed,
                    os.path.join(
                        self.workspace_path, "playground_data", "doc_embeddings.csv"
                    ),
                )
                # stop (the while true loop)
                break

        persist_checksums(
            current_checksums,
            os.path.join(self.workspace_path, "playground_data", "checksums.json"),
        )
        try:
            self.document_embeddings = self.load_embeddings(
                os.path.join(
                    self.workspace_path, "playground_data", "doc_embeddings.csv"
                )
            )
        except:
            pass

    def extract_info(
        self, repository_path: str, ignore_files: list, continue_from: str = None
    ) -> dict:
        """
        Extract information from files in the repository in chunks
        and return a list of (filePath, lineCoverage, chunkContent).

        Parameters
        ----------
        REPOSITORY_PATH : str
            Path to the repository.
        ignore_files : list
            List of files to ignore.
        continue_from : str, optional
            File path to start processing from, default is None.

        Returns
        -------
        dict
            A dictionary containing:
                - "info": list of (filePath, lineCoverage, chunkContent)
                - "last_file_processed": the last processed file path
        """
        # Initialize an empty list to store the information
        info = []
        last_file_processed = None
        tokens_count = 0
        max_exceeded = False

        LINES_PER_CHUNK = 60

        # Collect all file paths
        file_paths = []

        # Iterate over directories and files
        for dirpath, dirnames, filenames in os.walk(repository_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                file_paths.append(file_path)

        # Sort the file paths
        file_paths.sort()
        file_paths = list(set(file_paths) - set(ignore_files))

        # Process the files in sorted order
        for file_path in file_paths:
            if not max_exceeded:
                if file_path == continue_from:
                    continue_from = None
                    continue
                elif continue_from is None:
                    # Read the contents of the file
                    with open(file_path, "r", encoding="utf-8") as f:
                        try:
                            contents = f.read()
                        except:
                            continue

                    # Split the contents into lines
                    lines = contents.split("\n")
                    # Ignore empty lines
                    lines = [line for line in lines if line.strip()]
                    # Split the lines into chunks of LINES_PER_CHUNK lines
                    chunks = [
                        lines[i : i + LINES_PER_CHUNK]
                        for i in range(0, len(lines), LINES_PER_CHUNK)
                    ]
                    # Iterate through the chunks
                    for i, chunk in enumerate(chunks):
                        # Join the lines in the chunk back into a single string
                        chunk = "\n".join(chunk)
                        # Get the first and last line numbers
                        first_line = i * LINES_PER_CHUNK + 1
                        last_line = first_line + len(chunk.split("\n")) - 1
                        line_coverage = (first_line, last_line)
                        # Add the file path, line coverage, and content to the list
                        info.append((file_path, line_coverage, chunk))
                        tokens_count += len(self.tokenizer.tokenize(chunk))
                        if tokens_count > int(0.5 * MAX_TOKENS_EMBEDDINGS_MODEL):
                            max_exceeded = True

                    last_file_processed = file_path
                else:
                    if file_path in ignore_files:
                        logger.debug(
                            f"Skipping file {file_path} - in list of unchanged files"
                        )
                    else:
                        logger.debug(f"Skipping file {file_path} - already processed")

            else:
                logger.debug(
                    f"Last processed file {file_path}, skipping rest because max token count exceeded ({tokens_count} > {MAX_TOKENS_EMBEDDINGS_MODEL}))"
                )
                return {"info": info, "last_file_processed": last_file_processed}

        # Return the list of information
        return {"info": info, "last_file_processed": None}

    def save_info_to_csv(self, info, filename=None):
        """
        Save file information to a CSV file.

        Parameters
        ----------
        info : list
            List containing file info as tuples.
        filename : str, optional
            The output CSV file name; default is None (in that case playground_data/repository.csv is used).
        """
        os.makedirs(os.path.join(self.workspace_path, "playground_data"), exist_ok=True)

        # for default repository: append, for different store (temporary store): write
        if not filename:
            filename = os.path.join(
                self.workspace_path, "playground_data", "repository_info.csv"
            )
            if os.path.isfile(filename):
                mode = "a"
            else:
                mode = "w"
        else:
            filename = filename
            mode = "w"

        with open(filename, mode=mode, newline="") as csvfile:
            writer = csv.writer(csvfile)
            if mode == "w":
                writer.writerow(["filePath", "lineCoverage", "content"])
            for file_path, line_coverage, content in info:
                # Write a row for each chunk of data
                writer.writerow([file_path, line_coverage, content])

    def get_relevant_code_chunks(
        self, task_description: str, task_context: str
    ) -> list:
        """
        Get the most relevant code chunks for a given task description and context.

        Parameters
        ----------
        task_description : str
            Text description of the task.
        task_context : str
            Context to use for the task.

        Returns
        -------
        list
            List of most relevant code chunks containing:
                - "relevance_score": The similarity score.
                - "filePath": The file path of the code chunk.
                - "(from_line,to_line)": The line coverage of the code chunk.
                - "content": The content of the code chunk.
        """
        query = task_description + "\n" + task_context
        most_relevant_document_sections = (
            self.order_document_sections_by_query_similarity(
                query, self.document_embeddings
            )
        )
        max_score = most_relevant_document_sections[0][0]
        selected_chunks = []
        for score, file_and_line in most_relevant_document_sections:
            try:
                file_path = file_and_line[0]
                line_coverage = file_and_line[1]
                if (file_path, line_coverage) in self.df.index:
                    relevant_chunk = self.df.loc[(file_path, line_coverage)]
                    if score > 0.9 * max_score:
                        selected_chunks.append(
                            {
                                "relevance_score": score,
                                "filePath": file_path,
                                "(from_line,to_line)": line_coverage,
                                "content": relevant_chunk["content"],
                            }
                        )
                if len(selected_chunks) >= 3:
                    break
            except Exception as e:
                err = f"Error: {str(e)}"
                print(err)
                logger.error(err)
                pass

        return selected_chunks

    def get_embedding(self, text: str, model: str) -> list[float]:
        """
        Get the embedding for a given text using the specified model.

        Parameters
        ----------
        text : str
            Text to be embedded.
        model : str
            OpenAI model to be used for embedding.

        Returns
        -------
        list[float]
            Embedding vector.
        """
        global openai_calls_retried

        try:
            result = openai.Embedding.create(model=model, input=text)
            openai_calls_retried = 0
            return result["data"][0]["embedding"]
        except Exception as e:
            # try again
            if openai_calls_retried < self.max_openai_calls_retries:
                openai_calls_retried += 1
                print(
                    f"Error calling OpenAI embeddings. Retrying {openai_calls_retried} of {self.max_openai_calls_retries}..."
                )
                time.sleep(1)
                return self.get_embedding(text, model)

    def get_doc_embedding(self, text: str) -> list[float]:
        """
        Get the document embedding for a given text using the DOC_EMBEDDINGS_MODEL.

        Parameters
        ----------
        text : str
            Text to be embedded.

        Returns
        -------
        list[float]
            Embedding vector.
        """
        return self.get_embedding(text, self.DOC_EMBEDDINGS_MODEL)

    def get_query_embedding(self, text: str) -> list[float]:
        """
        Get the query embedding for a given text using the QUERY_EMBEDDINGS_MODEL.

        Parameters
        ----------
        text : str
            Text to be embedded.

        Returns
        -------
        list[float]
            Embedding vector.
        """
        return self.get_embedding(text, self.QUERY_EMBEDDINGS_MODEL)

    def compute_doc_embeddings(
        self, df: pd.DataFrame
    ) -> dict[tuple[str, str], list[float]]:
        """
        Calculate the repository's embeddings (only for modified files).
        """
        embeddings = {}
        for idx, r in df.iterrows():
            # Wait one second before making the next call to the OpenAI Embeddings API
            # print("Waiting one second before embedding next row\n")
            time.sleep(1)
            embeddings[idx] = self.get_doc_embedding(r.content.replace("\n", " "))
        return embeddings

    def save_doc_embeddings_to_csv(
        self, doc_embeddings: dict, df: pd.DataFrame, csv_filepath: str
    ) -> None:
        """
        Save document embeddings to a CSV file.

        Parameters
        ----------
        doc_embeddings : dict
            Dictionary containing document embeddings.
        df : pd.DataFrame
            Dataframe containing code chunks information.
        csv_filepath : str
            Filepath for the CSV file.
        """
        # Get the dimensionality of the embedding vectors from the first element in the doc_embeddings dictionary
        if len(doc_embeddings) == 0:
            return

        EMBEDDING_DIM = len(list(doc_embeddings.values())[0])

        # Create a new dataframe with the filePath, lineCoverage, and embedding vector columns
        embeddings_df = pd.DataFrame(
            columns=["filePath", "lineCoverage"]
            + [f"{i}" for i in range(EMBEDDING_DIM)]
        )

        # Iterate over the rows in the original dataframe
        for idx, _ in df.iterrows():
            # Get the embedding vector for the current row
            embedding = doc_embeddings[idx]
            # Create a new row in the embeddings dataframe with the filePath, lineCoverage, and embedding vector values
            row = [idx[0], idx[1]] + embedding
            embeddings_df.loc[len(embeddings_df)] = row

        # Check if the file exists
        file_exists = os.path.isfile(csv_filepath)

        # Save the embeddings DataFrame to a CSV file
        if file_exists:
            embeddings_df.to_csv(csv_filepath, index=False, mode="a", header=False)
        else:
            embeddings_df.to_csv(csv_filepath, index=False, mode="w")

    def vector_similarity(self, x: list[float], y: list[float]) -> float:
        """
        Compute the similarity between two vectors.

        Parameters
        ----------
        x : list[float]
            First vector.
        y : list[float]
            Second vector.

        Returns
        -------
        float
            The similarity score.
        """
        return np.dot(np.array(x), np.array(y))

    def order_document_sections_by_query_similarity(
        self, query: str, contexts: dict[(str, str), np.array]
    ) -> list[(float, (str, str))]:
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections.

        Parameters
        ----------
        query : str
            The query text.
        contexts : dict[(str, str), np.array]
            Dictionary of document sections with their embeddings.

        Returns
        -------
        list[(float, (str, str))]
            The list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_query_embedding(query)

        document_similarities = sorted(
            [
                (self.vector_similarity(query_embedding, doc_embedding), doc_index)
                for doc_index, doc_embedding in contexts.items()
            ],
            reverse=True,
        )

        return document_similarities

    def load_embeddings(self, fname: str) -> dict[tuple[str, str], list[float]]:
        """
        Load embeddings from a CSV file.

        Parameters
        ----------
        fname : str
            Filepath for the CSV file.

        Returns
        -------
        dict[tuple[str, str], list[float]]
            Dictionary containing embeddings.
        """
        df = pd.read_csv(fname, header=0)
        max_dim = max(
            [int(c) for c in df.columns if c != "filePath" and c != "lineCoverage"]
        )
        return {
            (r.filePath, r.lineCoverage): [r[str(i)] for i in range(max_dim + 1)]
            for _, r in df.iterrows()
        }
