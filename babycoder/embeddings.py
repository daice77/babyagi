import os
import csv 
import shutil
import openai
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast
from dotenv import load_dotenv
import time
from checksums import calculate_checksums, compare_checksums, persist_checksums, load_checksums

import logging

logger = logging.getLogger(__name__)

MAX_TOKENS = 4096   # model max token is 8192, but we need to leave some space for the query

# Heavily derived from OpenAi's cookbook example
load_dotenv()

# the dir is the ./playground directory
REPOSITORY_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "playground")

class Embeddings:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        openai.api_key = os.getenv("OPENAI_API_KEY", "")

        self.DOC_EMBEDDINGS_MODEL = f"text-embedding-ada-002"
        self.QUERY_EMBEDDINGS_MODEL = f"text-embedding-ada-002"

        self.SEPARATOR = "\n* "

        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.model_max_length = MAX_TOKENS

        self.separator_len = len(self.tokenizer.tokenize(self.SEPARATOR))

        self.max_openai_calls_retries = 3

    def remove_changed_or_deleted(compare, workspace_path):
        # Load CSV files
        repo_info_file = os.path.join(workspace_path, 'playground_data', 'repository_info.csv')
        embeddings_file = os.path.join(workspace_path, 'playground_data', 'doc_embeddings.csv')

        df_repo_info = pd.read_csv(repo_info_file)
        df_embeddings = pd.read_csv(embeddings_file)

        # Set indexes for easier row dropping
        df_repo_info = df_repo_info.set_index(["filePath", "lineCoverage"])
        df_embeddings = df_embeddings.set_index(["filePath", "lineCoverage"])

        # Remove stored embeddings for files that have changed or are deleted
        for changed_file in compare['changed']:
            # Drop rows from repository_info.csv
            if changed_file in df_repo_info.index:
                df_repo_info = df_repo_info.drop(changed_file)

            # Drop rows from doc_embeddings.csv
            if changed_file in df_embeddings.index:
                df_embeddings = df_embeddings.drop(changed_file)

        # Save modified dataframes back to CSV files
        df_repo_info.to_csv(repo_info_file)
        df_embeddings.to_csv(embeddings_file)
        
    def compute_repository_embeddings(self):
        # Load stored checksums and calculate checksums for the current repository  
        old_checksums = load_checksums(os.path.join(self.workspace_path, 'playground_data', 'checksums.json'))
        current_checksums = calculate_checksums(REPOSITORY_PATH)

        compare = compare_checksums(REPOSITORY_PATH, old_checksums)

        # remove stored embeddings for files that have changed or are deleted
        self.remove_changed_or_deleted(compare, self.workspace_path)

        # Extract information from files in the repository in chunks
        info_and_last_file["last_file_processed"] = None
        while True:
            info_and_last_file = self.extract_info(REPOSITORY_PATH, ignore_files=compare['unchanged'], continue_from=info_and_last_file["last_file_processed"])
            self.save_info_to_csv(info_and_last_file["info"])
            self.save_info_to_csv(info_and_last_file["info"], filename=os.path.join(self.workspace_path, 'playground_data', 'repository_info_current_chunk.csv'))

            df = pd.read_csv(os.path.join(self.workspace_path, 'playground_data', 'repository_info_current_chunk.csv'))
            df = df.set_index(["filePath", "lineCoverage"])
            self.df = df
            context_embeddings = self.compute_doc_embeddings(df)
            self.save_doc_embeddings_to_csv(context_embeddings, df, os.path.join(self.workspace_path, 'playground_data', 'doc_embeddings.csv'))

            if info_and_last_file["last_file_processed"] is None:
                break
            
        persist_checksums(current_checksums, os.path.join(self.workspace_path, 'playground_data', 'checksums.json'))
        try:
            self.document_embeddings = self.load_embeddings(os.path.join(self.workspace_path, 'playground_data', 'doc_embeddings.csv'))
        except:
            pass


    # Extract information from files in the repository in chunks
    # Return a list of [filePath, lineCoverage, chunkContent]
    def extract_info(self, REPOSITORY_PATH, ignore_files: list, continue_from = None):
        # Initialize an empty list to store the information
        info = []

        tokens_count = 0
        max_exceeded = False
        
        LINES_PER_CHUNK = 60

        # Iterate through the files in the repository
        for root, dirs, files in os.walk(REPOSITORY_PATH):
            for file in files:

                if (not max_exceeded) and (file not in ignore_files):
                    file_path = os.path.join(root, file)

                    if (file_path == continue_from) or (continue_from is None):
                        continue_from = None

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
                                lines[i:i+LINES_PER_CHUNK]
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
                            info.append((os.path.join(root, file), line_coverage, chunk))
                            tokens_count += len(self.tokenizer.tokenize(chunk))
                            if tokens_count > MAX_TOKENS:
                                max_exceeded = True
                        
                        last_file_processed = file_path
                    else:
                        logger.info(f"Skipping file {file_path} - already processed or not in the list of unchanged files")
                else:
                    logger.info(f"Last processed file {file_path}, skipping rest because max token count exceeded ({tokens_count} > {MAX_TOKENS}))")
                    return {"info": info, "last_file_processed": last_file_processed}
            
        # Return the list of information
        return {"info": info, "last_file_processed": None}

    def save_info_to_csv(self, info, filename = None):
        # Open a CSV file for writing
        os.makedirs(os.path.join(self.workspace_path, "playground_data"), exist_ok=True)
        
        if not filename:
            mode = "w"
            filename = os.path.join(self.workspace_path, 'playground_data', 'repository_info.csv')
        else:
            mode = "a"
            filename = filename

        with open(filename, mode=mode, newline="") as csvfile:
            # Create a CSV writer
            writer = csv.writer(csvfile)
            # Write the header row
            writer.writerow(["filePath", "lineCoverage", "content"])
            # Iterate through the info
            for file_path, line_coverage, content in info:
                # Write a row for each chunk of data
                writer.writerow([file_path, line_coverage, content])

    def get_relevant_code_chunks(self, task_description: str, task_context: str):
        query = task_description + "\n" + task_context
        most_relevant_document_sections = self.order_document_sections_by_query_similarity(query, self.document_embeddings)
        selected_chunks = []
        for _, section_index in most_relevant_document_sections:
            try:
                document_section = self.df.loc[section_index]
                selected_chunks.append(self.SEPARATOR + document_section['content'].replace("\n", " "))
                if len(selected_chunks) >= 2:
                    break
            except:
                pass

        return selected_chunks

    def get_embedding(self, text: str, model: str) -> list[float]:
        global openai_calls_retried

        try:
            result = openai.Embedding.create(
            model=model,
            input=text
            )
            openai_calls_retried = 0
            return result["data"][0]["embedding"]
        except Exception as e:
            # try again
            if openai_calls_retried < self.max_openai_calls_retries:
                openai_calls_retried += 1
                print(f"Error calling OpenAI embeddings. Retrying {openai_calls_retried} of {max_openai_calls_retries}...")
                return self.get_embedding(text, model)
            

    def compute_doc_embeddings(self, df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
        """
        Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

        Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        embeddings = {}
        for idx, r in df.iterrows():
            # Wait one second before making the next call to the OpenAI Embeddings API
            # print("Waiting one second before embedding next row\n")
            time.sleep(1)
            embeddings[idx] = self.get_doc_embedding(r.content.replace("\n", " "))
        return embeddings

    def save_doc_embeddings_to_csv(self, doc_embeddings: dict, df: pd.DataFrame, csv_filepath: str):
        # Get the dimensionality of the embedding vectors from the first element in the doc_embeddings dictionary
        if len(doc_embeddings) == 0:
            return

        EMBEDDING_DIM = len(list(doc_embeddings.values())[0])

        # Create a new dataframe with the filePath, lineCoverage, and embedding vector columns
        embeddings_df = pd.DataFrame(columns=["filePath", "lineCoverage"] + [f"{i}" for i in range(EMBEDDING_DIM)])

        # Iterate over the rows in the original dataframe
        for idx, _ in df.iterrows():
            # Get the embedding vector for the current row
            embedding = doc_embeddings[idx]
            # Create a new row in the embeddings dataframe with the filePath, lineCoverage, and embedding vector values
            row = [idx[0], idx[1]] + embedding
            embeddings_df.loc[len(embeddings_df)] = row

        # Save the embeddings dataframe to a CSV file
        embeddings_df.to_csv(csv_filepath, index=False, append=True)

    def vector_similarity(self, x: list[float], y: list[float]) -> float:
        return np.dot(np.array(x), np.array(y))

    def order_document_sections_by_query_similarity(self, query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 
        
        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_query_embedding(query)
        
        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)
        
        return document_similarities
    
    def load_embeddings(self, fname: str) -> dict[tuple[str, str], list[float]]:       
        df = pd.read_csv(fname, header=0)
        max_dim = max([int(c) for c in df.columns if c != "filePath" and c != "lineCoverage"])
        return {
            (r.filePath, r.lineCoverage): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
        }