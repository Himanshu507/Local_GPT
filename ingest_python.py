import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import click
import torch
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    VENV_DIRECTORY
)

class PythonLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        with open(self.file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        document = Document(page_content=content, metadata={"file_path": self.file_path})
        return [document]

DOCUMENT_MAP.update({
    ".py": PythonLoader
})

def get_embeddings(device_type: str):
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    model.to(device_type)
    return tokenizer, model

def file_log(logentry):
    with open("file_ingest.log", "a") as file:
        file.write(logentry + "\n")
    print(logentry + "\n")

def load_single_document(file_path: str) -> Document:
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            file_log(file_path + " loaded.")
            loader = loader_class(file_path)
        else:
            file_log(file_path + " document type is undefined.")
            raise ValueError("Document type is undefined")
        document = loader.load()[0]
        document.metadata["file_path"] = file_path
        return document
    except Exception as ex:
        file_log("%s loading error: \n%s" % (file_path, ex))
        return None

def load_document_batch(filepaths):
    logging.info("Loading document batch")
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        if futures is None:
            file_log(name + " failed to submit")
            return None
        else:
            data_list = [future.result() for future in futures]
            return (data_list, filepaths)

def load_documents(source_dir: str, venv_dir: str) -> list[Document]:
    paths = []
    for root, _, files in os.walk(source_dir):
        if '/.' in root or root.startswith('.') or venv_dir in root:
            continue
        for file_name in files:
            if file_name.startswith('.'):
                continue
            file_extension = os.path.splitext(file_name)[1]
            if file_extension == '.py':
                source_file_path = os.path.join(root, file_name)
                paths.append(source_file_path)

    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        for i in range(0, len(paths), chunksize):
            filepaths = paths[i: (i + chunksize)]
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        for future in as_completed(futures):
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log("Exception: %s" % (ex))

    return docs

def split_documents(documents: list[Document]) -> list[Document]:
    python_docs = [doc for doc in documents if doc is not None]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return [split_doc for doc in python_docs for split_doc in text_splitter.split_documents([doc])]

def embed_code_with_metadata(documents, embeddings):
    tokenizer, model = embeddings
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model.to(device)

    vectors = []
    metadata_list = []
    for doc in tqdm(documents, desc="Processing Documents"):
        chunks = doc.page_content
        file_path = doc.metadata["file_path"]
        inputs = tokenizer(chunks, return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
            else:
                embedding = outputs[0].mean(dim=1).squeeze().cpu().numpy().tolist()
        metadata = {"file_name": os.path.basename(file_path)}
        vectors.append(embedding)
        metadata_list.append(metadata)
    return vectors, metadata_list

@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
        ]
    ),
    help="The device type to use for computation",
)
def main(device_type: str):
    logging.basicConfig(level=logging.INFO)

    file_log(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY, VENV_DIRECTORY)
    file_log(f"Loaded {len(documents)} documents")

    file_log("Splitting documents into chunks")
    split_docs = split_documents(documents)
    file_log(f"Split into {len(split_docs)} chunks")

    embeddings = get_embeddings(device_type)
    file_log("Embedding code with metadata")
    vectors, metadata_list = embed_code_with_metadata(split_docs, embeddings)

    # Check lengths of the lists
    file_log(f"Number of split docs: {len(split_docs)}, vectors: {len(vectors)}, metadata: {len(metadata_list)}")

    if len(split_docs) != len(vectors) or len(split_docs) != len(metadata_list):
        file_log("Error: Lengths of split_docs, vectors, and metadata_list do not match")
        return

    file_log(f"Storing embeddings in ChromaDB at {PERSIST_DIRECTORY}")
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY
    )

    db.add_texts(
        texts=[doc.page_content for doc in split_docs],
        metadatas=metadata_list,
        embeddings=vectors
    )
    db.persist()

    file_log("Ingestion complete")

if __name__ == "__main__":
    main()
