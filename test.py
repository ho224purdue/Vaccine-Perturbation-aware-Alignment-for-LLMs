from huggingface_hub import login
import torch
from transformers import pipeline
import time

def initial_tests():
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(hasattr(torch, 'compiler'))

def initialize():
    # Use pipeline
    pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device = -1)
    return pipe

def query(pipe):
    start_time = time.perf_counter()
    # Example usage
    response = pipe("Hi there! How do you do?", max_length=100)
    print(response[0]['generated_text'])
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Query execution time: {execution_time // 60} minutes {execution_time % 60} seconds")

if __name__ == "__main__":
    # initial_tests()
    pipe = initialize()
    query(pipe)
