import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.response_generation import generate_response
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

logger.info("Loading model... This may take a few moments.")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# Create pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

system_message = {"role": "system", "content": "You are a helpful but snarky AI assistant."}


logger.info("Model loaded. Ready for interaction!")


while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        logger.info("Exiting the program.")
        print("Goodbye!")
        break

    print("\nAI:")
    response = generate_response(system_message, tokenizer, pipeline, logger, user_input)
    print("\n" + "-"*50)
