model_path = "models/functiongemma-270m-it"

import json
import torch
from datasets import Dataset, load_dataset
from transformers.utils import get_json_schema
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Tool Definitions ---
def search_knowledge_base(query: str) -> str:
    """
    Search internal company documents, policies and project data.

    Args:
        query: query string
    """
    return "Internal Result"

def search_google(query: str) -> str:
    """
    Search public information.

    Args:
        query: query string
    """
    return "Public Result"


TOOLS = [get_json_schema(search_knowledge_base), get_json_schema(search_google)]

DEFAULT_SYSTEM_MSG = "You are a model that can do function calling with the following functions"

def create_conversation(sample):
  return {
      "messages": [
          {"role": "developer", "content": DEFAULT_SYSTEM_MSG},
          {"role": "user", "content": sample["user_content"]},
          {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": sample["tool_name"], "arguments": json.loads(sample["tool_arguments"])} }]},
      ],
      "tools": TOOLS
  }

dataset = load_dataset("bebechien/SimpleToolCalling", split="train")

# Convert dataset to conversational format
dataset = dataset.map(create_conversation, remove_columns=dataset.features, batched=False)

# Split dataset into 50% training samples and 50% test samples
dataset = dataset.train_test_split(test_size=0.5, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype = "auto",
    device_map = "auto"
)

def check_success_rate(model):
    success_count = 0
    for idx, item in enumerate(dataset["test"]):
        messages = [
            item["messages"][0],
            item["messages"][1]
        ]
        inputs = tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        out = model.generate(**inputs.to(model.device), pad_token_id=tokenizer.eos_token_id, max_new_tokens=128)
        output = tokenizer.decode(out[0][len(inputs["input_ids"][0]) :], skip_special_tokens=False)
        
        print(f"{idx+1} Prompt: {item["messages"][1]["content"]}")
        print(f"Output: {output}")

        expected_tool = item["messages"][2]["tool_calls"][0]["function"]["name"]
        other_tool = "search_knowledge_base" if expected_tool == "search_google" else "search_google"

        if expected_tool in output and other_tool not in output:
            print("Correct")
            success_count += 1
        elif expected_tool not in output:
            print(f"Wrong expected: {expected_tool}")
        else:
            print("Wrong, halucinated tool")
    return success_count/len(dataset["test"])

if __name__ == "__main__":
    print("Device:", model.device)
    print("Dtype:", model.dtype)
    print("\n--- FORMATED DATESET MESSAGE ---")
    debug_message = tokenizer.apply_chat_template(dataset["train"][0]["messages"], tools=dataset["train"][0]["tools"], add_generation_prompt=False, tokenize=False)
    print(debug_message)
    print("\n\nBase model achieved success rate of", check_success_rate(model))
