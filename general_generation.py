import json
import argparse
from huggingface_hub import login
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_jsonl_indent(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_jsonl(file_path, data):
    with open(file_path, 'a') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

login(token='')

sampling_params = SamplingParams(max_tokens=8192, temperature=0)
model_name = 'google/gemma-3-27b-it'

def extract_output(batch_outputs, batch_seed_data):
    parsed_outputs = []
    for idx, output in enumerate(batch_outputs):
            s = output.outputs[0].text
            try:
                match = re.search(r'```json\s*(.*?)\s*```', s, re.DOTALL)
                if match:
                    s1 = match.group(1)
                    try:
                        o = json.loads(s1)
                    except json.JSONDecodeError:
                        o = s
            except json.JSONDecodeError:
                o = s
            
            parsed_outputs.append({
                "output": o,
                **batch_seed_data[idx]
            })

            if idx % 500 == 0:
                print(f"Processed {idx} outputs")
                print(f"Output: {o}")
                print(batch_seed_data[idx])
                print("-"*50)
    return parsed_outputs

def generate(prompt_path, seed_data_path, output_name):
    seed_data = load_jsonl_indent(seed_data_path)
    seed_data = seed_data[len(seed_data)//2:]
    prompts = load_json(prompt_path)

    system_prompt = prompts['system']
    user_prompt = prompts['user']

    print(system_prompt)
    print("-"*20)
    print(user_prompt)

    conversations = [
        [{"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(**seed)}]
        for seed in seed_data
    ]

    llm = LLM(model=model_name, tensor_parallel_size=2, 
              enable_prefix_caching=True, gpu_memory_utilization=0.97,
              enable_chunked_prefill=True, max_num_batched_tokens=256,
              max_num_seqs=256, max_model_len=2048)
    
    output_file = f"/datadrive/pavan/CurLL/data/skill_graph/{output_name}.jsonl"
    total_tokens = 0
    batch_size = 1024
    all_outputs = []

    try:
        for i in tqdm(range(0, len(conversations), batch_size)):
            batch_conversations = conversations[i:i+batch_size]
            batch_seed_data = seed_data[i:i+batch_size]
            batch_outputs = llm.chat(messages=batch_conversations, 
                                    sampling_params=sampling_params, 
                                    use_tqdm=True)
            
            all_outputs.extend(extract_output(batch_outputs, batch_seed_data))
    except Exception as e:
        print(f"Error during generation: {e}")

    #load the file if it exists
    try:
        with open(output_file, "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    # save the outputs to a file
    with open(output_file, "w") as f:
        existing_data.extend(all_outputs)
        json.dump(existing_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt_path', type=str, default='prompts.json')
    parser.add_argument('-s', '--seed_data_path', type=str, default='seed_data.json')
    parser.add_argument('-o', '--output_name', type=str, default='output')

    args = parser.parse_args()


    generate(args.prompt_path, args.seed_data_path, args.output_name)

if __name__ == "__main__":
    main()


