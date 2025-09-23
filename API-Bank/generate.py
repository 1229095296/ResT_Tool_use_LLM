import os
import json
from vllm import LLM, SamplingParams
from tqdm import tqdm


if __name__ == "__main__":
    # add args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, default=None, help="Comma-separated model paths")
    args = parser.parse_args()

    # Convert the comma-separated string into a list
    args.model_paths = args.model_paths.split(",") if args.model_paths else []
    args.model_paths = [path.strip() for path in args.model_paths if path.strip()]
    
    for model_path in args.model_paths:
        
        model_name = model_path.split("verl_example_rlla/")[-1].replace("/", "_")
        save_path = f"PATH_TO_YOUR_SCORE_ROOT/{model_name}"
        os.makedirs(save_path, exist_ok=True)
        result_save_path = os.path.join(save_path, "result.json")
        score_save_path = os.path.join(save_path, "score.json")
        
        if os.path.exists(result_save_path):
            results = json.load(open(result_save_path, "r", encoding="utf-8"))
        else:
            results = {}
        if os.path.exists(score_save_path):
            scores = json.load(open(score_save_path, "r", encoding="utf-8"))
        else:
            scores = {}
        
        has_unfinished_flag = False
        for level in ["1", "2", "3"]:
            data_path = f"./level-{level}-api_processed.json"
            datas = json.load(open(data_path, "r", encoding="utf-8"))
            for id, data in tqdm(enumerate(datas)):
                gold = "Level" + level + "_" + str(id)
                if gold not in results:
                    has_unfinished_flag = True
                    break
        # if not has_unfinished_flag:
        #     print("All done for", model_path)
        #     continue
        
        print("Loading model with vLLM...")
        try:
            llm = LLM(
                model=model_path,
                tensor_parallel_size=int(os.getenv("WORLD_SIZE", 4)),
                gpu_memory_utilization=0.6,
                max_model_len=4096,
                trust_remote_code=True,  # Add this parameter to support models like Qwen3
                # dtype="bfloat16"
            )
            print(f"Successfully loaded model from {model_path}")
            print(f"vLLM version: {llm.__class__.__module__}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please check if the model path is correct and vLLM supports the model format")
            continue
        sampling_params = SamplingParams(
            max_tokens=4096, 
            temperature=0.0001,
        )
        
        log = {"success": 0, "fail": 0, "exist": 0}
        
        for level in ["1", "2", "3"]:
            data_path = f"./level-{level}-api_processed.json"
            datas = json.load(open(data_path, "r", encoding="utf-8"))
            for id, data in tqdm(enumerate(datas)):
                gold = "Level" + level + "_" + str(id)
                
                # if gold in results:
                #     continue
                if gold in scores and scores[gold]["score"] == 1:
                    print("Already scored: ", gold, "!!!!!")
                    log["exist"] += 1
                    continue
                
                sys_prompt = data["system"]
                user_prompt = data["user"]
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                try:
                    result = llm.chat(messages, sampling_params=sampling_params)
                    # vLLM 0.8.2 return value structure may differ, need compatibility handling
                    if hasattr(result, 'outputs') and len(result.outputs) > 0:
                        assistant_output = result.outputs[0].text.strip()
                    elif hasattr(result, 'choices') and len(result.choices) > 0:
                        assistant_output = result.choices[0].message.content.strip()
                    else:
                        # Compatibility with older versions
                        assistant_output = result[0].outputs[0].text.strip()
                    
                    thought = ""
                    all_tool_calls = []
                    thought = assistant_output.split("<think>")[-1].split("</think>")[0].strip()
                    
                    tool_calls = assistant_output.split("<tool_call>")[-1].split("</tool_call>")[0].strip()
                    tool_calls = tool_calls.strip().split("\n")
                    for tool_call in tool_calls:
                        if tool_call.strip() == "":
                            continue
                        try:
                            tool_call = json.loads(tool_call)
                            all_tool_calls.append(tool_call)
                        except:
                            pass
                    
                    record = {
                        "data": data,
                        "raw_output": assistant_output,
                        "thought": thought,
                        "tool_calls": all_tool_calls
                    }
                    results[gold] = record
                    
                    with open(result_save_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                    log["success"] += 1
                    
                except Exception as e:
                    print(f"Error in processing {gold}: {e}")
                    log["fail"] += 1
                    
                    record = {
                        "data": data,
                        "raw_output": assistant_output,
                        "thought": "",
                        "tool_calls": []
                    }
                    
                    results[gold] = record
                    continue
                
                print(log)
                
        with open(result_save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(log)
        print("All done for", model_path)