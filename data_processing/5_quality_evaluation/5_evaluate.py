import base64
from openai import OpenAI
import time
import os
from tqdm import tqdm
# Set OpenAI's API key and API base to use vLLM's API server.

openai_api_key = " " #set the api key
openai_api_base = " " #set the api base

model_name = "gpt-4o"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def call_api(images, prompt):
    """
        To support multiple images, should add "--limit-mm-per-prompt" option to the "python -m vllm.entrypoints.openai.api_server" command.
    """
    content_list = []
    for image in images:
        content_list.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image
                },
            },
        )
    content_list.append({"type": "text", "text": prompt})
    
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": content_list
            },
        ],
    )
    return chat_response


#--------
import base64
import io
import json
from PIL import Image


def pil_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")  # Save the image to the buffer in PNG format (you can change the format if needed)
    image_bytes = buffered.getvalue()
    encoded_image_text = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image;base64,{encoded_image_text}"

def resize_if_need(ori_img, tar_img, max_side_limit=1500):
    w_ori, h_ori = ori_img.size
    w_tar, h_tar = tar_img.size
    new_w, new_h = min(w_ori, w_tar), min(h_ori, h_tar)

    curr_max_side = max(new_w, new_h)
    if curr_max_side > max_side_limit:
        print(f"current max side '{curr_max_side}' too large.")
        scale_factor = max_side_limit / curr_max_side
        
        # Calculate new dimensions
        new_w = int(new_w * scale_factor)
        new_h = int(new_h * scale_factor)

    if w_ori > new_w:
        ori_img = ori_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"resize original image: {w_ori}*{h_ori} to {new_w}*{new_h}")
    if w_tar > new_w:
        tar_img = tar_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"resize target image: {w_tar}*{h_tar} to {new_w}*{new_h}")
    return ori_img, tar_img


def extract_json_delimiter(chat_response, delimiter='||V^=^V||'):
    try:
        result = chat_response.choices[0].message.content
    except Exception as e:
        print(f"Parse chat_response failed, ")
    if type(result) == list:
        print("Just pick index 0 element of the list")
        result = result[0]
    try:
        json_part = result.split(delimiter)[1].split(delimiter)[0].strip()
        json_data = json.loads(json_part)  # dict
        return json_data
    except Exception as e:
        print(f"Error parsing the string as JSON: {e}.\nThe string is {result}")
        return None


def extract_json(chat_response):
    try:
        result = chat_response.choices[0].message.content
        print(f"++++\n{result}\n++++")
    except Exception as e:
        print(f"Parse chat_response failed, ")
    
    if type(result) == list:
        print("Just pick index 0 element of the list")
        result = result[0]

    return extract_json_from_text(result)

def extract_json_from_text(text):
    if type(text) in [list, tuple]:
        text = text[0]
    try:
        text = text.replace('```json', '')
        text = text.replace('```', '')
        json_data = json.loads(text)  # dict
        return json_data
    except Exception as e:
        print(f"Error parsing the string as JSON: {e}.\nThe string is {text}")
        return None


import sys
from gpt_api_call import GPT4API

gpt4api = GPT4API(model=model_name)
def call_GPT(prompt, base64_images):
    resp = gpt4api.invoke(prompt, base64_images, img_detail='auto')
    if resp:
        text, response_time = resp
        print(response_time, text)
        return text, response_time
    else:
        print("Call GPT return None")
        return None

##------------ Eval the SC score
prompt_SC_fp = '../prompt_SC.txt'
with open(prompt_SC_fp) as f:
    prompt_SC = f.read()

def eval_SC(edit_instruction, ori_img_pil, tar_img_pil):
    if type(edit_instruction) == list:
        edit_instruction = edit_instruction[0]
    prompt = prompt_SC.replace('<instruction>', edit_instruction)
    images = [pil_image_to_base64(pil_image) for pil_image in (ori_img_pil, tar_img_pil)]
    
    if False:
        t0 = time.time()
        chat_response = call_api(images, prompt)
        print(f"Take {time.time() - t0:.2f}s to get SC score response")
        json_object = extract_json(chat_response)
    else:
        text = call_GPT(prompt, images)
        json_object = extract_json_from_text(text)
        
    try:
        qwen_VIE_SC = {
            'instruction_follow': json_object['score'][0],
            'non-overedit': json_object['score'][1],
            'reasoning': json_object['reasoning']
        }
        return qwen_VIE_SC
    except Exception as e:
        print(f"Parse keys for Semantic Consistency failed: {e}")
        print(f"the json object is {json_object}")
        return None


##------------ Eval the PQ score
prompt_PQ_fp = '../prompt_PQ.txt'
with open(prompt_PQ_fp) as f:
    prompt_PQ = f.read()

def eval_PQ(tar_img_pil):
    images = [pil_image_to_base64(tar_img_pil)]
    
    if False:
        t0 = time.time()
        chat_response = call_api(images, prompt_PQ)
        print(f"Take {time.time() - t0:.2f}s to get PQ score response.")
        json_object = extract_json(chat_response)
    else:
        text = call_GPT(prompt_PQ, images)
        json_object = extract_json_from_text(text)
    try:
        qwen_VIE_PQ = {
            'naturalness': json_object['score'][0],
            'no_artifacts': json_object['score'][1],
            'reasoning': json_object['reasoning']
        }
        return qwen_VIE_PQ
    except Exception as e:
        print(f"Parse keys for Perceptual Quality failed: {e}")
        print(f"the json object is {json_object}")
        return None


##------------ Read the items ----
import os

def add_scores_to_item(item):
    origin_img_path = item['origin_img_path']
    target_img_path = item['target_img_path']
    # print(f"\norigin_img_path {origin_img_path}, target_img_path {target_img_path}, instruction {item['instruction']}")
    
    edit_instruction = item['instruction']
    if type(edit_instruction) == list:
        edit_instruction = edit_instruction[0]

    need_run_sc = 'qwen_VIE_SC' not in item or item['qwen_VIE_SC'] is None
    need_run_pq = 'qwen_VIE_PQ' not in item or item['qwen_VIE_PQ'] is None

    if need_run_sc or need_run_pq:
        ori_img_pil = Image.open(origin_img_path)
        tar_img_pil = Image.open(target_img_path)
        ori_img_pil, tar_img_pil = resize_if_need(ori_img_pil, tar_img_pil)  # PIL Image

    if need_run_sc:
        qwen_VIE_SC = eval_SC(edit_instruction, ori_img_pil, tar_img_pil)
        item['qwen_VIE_SC'] = qwen_VIE_SC

    if need_run_pq:
        qwen_VIE_PQ = eval_PQ(tar_img_pil)
        item['qwen_VIE_PQ'] = qwen_VIE_PQ

    return need_run_sc or need_run_pq

def update_json_file(json_fp):
    try:
        with open(json_fp) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading json file {json_fp}: {e}")
        return
    
    need_update = False
    for item_idx, item in data.items():
        if item['task_type'] == 'add':
            continue
        elif item['task_type'] == 'replace':
            if 'wangjiazhi/01_data' in item['target_img_path']:
                continue
        elif item['task_type'] == 'remove':
            pass
        else:
            print(f"Unexpected task_type {item['task_type']}")
            continue
        
        need_update = add_scores_to_item(item)

    if need_update:
        new_json_fp = os.path.join(output_dir, os.path.basename(json_fp))  # 保存到目标文件夹
        with open(new_json_fp, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Updated JSON file saved to {new_json_fp}.")

def find_json_files(root_dir):
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files

# 原始文件夹路径
root_dir = '../InsightEdit/data_processing/assets/4_instruction_recaption'
json_files = find_json_files(root_dir)
print(f"Found {len(json_files)} JSON files:")

# 设置目标文件夹路径
output_dir = '../InsightEdit/data_processing/assets/5_quality_evaluation'  # 目标文件夹
os.makedirs(output_dir, exist_ok=True)

def do_task(json_fp):
    update_json_file(json_fp)


do_parallel = True
if not do_parallel:
    for json_fp in tqdm(json_files):
        do_task(json_fp)
else:
    worker_num = 10

    import concurrent

    while True:
        with concurrent.futures.ThreadPoolExecutor(worker_num) as executor:
            list(tqdm(executor.map(do_task, json_files), desc="All jsons", total=len(json_files)))
        print("\nComplete one run!\n")
        time.sleep(10)

