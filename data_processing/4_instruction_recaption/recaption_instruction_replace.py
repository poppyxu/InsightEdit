from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import re
import time


openai_api_key = "" #set the api key
openai_api_base = "" #set the api base

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

system_prompt = '''
            作为一名专业的Prompt工程师，您的任务是为当前的图像编辑任务生成两条不同的带有推理或者理解能力的指令。请遵循以下准则：

            **任务描述**：
            这是一个图像替换的任务，你需要根据给定的信息来生成一条图像编辑的指令文本。您将接收到两张张图像，一张是替换前的图像，一张是替换后的图像，和一条简单的替换内容的指令。您的目标是根据这些信息将原始指令转写成带有推理或理解能力的指令。

            **遵循原则**：
            1.JSON文件中的所有内容必须使用英文。
            2.该任务涉及将当前图像中的“class_origin”替换为“class_target”。请生成一条指令，避免简单的“replace object1 with object2”表述，指令需体现理解和推理能力。
            3.指令应明确且精简，不超过20个单词。
            4.请考虑从以下角度出发：颜色、位置（左右、上下）、大小等。
            5.尽量避免提及原有物体的具体名称，使用抽象的表述，例如“左边的物体”、“旁边的物体”等。         
            6.请你优先采用问句，通过一些图像的内容指代找出被替换的物体，然后再修改。   
            
            **JSON模板**：
            
            ```json
            {
                "instruction1": "生成的带有理解能力的图像编辑指令",   
                "instruction2": "生成的带有推理能力的图像编辑指令",
            }
            ```

            **示例**：
            **输入**：
            json
            {
                "origin_img": origin_img, //原始图片
                "target_img": target_img, //目标图片
                "instruction": instruction //原始指令
            }

            **输出**：
            json
            {
                "instruction1": "What's on the man's head? Please remove it.",   
                "instruction2": "Please identify what the man is wearing on his head and then take the necessary action to remove it.",
            }
            '''

# We use Chinese prompt in data construction, here's a English_version of the prompt
system_prompt_en = '''
            As a professional Prompt Engineer, your task is to generate two different instructions with reasoning or understanding for the current image editing task. Please follow these guidelines:

            **Task Description**:
            This is an image replacement task. You need to generate an image editing instruction based on the given information. You will receive two images, one is the original image, and the other is the target image, along with a simple instruction for the replacement. Your goal is to rewrite the original instruction into one that demonstrates reasoning or understanding.

            **Guidelines**:
            1. All content in the JSON file must be in English.
            2. This task involves replacing the “class_origin” in the current image with “class_target.” Please generate an instruction that avoids a simple “replace object1 with object2” statement, and the instruction should reflect reasoning and understanding.
            3. The instruction should be clear and concise, no more than 20 words.
            4. Consider the following aspects: color, position (left-right, up-down), size, etc.
            5. Try to avoid mentioning the specific name of the original object, and use abstract descriptions like "the object on the left", "the object next to it", etc.
            6. Preferably, frame the instruction as a question to identify the object being replaced through the image content and then make the necessary modification.

            **JSON Template**:

            ```json
            {
                "instruction1": "Generated image editing instruction with understanding",   
                "instruction2": "Generated image editing instruction with reasoning",
            }
            ```

            **Example**:
            **Input**:
            json
            {
                "origin_img": origin_img, // Original image
                "target_img": target_img, // Target image
                "instruction": instruction // Original instruction
            }

            **Output**:
            json
            {
                "instruction1": "What's on the man's head? Please remove it.",   
                "instruction2": "Please identify what the man is wearing on his head and then take the necessary action to remove it.",
            }
            '''

def image_to_base64(image_path, max_size=(256, 256)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.LANCZOS) 
        buffered = BytesIO()
        img.save(buffered, format="JPEG") 
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def parser_simple_json(text):
    text = text.strip().strip("'''json").strip("'''")
    start = text.find("{")
    end = text.rfind("}")
    json_string = text[start:end + 1]
    
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    
    return data

def call_gpt(user_prompt, retries=3):
    for attempt in range(retries):
        chat_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        response = chat_response.choices[0].message.content
        res = parser_simple_json(response)
        print(res)
        
        if res is not None:
            return res
        else:
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(1)
    
    print("All attempts failed.")
    return None

def process_file(json_file, src_dir, dst_dir):
    dst_file_path = os.path.join(dst_dir, json_file)
    if os.path.exists(dst_file_path):
        print(f"File: {json_file} already exists in {dst_dir}, skipping.")
        return

    file_path = os.path.join(src_dir, json_file)

    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            for key, value in data.items():
                origin_img_path = value.get('origin_img_path')
                target_img_path = value.get('target_img_path')

                origin_img_base64 = image_to_base64(origin_img_path)
                target_img_base64 = image_to_base64(target_img_path)

                instruction = value.get('instruction')
                user_prompt = {
                    "origin_img": origin_img_base64,
                    "target_img": target_img_base64,
                    "instruction": instruction
                }
                processed_data = call_gpt(json.dumps(user_prompt))
                
                if processed_data:
                    value["reasoning_instruction1"] = processed_data["instruction1"]
                    value["reasoning_instruction2"] = processed_data["instruction2"]
                    data[key] = value

            with open(dst_file_path, 'w') as dst_file:
                json.dump(data, dst_file, indent=4)
            print("Processed and saved:", json_file)

        except json.JSONDecodeError:
            print("File:", json_file, "is not a valid JSON file.")

def extract_and_process_jsons(src_dir, dst_dir):
    try:
        json_files = [f for f in os.listdir(src_dir) if f.endswith('.json')]
        json_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

        
        print(json_files[0])
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for json_file in json_files:
            process_file(json_file, src_dir, dst_dir)
    except Exception as e:
        print(f"Error in extract_and_process_jsons: {e}")

if __name__ == "__main__":
    source_directory = '../InsightEdit/data_processing/assets/3_editing_pair_construction/replace/json_file'
    destination_directory = "../InsightEdit/data_processing/assets/4_instruction_recaption/replace"

    extract_and_process_jsons(source_directory, destination_directory)
