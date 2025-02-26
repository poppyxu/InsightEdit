from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import json
from openai import OpenAI
from multiprocessing import Pool, cpu_count
import time

device = "cuda" # the device to load the model onto

openai_api_key = "" #set the api key
openai_api_base = "" #set the api base
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


system_prompt = """
        作为一名专业的 Prompt 工程师，您的任务是将图像描述转化为 JSON 格式的文件，并判断图像中的某个特定对象是否适合被移除。请严格遵循以下准则进行判断：

        ---

        ### **任务描述**：
        您将接收到图像的描述以及需要评估是否适合移除的对象（`class_name` 所指代的物体）。请根据准则判断该物体是否具体且适合移除。

        ---

        ### **判断原则**：
        1. **可移除对象的标准**：
           - **具体的物体**（如人物、植物、动物、工具、家具、交通工具、电子产品、建筑物等）倾向于可以移除。
           - 这些物体通常是可触摸、可单独识别的物件。

        2. **不适合移除的对象**：
           - **自然景观**（如山川、河流、天空、海洋、森林、沙漠等）不应被移除。
           - **人体的一部分**（如胳膊、腿、服饰或配件，如果它是穿戴在人体上的一部分）。
           - **抽象概念或动名词**（如“风景”、“背景”、“视野” 、unknown等）。
           - **整体环境或场景**，如风景或场景背景构成图像核心的情感或主题部分。
           - **描述视角的词**，如view、perspective、wide-angle view。

        3. **返回结果**：
           - 如果对象符合移除条件，请返回 `flag` 为 1。
           - 如果对象不适合移除，请返回 `flag` 为 0。
           - 仅仅返回json

        ---

        ### **JSON模板**：
        ```json
        {
            "flag": 1  // 或 0
        }

        **示例一**：
        **输入**：
        json
        {
            "img_caption": "A picturesque wide-angle photograph of a man in his 30s, sitting on a large rock and playing an acoustic guitar, set against a stunning natural landscape with lush greenery, a river, and mountains in the background. The image highlights the man's tattoos, his casual attire, and captures the serene atmosphere of his surroundings.",
            "class_name": "red bandana",
            "detailed_caption": "A red bandana wrapped around the man's neck as an accessory, often used for fashion or as a personal style element."
        }


        **输出**：
        json
        {
            "flag": 0
        }

        **示例二**：
        **输入**：
        json
        {
            "img_caption": "A photo showing a rustic wooden dining table surrounded by chairs in a cozy living room.",
            "class_name": "wooden chair",
            "detailed_caption": "A wooden chair placed next to the dining table, with a simple but elegant design."
        }


        **输出**：
        json
        {
            "flag": 1
        }


        """

system_prompt_en = """
        As a professional Prompt Engineer, your task is to convert an image description into a JSON formatted file and determine whether a specific object in the image, as indicated by `class_name`, should be removed. Please strictly follow the guidelines below when making your judgment:

        ---

        ### **Task Description**:
        You will receive a description of the image and an object (`class_name`) to assess whether it is suitable for removal. Please use the guidelines to determine if the object is specific and suitable for removal.

        ---

        ### **Judgment Principles**:
        1. **Criteria for removable objects**:
           - **Concrete objects** (such as people, plants, animals, tools, furniture, vehicles, electronics, buildings, etc.) are more likely to be removed.
           - These objects are typically tangible and identifiable.

        2. **Objects that should not be removed**:
           - **Natural landscapes** (e.g., mountains, rivers, sky, oceans, forests, deserts, etc.) should not be removed.
           - **Body parts** (e.g., arms, legs, clothing or accessories worn on the body).
           - **Abstract concepts or gerunds** (e.g., "scenery", "background", "view", "unknown", etc.).
           - **Overall environment or scene**: The scenery or scene background that constitutes the core emotional or thematic part of the image.
           - **Words describing perspective**, such as "view", "perspective", "wide-angle view".

        3. **Return Result**:
           - If the object meets the removal criteria, return `flag` as 1.
           - If the object is not suitable for removal, return `flag` as 0.
           - Only return the JSON.

        ---

        ### **JSON Template**:
        ```json
        {
            "flag": 1  // or 0
        }

        **Example 1**:
        **Input**:
        json
        {
            "img_caption": "A picturesque wide-angle photograph of a man in his 30s, sitting on a large rock and playing an acoustic guitar, set against a stunning natural landscape with lush greenery, a river, and mountains in the background. The image highlights the man's tattoos, his casual attire, and captures the serene atmosphere of his surroundings.",
            "class_name": "red bandana",
            "detailed_caption": "A red bandana wrapped around the man's neck as an accessory, often used for fashion or as a personal style element."
        }


        **Output**:
        json
        {
            "flag": 0
        }

        **Example 2**:
        **Input**:
        json
        {
            "img_caption": "A photo showing a rustic wooden dining table surrounded by chairs in a cozy living room.",
            "class_name": "wooden chair",
            "detailed_caption": "A wooden chair placed next to the dining table, with a simple but elegant design."
        }


        **Output**:
        json
        {
            "flag": 1
        }

        """
def parser_simple_json(text):
    # Remove the leading '''json and trailing '''
    text = text.strip().strip("'''json").strip("'''")
    
    # Find the position of the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    
    # Extract the JSON string
    json_string = text[start:end+1]
    # print("json string")
    # print(json_string)
    
    # Parse the JSON string
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    
    return data


def call_qwen(user_prompt, retries=3):
    for attempt in range(retries):
        try:
            chat_response = client.chat.completions.create(
                model="Qwen2-7B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            response = chat_response.choices[0].message.content
            print("response:", response)
            res = parser_simple_json(response)
            
            if res is not None:
                return res
            else:
                print(f"Attempt {attempt + 1} failed due to JSON decoding error, retrying...")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}, retrying...")
        
        time.sleep(1)  # Wait for 1 second before retrying

    print("All attempts failed.")
    return None

def process_file(args):
    json_file, src_dir, dst_dir = args
    dst_file_path = os.path.join(dst_dir, json_file)

    # Check if the destination file already exists
    if os.path.exists(dst_file_path):
        print(f"File: {json_file} already exists in {dst_dir}, skipping.")
        return

    file_path = os.path.join(src_dir, json_file)

    try:
        
        with open(file_path, 'r') as src_file:
            data = json.load(src_file)
            annotations = data.get('annotations', [])
            img_caption = data.get('img_caption')

            # Create a new list to store valid annotations
            filtered_annotations = []
            if len(annotations) == 0:
                return

            # Iterate through the original annotations list
            for anno in annotations:
                user_prompt = {
                    "img_caption": img_caption,
                    "class_name": anno["class_name"],
                    "detailed_caption": anno["detailed_caption"]
                }
                #print("user prompt:", user_prompt)
                # Call Qwen API and get the response
                processed_data = call_qwen(json.dumps(user_prompt))
    
                # Check if flag is not equal to 0
                if processed_data["flag"] != 0:
                    filtered_annotations.append(anno)

            if len(filtered_annotations) == 0:
                return 

            # Update the original data
            data['annotations'] = filtered_annotations
            
            # Save the filtered data
            with open(dst_file_path, 'w') as dst_file:
                json.dump(data, dst_file, indent=4)

            print("Processed and saved:", json_file)

    except Exception as e:
        print(f"Error processing file {json_file}: {e}")


def extract_and_process_jsons(src_dir, dst_dir):
    try:
        # Get all JSON files from the source directory
        json_files = [f for f in os.listdir(src_dir) if f.endswith('.json')]

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        # Prepare arguments for multiprocessing
        args = [(json_file, src_dir, dst_dir) for json_file in json_files]

        count = cpu_count() - 10
        # Use multiprocessing Pool to process files concurrently
        with Pool(count) as pool:
            pool.map(process_file, args)
    except Exception as e:
        print(f"Error in extract_and_process_jsons: {e}")


if __name__ == "__main__":
    source_directory = "../data_processing/assets/1_caption_object_extraction/1_2_object_extraction"
    destination_directory = ""

    #user_prompt = "A digital artwork featuring an abstract design with swirling, colorful lines and a central Windows logo. The image is set against a dark background, with vibrant hues of orange, purple, green, and blue creating a dynamic and modern visual effect. The Windows logo, consisting of four colored squares, is prominently displayed in the center, surrounded by luminous, curved lines that give the impression of motion and energy. This contemporary piece exemplifies digital art and abstract design, capturing the essence of technology and innovation."

    extract_and_process_jsons(source_directory, destination_directory)