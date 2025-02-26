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


system_prompt = '''
            作为一名专业的Prompt工程师，您的任务是将图像描述转化为JSON格式的文件。请遵循以下准则：

            **任务描述**：
            您将接收到一张图像的描述以及图像中的一个特定物体。您的目标是判断该物体是否适合被替代，并在适合的情况下提供一个新的物品描述以替代原有的物体。

            **遵循原则**：
            1. JSON文件中的所有内容必须使用英文。
            2. 请首先判断“class_name”所指的物体是否适合被替代。适合替代的物体应具体明确，避免过大或抽象的对象（例如，风景、视野、背景等不应被替代）。如果不适合，请返回 flag 为 0，其他字段可设置为 null。
            3. 如果适合替换，请返回 `flag` 为 1，并提供一个新的描述。
            4. 在保持合理性的基础上，鼓励您发挥创意和想象力。
            5. JSON文件中的“replace_class_name”字段应包含一个简短的词汇，而“replace_detailed_caption”字段则应详细描述该词汇在整张图像中的表现。
            6. flag为int类型


            **不适合替代的对象**：
           - **自然景观**（如山川、河流、天空、海洋、森林、沙漠等）不应被移除。
           - **人体的一部分**（如胳膊、腿、服饰或配件，如果它是穿戴在人体上的一部分）。
           - **抽象概念或动名词**（如“风景”、“背景”、“视野” 、unknown等）。
           - **整体环境或场景**，如风景或场景背景构成图像核心的情感或主题部分。
           - **描述视角的词**，如view、perspective、wide-angle view。


            **JSON模板**：
            
            ```json
            {
                "flag": 1,  // 或 0
                "replace_class_name": "替代物品的名称",
                "replace_detailed_caption": "替代物品在图像中的详细描述"
            }
            ```

            **示例**：
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
                "flag": 1, 
                "replace_class_name": "light scarf",
                "replace_detailed_caption": "A light scarf draped around the man's neck, adding a touch of elegance to his casual attire. It complements the serene atmosphere of the natural landscape, blending harmoniously with the lush greenery, river, and mountains in the background."
            }
            '''

system_prompt_en = '''
            As a professional Prompt Engineer, your task is to convert an image description into a JSON formatted file. Please follow these guidelines:

            **Task Description**:
            You will receive a description of an image and a specific object in the image. Your goal is to determine whether the object is suitable for replacement, and if so, provide a new description to replace the original object.

            **Guidelines**:
            1. All content in the JSON file must be in English.
            2. First, assess whether the object indicated by "class_name" is suitable for replacement. Replaceable objects should be specific and clear, avoiding large or abstract objects (e.g., landscapes, views, background, etc., should not be replaced). If not suitable, return flag as 0, and other fields can be set to null.
            3. If suitable for replacement, return `flag` as 1 and provide a new description.
            4. Be encouraged to use creativity and imagination while maintaining reasonable context.
            5. The "replace_class_name" field in the JSON file should contain a brief word, and the "replace_detailed_caption" field should describe the appearance of the replaced object in the entire image.
            6. The flag is of int type.

            **Objects that should not be replaced**:
           - **Natural landscapes** (e.g., mountains, rivers, sky, oceans, forests, deserts, etc.) should not be removed.
           - **Body parts** (e.g., arms, legs, clothing or accessories worn on the body).
           - **Abstract concepts or gerunds** (e.g., "scenery", "background", "view", "unknown", etc.).
           - **Overall environment or scene**, such as landscapes or scene backgrounds that constitute the core emotional or thematic part of the image.
           - **Words describing perspective**, such as view, perspective, wide-angle view.

            **JSON Template**:

            ```json
            {
                "flag": 1,  // or 0
                "replace_class_name": "The name of the replaced item",
                "replace_detailed_caption": "Detailed description of the replaced item in the image"
            }
            ```

            **Example**:
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
                "flag": 1, 
                "replace_class_name": "light scarf",
                "replace_detailed_caption": "A light scarf draped around the man's neck, adding a touch of elegance to his casual attire. It complements the serene atmosphere of the natural landscape, blending harmoniously with the lush greenery, river, and mountains in the background."
            }
            '''
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
    # Check if the target file already exists
    dst_file_path = os.path.join(dst_dir, json_file)
    if os.path.exists(dst_file_path):
        print(f"File: {json_file} already exists in {dst_dir}, skipping.")
        return  # Skip the file if it already exists

    file_path = os.path.join(src_dir, json_file)

    # Read the JSON file
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            img_caption = data.get('img_caption')
            annotations = data.get('annotations')
            if len(annotations) == 0:
                return 

            new_annotations = []

            for anno in annotations:
                user_prompt = {
                    "img_caption": img_caption,
                    "class_name": anno["class_name"],
                    "detailed_caption": anno["detailed_caption"]
                }
                processed_data = call_qwen(json.dumps(user_prompt))
    
                if processed_data["flag"] != 0:
                    new_annotations.append({
                        **anno, 
                        "replace_class_name": processed_data["replace_class_name"],
                        "replace_detailed_caption": processed_data["replace_detailed_caption"]
                    })
            if len(new_annotations) == 0:
                return
            # Update the data and save it
            data['annotations'] = new_annotations
            with open(dst_file_path, 'w') as dst_file:
                json.dump(data, dst_file, indent=4)
            print("Processed and saved:", json_file)

        except json.JSONDecodeError:
            print("File:", json_file, "is not a valid JSON file.")


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

    # user_prompt = "A digital artwork featuring an abstract design with swirling, colorful lines and a central Windows logo. The image is set against a dark background, with vibrant hues of orange, purple, green, and blue creating a dynamic and modern visual effect. The Windows logo, consisting of four colored squares, is prominently displayed in the center, surrounded by luminous, curved lines that give the impression of motion and energy. This contemporary piece exemplifies digital art and abstract design, capturing the essence of technology and innovation."

    extract_and_process_jsons(source_directory, destination_directory)