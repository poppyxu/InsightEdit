from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import json
from openai import OpenAI
from multiprocessing import Pool, cpu_count

device = "cuda" # the device to load the model onto

openai_api_key = "" #set the api key
openai_api_base = "" #set the api base

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


system_prompt = """
你是一个prompt工程师，你现在要把一段对图像的描述转换成json文件，请你遵循以下的步骤：

1. 分析文本内容：
仔细阅读文本，识别出主要的对象、场景和细节。
请注意这里的对象需要是实体，具体的事物，而不是抽象的事物。
2. 定义对象类别：
为文本中描述的每个独特对象确定一个合适的类别。
3. 创建对象条目：
对于每个对象，创建一个JSON对象，包括以下字段,生成的内容为英文：
- category: 对象的类别，用于分类和识别。
- obj_simple_caption: 一个简洁的描述，概括对象的主要特征，不要有括号，补充的信息放在正文中。
- obj_detailed_caption: 一个详细的描述，提供对象的具体信息和上下文，生成的内容在50个words以内
4. 撰写图像总体描述：
在JSON的顶层，编写一个字段来描述整个图像或场景的总体印象。
5. 构建JSON结构：
将所有对象条目和图像描述整合成一个完整的JSON对象。


以下是一个根据这些步骤设计的JSON模板，以及一个实际的转换示例：

### JSON模板

```json
{
  "object_list": [
    {
      "category": "类别",
      "obj_simple_caption": "简短标题",
      "obj_detailed_caption": "详细描述"
    },
    // 根据需要可以继续添加更多的对象
  ],
  "img_caption": "整个图像或场景的描述"
}
示例
原有的描述
A photograph capturing two young wolf pups resting on a fallen log in a natural setting. The pups, one with light brown fur and the other with dark black fur, are lying close together with their eyes closed, showcasing their soft, fluffy fur and small paws. The background features green grass and blurred foliage, indicating a wide-angle view. The image is realistic and detailed, highlighting the natural beauty and innocence of the young wolves in their natural habitat.

转换后的JSON文件:
{
  "object_list": [
    {
      "category": "Animal",
      "obj_simple_caption": "Young Wolf Pup (Light Brown)",
      "obj_detailed_caption": "A young wolf pup with light brown fur, resting on a fallen log, eyes closed, showcasing its soft, fluffy fur and small paws."
    },
    {
      "category": "Animal",
      "obj_simple_caption": "Young Wolf Pup (Dark Black)",
      "obj_detailed_caption": "A young wolf pup with dark black fur, resting on a fallen log, eyes closed, showcasing its soft, fluffy fur and small paws."
    }
  ],
  "img_caption": "A photograph capturing two young wolf pups resting on a fallen log in a natural setting. The pups, one with light brown fur and the other with dark black fur, are lying close together with their eyes closed, showcasing their soft, fluffy fur and small paws. The background features green grass and blurred foliage, indicating a wide-angle view. The image is realistic and detailed, highlighting the natural beauty and innocence of the young wolves in their natural habitat."
}

"""

system_prompt_en = """
You are a prompt engineer, and you are tasked with converting a description of an image into a JSON file. Please follow these steps:

1. Analyze the text:
Carefully read the text and identify the main objects, scenes, and details.
Note that the objects should be tangible, specific things, not abstract concepts.

2. Define object categories:
Determine a suitable category for each unique object described in the text.

3. Create object entries:
For each object, create a JSON entry with the following fields. The content should be in English:
- category: The object's category for classification and identification.
- obj_simple_caption: A brief description summarizing the main features of the object. Avoid parentheses; supplementary information should be included in the body.
- obj_detailed_caption: A detailed description providing specific information and context for the object, keeping the description within 50 words.

4. Write the overall image description:
At the top level of the JSON, write a field that describes the overall impression of the entire image or scene.

5. Build the JSON structure:
Combine all object entries and the overall image description into a complete JSON object.

Here is a JSON template designed based on these steps, along with an actual conversion example:

### JSON Template

```json
{
  "object_list": [
    {
      "category": "Category",
      "obj_simple_caption": "Brief Caption",
      "obj_detailed_caption": "Detailed Description"
    }
    // More objects can be added as needed
  ],
  "img_caption": "Description of the entire image or scene"
}
"""
def parser_simple_json(text):
    text = text.strip().strip("'''json").strip("'''")
    start = text.find("{")
    end = text.rfind("}")

    json_string = text[start:end+1]
    
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    
    return data

def call_qwen(user_prompt):
    chat_response = client.chat.completions.create(
        model="Qwen2-7B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    response = chat_response.choices[0].message.content
    res = parser_simple_json(response)
    print(res)
    return res

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

            # Extract the 'yivl4' field
            yivl4_value = data.get('caption', None)
                

            if yivl4_value is not None:
                # Call call_qwen function to process yivl4_value
                processed_data = call_qwen(yivl4_value)
                if processed_data is None:
                    return

                # Save the processed data to the target directory
                with open(dst_file_path, 'w') as dst_file:
                    json.dump(processed_data, dst_file, indent=4)
                print("Processed and saved:", json_file)
            else:
                print("File:", json_file, "does not contain 'yivl4' field.")

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

        # Use multiprocessing Pool to process files concurrently
        with Pool(cpu_count()) as pool:
            pool.map(process_file, args)
    except Exception as e:
        print(f"Error in extract_and_process_jsons: {e}")


if __name__ == "__main__":

    source_directory = "../data_processing/assets/1_caption_object_extraction/1_1_recaption" 
    destination_directory = "../data_processing/assets/1_caption_object_extraction/1_2_object_extraction"

    #user_prompt = "A digital artwork featuring an abstract design with swirling, colorful lines and a central Windows logo. The image is set against a dark background, with vibrant hues of orange, purple, green, and blue creating a dynamic and modern visual effect. The Windows logo, consisting of four colored squares, is prominently displayed in the center, surrounded by luminous, curved lines that give the impression of motion and energy. This contemporary piece exemplifies digital art and abstract design, capturing the essence of technology and innovation."

    extract_and_process_jsons(source_directory, destination_directory)