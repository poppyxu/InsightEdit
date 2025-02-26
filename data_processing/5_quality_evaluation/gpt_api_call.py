"""
official API doc: https://platform.openai.com/docs/guides/vision
should close proxy to access openai API
"""

import base64
import os
import json
import time
import requests
from PIL import Image
import io


def image_to_base64(image_path, max_side=2048):
    with Image.open(image_path) as img:
        if max(img.size) > max_side:
            # Determine the scaling factor to resize the image
            scale_factor = max_side / max(img.size)
            
            # Resize the image
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            print(f"Resizing {image_path} from {img.size} to {new_size} before encoding to base64.")
            # img = img.resize(new_size, Image.ANTIALIAS)  # w, h
            img = img.resize(new_size)  # w, h
        
        # Save the (possibly resized) image to a bytes buffer
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        # Encode the image to base64
        base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_string


class GPT4API(object):
    def __init__(self, model="gpt-4o"):
        for env_name in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
            os.environ.pop(env_name, None)

        self.url = ""
        API_KEY = ""
        self.headers = {
            'Authorization': API_KEY,
            'Content-Type': 'application/json'
        }
        
        self.model = model
        assert self.model in ["gpt-4-turbo", "gpt-4o", "yi-vision"], f"model {self.model} not supported"
        print(f"API using GPT model: {self.model}")

    def invoke(self, prompt, images=None, img_max_side=2048, img_detail='auto', max_retries=2, use_json_mode=False):
        if images is None:
            user_content = prompt
        else:
            """ img_detail: low, high, or auto """
            user_content = [
                {
                    "type": "text", 
                    "text": f"{prompt}"
                },
            ]

            for image in images:
                if image.startswith('data:image'):
                    img_content = image
                else:
                    assert os.path.exists(image), f"image file not exists: {image}"
                    img_ext = os.path.splitext(image)[-1][1:]
                    ext = 'jpeg' if img_ext.lower() in ["jpg", "jpeg"] else 'png'
                    img_content = f"data:image/{ext};base64,{image_to_base64(image, img_max_side)}"  # or web URL of an image

                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{img_content}",
                            # "detail": f"{img_detail}"
                        }
                    }
                )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "max_tokens": 2700,
            "stream": "false"
        }

        if use_json_mode:
            payload['response_format'] = { "type": "json_object" }

        retry_count = 0
        while retry_count < max_retries:
            t0 = time.time()
            try:
                # choose proper timeout to allow the API be in time for return the full reponse
                response = requests.post(self.url, json=payload, headers=self.headers, timeout=60)
            except Exception as e:
                print(f"call attempt[{retry_count}] got post exception {e}\n")
                retry_count += 1
                time.sleep(2)
                continue
                
            if response.status_code == 200:
                response_time = time.time() - t0
                response = response.json()
                text = response['choices'][0]['message']['content']  # from role 'assistant'
                
                return text, response_time

            else:
                # print(f"call attempt[{retry_count}] {img_local_path} got abnormal response status {response.status_code}: {response.text}")
                print(f"call attempt[{retry_count}] got abnormal response status {response.status_code}: {response.text}")
                retry_count += 1
                time.sleep(2)
        
        return None


if __name__ == "__main__":

    gpt4api = GPT4API(model="gpt-4o")
    img_local_paths = [
        "",
        "",
    ]

    prompt = "Describe the images."
    resp = gpt4api.invoke(prompt, img_local_paths, img_max_side=1030, img_detail='auto', use_json_mode=False)
    if resp:
        text, response_time = resp
        print(response_time, text)
        # return text, response_time
    else:
        print("Call GPT return None")
        # return None