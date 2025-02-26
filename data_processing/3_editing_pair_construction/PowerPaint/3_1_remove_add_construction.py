import argparse
import os
import random
from pycocotools import mask as coco_mask
import json
import cv2
import gradio as gr
import numpy as np
import torch
import time
from controlnet_aux import HEDdetector, OpenposeDetector
from PIL import Image, ImageFilter
from safetensors.torch import load_model
from transformers import CLIPTextModel, DPTFeatureExtractor, DPTForDepthEstimation

from diffusers import UniPCMultistepScheduler
from diffusers.pipelines.controlnet.pipeline_controlnet import ControlNetModel
from scipy.ndimage import gaussian_filter

from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.pipelines.pipeline_PowerPaint_ControlNet import (
    StableDiffusionControlNetInpaintPipeline as controlnetPipeline,
)
from powerpaint.utils.utils import TokenizerWrapper, add_tokens
import re
from scipy.ndimage import binary_dilation, binary_erosion
from openai import OpenAI
#from multiprocessing import Process, Queue, current_process
from multiprocessing import Pool, cpu_count

import multiprocessing as mp

# 设置进程启动方法为 'spawn'
mp.set_start_method('spawn', force=True)
torch.set_grad_enabled(False)
openai_api_key = "" #set the api key
openai_api_base = "" #set the api base



client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_task(prompt, negative_prompt, control_type, version):
    pos_prefix = neg_prefix = ""
    if control_type == "object-removal" or control_type == "image-outpainting":
        if version == "ppt-v1":
            pos_prefix = "empty scene blur " + prompt
            neg_prefix = negative_prompt
        promptA = pos_prefix + " P_ctxt"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_obj"
    elif control_type == "shape-guided":
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_shape"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + "P_shape"
        negative_promptB = neg_prefix + "P_ctxt"
    else:
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = negative_prompt + ", worst quality, low quality, normal quality, bad quality, blurry "
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_obj"
        negative_promptA = neg_prefix + "P_obj"
        negative_promptB = neg_prefix + "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


def select_tab_text_guided():
    return "text-guided"


def select_tab_object_removal():
    return "object-removal"


def select_tab_image_outpainting():
    return "image-outpainting"


def select_tab_shape_guided():
    return "shape-guided"


class PowerPaintController:
    def __init__(self, weight_dtype, checkpoint_dir, local_files_only, version) -> None:
        self.version = version
        self.checkpoint_dir = checkpoint_dir
        self.local_files_only = local_files_only

        # initialize powerpaint pipeline
        if version == "ppt-v1":
            self.pipe = Pipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=weight_dtype, local_files_only=local_files_only
            )
            self.pipe.tokenizer = TokenizerWrapper(
                from_pretrained="runwayml/stable-diffusion-v1-5",
                subfolder="tokenizer",
                revision=None,
                local_files_only=local_files_only,
            )

            # add learned task tokens into the tokenizer
            add_tokens(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder,
                placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                initialize_tokens=["a", "a", "a"],
                num_vectors_per_token=10,
            )

            # loading pre-trained weights
            load_model(self.pipe.unet, os.path.join(checkpoint_dir, "unet/unet.safetensors"))
            load_model(self.pipe.text_encoder, os.path.join(checkpoint_dir, "text_encoder/text_encoder.safetensors"))
            self.pipe = self.pipe.to("cuda")

            # initialize controlnet-related models
            self.depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
            self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
            self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

            base_control = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, local_files_only=local_files_only
            )
            self.control_pipe = controlnetPipeline(
                self.pipe.vae,
                self.pipe.text_encoder,
                self.pipe.tokenizer,
                self.pipe.unet,
                base_control,
                self.pipe.scheduler,
                None,
                None,
                False,
            )
            self.control_pipe = self.control_pipe.to("cuda")

            self.current_control = "canny"
            # controlnet_conditioning_scale = 0.8
        else:
            # brushnet-based version
            unet = UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="unet",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            text_encoder_brushnet = CLIPTextModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="text_encoder",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            brushnet = BrushNetModel.from_unet(unet)
            base_model_path = os.path.join(checkpoint_dir, "realisticVisionV60B1_v51VAE")
            self.pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
                base_model_path,
                brushnet=brushnet,
                text_encoder_brushnet=text_encoder_brushnet,
                torch_dtype=weight_dtype,
                low_cpu_mem_usage=False,
                safety_checker=None,
            )
            self.pipe.unet = UNet2DConditionModel.from_pretrained(
                base_model_path,
                subfolder="unet",
                revision=None,
                torch_dtype=weight_dtype,
                local_files_only=local_files_only,
            )
            self.pipe.tokenizer = TokenizerWrapper(
                from_pretrained=base_model_path,
                subfolder="tokenizer",
                revision=None,
                torch_type=weight_dtype,
                local_files_only=local_files_only,
            )

            # add learned task tokens into the tokenizer
            add_tokens(
                tokenizer=self.pipe.tokenizer,
                text_encoder=self.pipe.text_encoder_brushnet,
                placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
                initialize_tokens=["a", "a", "a"],
                num_vectors_per_token=10,
            )
            load_model(
                self.pipe.brushnet,
                os.path.join(checkpoint_dir, "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
            )

            self.pipe.text_encoder_brushnet.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
            )

            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

            # self.pipe.enable_model_cpu_offload()
            self.pipe = self.pipe.to("cuda")

    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    def load_controlnet(self, control_type):
        if self.current_control != control_type:
            if control_type == "canny" or control_type is None:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            elif control_type == "pose":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-openpose",
                    torch_dtype=weight_dtype,
                    local_files_only=self.local_files_only,
                )
            elif control_type == "depth":
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-depth", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            else:
                self.control_pipe.controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-hed", torch_dtype=weight_dtype, local_files_only=self.local_files_only
                )
            self.control_pipe = self.control_pipe.to("cuda")
            self.current_control = control_type

    def predict(
        self,
        input_image,
        prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        task,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
    ):
        size1, size2 = input_image["image"].convert("RGB").size

        if task != "image-outpainting":
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        else:
            if size1 < size2:
                input_image["image"] = input_image["image"].convert("RGB").resize((512, int(size2 / size1 * 512)))
            else:
                input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 512), 512))

        if vertical_expansion_ratio is not None and horizontal_expansion_ratio is not None:
            o_W, o_H = input_image["image"].convert("RGB").size
            c_W = int(horizontal_expansion_ratio * o_W)
            c_H = int(vertical_expansion_ratio * o_H)

            expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * 127
            original_img = np.array(input_image["image"])
            expand_img[
                int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                :,
            ] = original_img

            blurry_gap = 10

            expand_mask = np.ones((c_H, c_W, 3), dtype=np.uint8) * 255
            if vertical_expansion_ratio == 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) : int((c_H - o_H) / 2.0) + o_H,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio != 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) + blurry_gap : int((c_W - o_W) / 2.0) + o_W - blurry_gap,
                    :,
                ] = 0
            elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio == 1:
                expand_mask[
                    int((c_H - o_H) / 2.0) + blurry_gap : int((c_H - o_H) / 2.0) + o_H - blurry_gap,
                    int((c_W - o_W) / 2.0) : int((c_W - o_W) / 2.0) + o_W,
                    :,
                ] = 0

            input_image["image"] = Image.fromarray(expand_img)
            input_image["mask"] = Image.fromarray(expand_mask)

        if self.version != "ppt-v1":
            if task == "image-outpainting":
                prompt = prompt + " empty scene"
            if task == "object-removal":
                prompt = prompt + " empty scene blur"
        promptA, promptB, negative_promptA, negative_promptB = add_task(prompt, negative_prompt, task, self.version)
        print(promptA, promptB, negative_promptA, negative_promptB)

        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))
        set_seed(seed)

        if self.version == "ppt-v1":
            # for sd-inpainting based method
            result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                image=input_image["image"].convert("RGB"),
                mask=input_image["mask"].convert("RGB"),
                width=H,
                height=W,
                guidance_scale=scale,
                num_inference_steps=ddim_steps,
            ).images[0]
        else:
            # for brushnet-based method
            np_inpimg = np.array(input_image["image"])
            np_inmask = np.array(input_image["mask"]) / 255.0
            np_inpimg = np_inpimg * (1 - np_inmask)
            input_image["image"] = Image.fromarray(np_inpimg.astype(np.uint8)).convert("RGB")
            result = self.pipe(
                promptA=promptA,
                promptB=promptB,
                promptU=prompt,
                tradoff=fitting_degree,
                tradoff_nag=fitting_degree,
                image=input_image["image"].convert("RGB"),
                mask=input_image["mask"].convert("RGB"),
                num_inference_steps=ddim_steps,
                generator=torch.Generator("cuda").manual_seed(seed),
                brushnet_conditioning_scale=1.0,
                negative_promptA=negative_promptA,
                negative_promptB=negative_promptB,
                negative_promptU=negative_prompt,
                guidance_scale=scale,
                width=H,
                height=W,
            ).images[0]

        mask_np = np.array(input_image["mask"].convert("RGB"))
        red = np.array(result).astype("float") * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        result_m = np.array(result)
        result_m = Image.fromarray(
            (
                result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
                + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )
        m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
        m_img = np.asarray(m_img) / 255.0
        img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
        ours_np = np.asarray(result) / 255.0
        ours_np = ours_np * m_img + (1 - m_img) * img_np
        dict_res = [input_image["mask"].convert("RGB"), result_m]

        # result_paste = Image.fromarray(np.uint8(ours_np * 255))
        # dict_out = [input_image["image"].convert("RGB"), result_paste]
        dict_out = [result]
        return dict_out, dict_res

    def predict_controlnet(
        self,
        input_image,
        input_control_image,
        control_type,
        prompt,
        ddim_steps,
        scale,
        seed,
        negative_prompt,
        controlnet_conditioning_scale,
    ):
        promptA = prompt + " P_obj"
        promptB = prompt + " P_obj"
        negative_promptA = negative_prompt
        negative_promptB = negative_prompt
        size1, size2 = input_image["image"].convert("RGB").size

        if size1 < size2:
            input_image["image"] = input_image["image"].convert("RGB").resize((640, int(size2 / size1 * 640)))
        else:
            input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 640), 640))
        img = np.array(input_image["image"].convert("RGB"))
        W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
        H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
        input_image["image"] = input_image["image"].resize((H, W))
        input_image["mask"] = input_image["mask"].resize((H, W))

        if control_type != self.current_control:
            self.load_controlnet(control_type)
        controlnet_image = input_control_image
        if control_type == "canny":
            controlnet_image = controlnet_image.resize((H, W))
            controlnet_image = np.array(controlnet_image)
            controlnet_image = cv2.Canny(controlnet_image, 100, 200)
            controlnet_image = controlnet_image[:, :, None]
            controlnet_image = np.concatenate([controlnet_image, controlnet_image, controlnet_image], axis=2)
            controlnet_image = Image.fromarray(controlnet_image)
        elif control_type == "pose":
            controlnet_image = self.openpose(controlnet_image)
        elif control_type == "depth":
            controlnet_image = controlnet_image.resize((H, W))
            controlnet_image = self.get_depth_map(controlnet_image)
        else:
            controlnet_image = self.hed(controlnet_image)

        mask_np = np.array(input_image["mask"].convert("RGB"))
        controlnet_image = controlnet_image.resize((H, W))
        set_seed(seed)
        result = self.control_pipe(
            promptA=promptB,
            promptB=promptA,
            tradoff=1.0,
            tradoff_nag=1.0,
            negative_promptA=negative_promptA,
            negative_promptB=negative_promptB,
            image=input_image["image"].convert("RGB"),
            mask=input_image["mask"].convert("RGB"),
            control_image=controlnet_image,
            width=H,
            height=W,
            guidance_scale=scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=ddim_steps,
        ).images[0]
        red = np.array(result).astype("float") * 1
        red[:, :, 0] = 180.0
        red[:, :, 2] = 0
        red[:, :, 1] = 0
        result_m = np.array(result)
        result_m = Image.fromarray(
            (
                result_m.astype("float") * (1 - mask_np.astype("float") / 512.0)
                + mask_np.astype("float") / 512.0 * red
            ).astype("uint8")
        )

        mask_np = np.array(input_image["mask"].convert("RGB"))
        m_img = input_image["mask"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=4))
        m_img = np.asarray(m_img) / 255.0
        img_np = np.asarray(input_image["image"].convert("RGB")) / 255.0
        ours_np = np.asarray(result) / 255.0
        ours_np = ours_np * m_img + (1 - m_img) * img_np
        result_paste = Image.fromarray(np.uint8(ours_np * 255))
        return [input_image["image"].convert("RGB"), result_paste], [controlnet_image, result_m]

    def infer(
        self,
        input_image,
        text_guided_prompt,
        text_guided_negative_prompt,
        shape_guided_prompt,
        shape_guided_negative_prompt,
        fitting_degree,
        ddim_steps,
        scale,
        seed,
        task,
        vertical_expansion_ratio,
        horizontal_expansion_ratio,
        outpaint_prompt,
        outpaint_negative_prompt,
        removal_prompt,
        removal_negative_prompt,
        enable_control=False,
        input_control_image=None,
        control_type="canny",
        controlnet_conditioning_scale=None,
    ):
        
        if task == "text-guided":
            prompt = text_guided_prompt
            negative_prompt = text_guided_negative_prompt
        elif task == "shape-guided":
            prompt = shape_guided_prompt
            negative_prompt = shape_guided_negative_prompt
        elif task == "object-removal":
            prompt = removal_prompt
            negative_prompt = removal_negative_prompt
        elif task == "image-outpainting":
            prompt = outpaint_prompt
            negative_prompt = outpaint_negative_prompt
            return self.predict(
                input_image,
                prompt,
                fitting_degree,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                task,
                vertical_expansion_ratio,
                horizontal_expansion_ratio,
            )
        else:
            task = "text-guided"
            prompt = text_guided_prompt
            negative_prompt = text_guided_negative_prompt

        # currently, we only support controlnet in PowerPaint-v1
        if self.version == "ppt-v1" and enable_control and task == "text-guided":
            return self.predict_controlnet(
                input_image,
                input_control_image,
                control_type,
                prompt,
                ddim_steps,
                scale,
                seed,
                negative_prompt,
                controlnet_conditioning_scale,
            )
        else:
            return self.predict(
                input_image, prompt, fitting_degree, ddim_steps, scale, seed, negative_prompt, task, None, None
            )

    
def preprocess_class_name(class_name):
    if class_name is None:  # 检查class_name是否为None
        return 'unknown'  # 返回一个默认值
    return class_name.replace(' ', '_').replace('/', '_').lower()


def parser_simple_json(text):
    text = text.strip().strip("'''json").strip("'''")

    start = text.find("{")
    end = text.rfind("}")

    json_string = text[start:end+1]
    # print("json string")
    # print(json_string)

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    
    return data

def call_gpt(global_caption, class_name, detailed_caption):
    try:
        user_prompt = {
            "img_caption": global_caption,
            "class_name": class_name,
            "detailed_caption": detailed_caption
        }
        user_prompt = json.dumps(user_prompt)

        # 让gpt判断是否消除
        system_prompt = '''
            作为一名专业的Prompt工程师，您的任务是将图像描述转化为JSON格式的文件。请遵循以下准则：

            **任务描述**：
            您将接收到一张图像的描述以及图像中的一个特定物体。您的目标是判断该物体是否适合被移除。

            **遵循原则**：
            1. JSON文件中的所有内容必须使用英文。
            2. 请首先判断“class_name”所指的物体是否适合在图像中被移除。
            4. 判断的标准是，适合移除的物体应具体明确，是图像描述以及图像中的一个特定的物体。不应该是动名词，避免过大或抽象的对象（例如，天空，风景、视野、背景等不应被移除，人体的一部分比如图像上一个人穿着卫衣，那么这个卫衣作为人体的一部分不应该被移除）。
            3. 如果不适合，请返回 flag 为 0。如果适合移除，请返回 `flag` 为 1。
            4. flag为int类型

            **JSON模板**：
            
            ```json
            {
                "flag": 1,  // 或 0
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
                "flag": 0
            }
            '''

        chat_response = client.chat.completions.create(
            model="Qwen2-7B-Instruct",
            #model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        response = chat_response.choices[0].message.content
        res = parser_simple_json(response)
        print(res)
        return res

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


#close operation of mask
def segmentation_to_rgb_dilated_erosion_mask(segmentation):
    """
    Converts RLE segmentation to a black & white RGB mask.

    Parameters:
        segmentation: RLE encoded segmentation.

    Returns:
        PIL Image: RGB mask with the object in white and background in black.
    """
    # Decode the RLE mask
    mask = coco_mask.decode(segmentation)

    y_indices, x_indices = np.nonzero(mask)
    bbox = []

    # calculate the bbox
    if len(y_indices) > 0 and len(x_indices) > 0:
        x_min = int(np.min(x_indices))  
        x_max = int(np.max(x_indices))  
        y_min = int(np.min(y_indices)) 
        y_max = int(np.max(y_indices))  
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]  # [x, y, width, height]
    else:
        bbox = [0, 0, 0, 0]  
    print("bbox:", bbox)

    t1 = time.time()

    # Perform dilation on the mask
    dilation_structure = np.ones((5, 5), dtype=bool)  # Define the structuring element for dilation
    #dilated_mask = binary_dilation(mask, structure=dilation_structure).astype(np.uint8)
    #dilated_mask = cv2.dilate(mask, dilation_structure, iterations=15)
    dilated_mask = binary_dilation(mask, structure=dilation_structure, iterations=40).astype(np.uint8)

    
    # Apply Gaussian blur to smooth the dilated mask
    blurred_mask = gaussian_filter(dilated_mask.astype(float), sigma=2)  # Adjust sigma for more/less smoothing
    blurred_mask = (blurred_mask > 0.5).astype(np.uint8)  # Convert back to binary mask

    t2 = time.time()
    # Perform erosion on the dilated mask
    erosion_structure = np.ones((5, 5), dtype=bool)  # Define the structuring element for erosion
    eroded_mask = binary_erosion(dilated_mask, structure=erosion_structure, iterations=15).astype(np.uint8)
    t3 = time.time()

    print("dilation time:", t2 - t1)
    print("erosion time:", t3 - t2)
   

    # Create an RGB array
    rgb_array = np.zeros((eroded_mask.shape[0], eroded_mask.shape[1], 3), dtype=np.uint8)

    # Set colors based on eroded mask values
    rgb_array[eroded_mask == 1] = [255, 255, 255]  # White for the object
    rgb_array[eroded_mask == 0] = [0, 0, 0]        # Black for the background

    # Convert numpy array to PIL Image
    rgb_mask_image = Image.fromarray(rgb_array)

    return rgb_mask_image, bbox

def extract_number(file_name):
    # Use a regular expression to extract digits from the file name
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else float('inf')






if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--weight_dtype", type=str, default="float16")
    args.add_argument("--checkpoint_dir", type=str, default="./checkpoints/ppt-v1")
    args.add_argument("--version", type=str, default="ppt-v1")
    args.add_argument("--share", action="store_true")
    args.add_argument("--local_files_only", action="store_true")
    args.add_argument("--port", type=int, default=7861)
    args.add_argument("--range", type=int, required=True, help="Multiplier for file range (0 to N)")
    
    args = args.parse_args()

    weight_dtype = torch.float16 if args.weight_dtype == "float16" else torch.float32
    controller = PowerPaintController(weight_dtype, args.checkpoint_dir, args.local_files_only, args.version)
 
    json_folder_path = "../data_processing/assets/2_mask_generation"
    remove_res_folder_path = "../data_processing/assets/3_editing_pair_construction/remove_add/target_imgs"
    output_folder = "../data_processing/assets/3_editing_pair_construction/remove_add/json_file"

    start_range = args.range * 10000
    end_range = start_range + 10000

    json_files = sorted((json_file for json_file in os.listdir(json_folder_path) if json_file.endswith(".json")),
                        key=extract_number)

    # Filter json_files based on the specified range
    json_files = [json_file for json_file in json_files if start_range <= extract_number(json_file) < end_range]

    print("启动")
    total_count = 0

    for json_file in json_files:
        output_json_path = os.path.join(output_folder, json_file)
        json_path = os.path.join(json_folder_path, json_file)

        if os.path.isfile(output_json_path):
            print("Output file ", output_json_path, " already exists. Skipping.")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_path = data.get("image_path")
            annotations = data.get("annotations", [])
            global_caption = data.get("img_caption")

            if not os.path.isfile(image_path):
                print(f"Image file {image_path} not found. Skipping.")
                continue

            input_image = {}
            global_count = 0
            res_json = {}

            for index, annotation in enumerate(annotations):
                segmentation = annotation['segmentation']
                class_name = annotation.get('class_name')
                detailed_caption = annotation.get('detailed_caption')

                input_image["image"] = Image.open(image_path).convert("RGB")

                mask_img, bbox = segmentation_to_rgb_dilated_erosion_mask(segmentation)
                input_image["mask"] = mask_img

                flag_json = call_gpt(global_caption, class_name, detailed_caption)
                if flag_json is None:
                    continue
                flag = flag_json["flag"]
                if flag == 0:
                    continue

                res1, _ = controller.infer(
                    input_image=input_image,
                    text_guided_prompt="",
                    text_guided_negative_prompt="",
                    shape_guided_prompt="",
                    shape_guided_negative_prompt="",
                    fitting_degree=1,
                    ddim_steps=45,
                    scale=12,
                    seed=random.randint(0, 2147483647),
                    task="object-removal",
                    vertical_expansion_ratio=1,
                    horizontal_expansion_ratio=1,
                    outpaint_prompt="",
                    outpaint_negative_prompt="",
                    removal_prompt="",
                    removal_negative_prompt="",
                    enable_control=False,
                    input_control_image=None,
                    control_type="canny",
                    controlnet_conditioning_scale=None
                )

                class_name = preprocess_class_name(annotation.get('class_name'))
                mask_filename = f"{json_file.replace('.json', '')}_{index + 1}_{class_name}.png"
                remove_res_path = os.path.join(remove_res_folder_path, mask_filename)
                res1[0].save(remove_res_path)

                class_instruction = class_name.replace('_', ' ')
                remove_instruction = "remove the " + class_instruction
                add_instruction = "add the " + class_instruction

                res_json[str(global_count)] = {
                    "target_img_path": remove_res_path,
                    "origin_img_path": image_path,
                    "mask_json_path": json_path,
                    "instruction": [remove_instruction],
                    "task_type": "remove",
                    "detailed_caption": annotation["detailed_caption"],
                    "img_caption": global_caption,
                    "class_origin": class_name,
                    "class_target": "",
                    "bbox": bbox,
                }
                global_count += 1

                res_json[str(global_count)] = {
                    "target_img_path": image_path,
                    "origin_img_path": remove_res_path,
                    "mask_json_path": json_path,
                    "instruction": [add_instruction],
                    "task_type": "add",
                    "detailed_caption": annotation["detailed_caption"],
                    "img_caption": global_caption,
                    "class_origin": "",
                    "class_target": class_name,
                    "bbox": bbox,
                }
                global_count += 1

            if res_json:
                with open(output_json_path, 'w') as f:
                    json.dump(res_json, f, indent=4)

            total_count += global_count

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print("total count:", total_count)