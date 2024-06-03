import os
from openai import AzureOpenAI
from openai import OpenAI
import random
import base64
import torch 
import numpy as np
from PIL import Image
import io
from io import BytesIO
from dotenv import load_dotenv

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFile, ImageSequence
import torch

import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
import logging

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

import numpy as np
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args

import importlib

import folder_paths
import latent_preview
import node_helpers
from ultralytics import YOLO
prompts =[] #tag缓冲临时存储

load_dotenv()
class Example:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "frame_count": ("INT", {
                    "default": 0, 
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 1, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "float_field": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number"}),
                "print_to_screen": (["enable", "disable"],),
                "string_field": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "Hello World!"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("视频反推词",)

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(self, text, string_field, frame_count, float_field, print_to_screen):
        if print_to_screen == "enable":
            print(f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {frame_count}
                float_field: {float_field}
            """)
        #do some processing on the image, in this example I just invert it
        #image = 1.0 - image
        print(text +"AFKAFAWSFAFDSF")
        return (text +"dawdadawdad",)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique


class Luamoon:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "tag_in": ("STRING", {"multiline": True}),
                "api_key": ("STRING",{"multiline": False}),
                # "float_field": ("FLOAT", {
                #     "default": 1.0,
                #     "min": 0.0,
                #     "max": 10.0,
                #     "step": 0.01,
                #     "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                #     "display": "number"}),
                # "print_to_screen": (["enable", "disable"],),

                "azure_endpoint": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                }),
                "field": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "过滤出和人体动作相关的词语"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tag_out",)

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(self, tag_in, api_key,azure_endpoint,field):
                    
        # if print_to_screen == "enable":
        #     print(f"""Your input contains:
        #         string_field aka input text: {field}
        #         int_field: {api_key}
        #         float_field: {float_field}
        #     """)
        # #do some processing on the image, in this example I just invert it
        # #image = 1.0 - image
        

        client = AzureOpenAI(
        azure_endpoint = azure_endpoint, 
        api_key=api_key, 
        api_version="2024-02-15-preview"
        )


        message_text = [{"role":"user","content":f'''
        <tag>{tag_in}</tag>,tag标签中的词语描述了图片的内容,<field>{field}</field>,
        field标签中描述了想要过滤出来的词语，请按照field的过滤要求从tag标签中筛选出来。最终仅输出符合要求的词语，直接输出词语，按照逗号隔开。
        '''}]

        completion = client.chat.completions.create(
        model="faxing-gpt35", # model = "deployment_name"
        messages = message_text,
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )

        print(completion.choices[0].message.content)  


        return (completion.choices[0].message.content,)


class Translate:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(self, prompt):
                    
        # if print_to_screen == "enable":
        #     print(f"""Your input contains:
        #         string_field aka input text: {field}
        #         int_field: {api_key}
        #         float_field: {float_field}
        #     """)
        # #do some processing on the image, in this example I just invert it
        # #image = 1.0 - image
        

        client = AzureOpenAI(
        azure_endpoint = "https://faxing.openai.azure.com/", 
        api_key="bb36bca535a54b1bafe3d9e6216a3c4f", 
        api_version="2024-02-15-preview"
        )


        messages=[
            {"role": "system", "content": "你是一个中文翻译为英文的专家，需要翻译的中文放在了<text></text>这个xml标签内，你只需要将标签内的中文翻译为英文。特别情况下，标签内如果原本就是英文，你只需要将原文返回给我。如果标签内是中英文混杂，你需要将它整体翻译为更通顺的英文给我。请注意，你仅回答我翻译好的英文内容的字符串，无<text>符号等其它无关的内容。"},
            {"role": "user", "content": f"<text>{prompt}</text>"}
        ]

        completion = client.chat.completions.create(
        model="faxing-gpt35", # model = "deployment_name"
        messages = messages,
        temperature=0.7,
        max_tokens=4096,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )

        print(completion.choices[0].message.content)  


        return (completion.choices[0].message.content,)


class KV:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Elements": ("STRING", {"multiline": True}),
                "Interaction": ("STRING", {"multiline": True}),
                "Composition": ("STRING", {"multiline": True}),
                "Lighting": ("STRING", {"multiline": True}),
                "Visual_Hierarchy": ("STRING", {"multiline": True}),
                "Atmospheric_Expression": ("STRING", {"multiline": True}),
                "Logo": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    #OUTPUT_IS_LIST = (True,)
    FUNCTION = "test"
    OUTPUT_NODE = False



    CATEGORY = "Example"

    def test(self, Elements,Interaction,Composition,Lighting,Visual_Hierarchy,Atmospheric_Expression,Logo):
                    
        example_texts = []
        #读取名为kv文件夹下的所有以.txt结尾的文本文件内容，存储到以example_texts =[] 的列表中。
        for filename in os.listdir('kv'):
            if filename.endswith('.txt'):
                with open(os.path.join('kv', filename), 'r',encoding='utf-8') as f:
                    example_texts.append(f.read())


        #print(example_texts[5])
        #在example_texts = []中随机选取4个元素

        random_texts = random.sample(example_texts, 4)

        #print(random_texts)

        control_prompt_for_gpt35 = f'''
        Key Visual Definition and Design Elements

        In the field of graphic design, Key Visual is an important design element commonly used to represent a brand, product, or event, conveying its core message and image. Below is a detailed introduction to Key Visual:

        1. Definition and Function:
        - Definition: Key Visual refers to the most important and core visual element used in advertising, brand promotion, and event marketing.
        - Function:
        - Reinforce brand image: Key Visual showcases the core features, ideas, and image of the brand.
        - Increase brand recognition: Unique and attractive Key Visuals can help a brand stand out in a competitive market.
        - Convey information: Through images and design elements, Key Visuals convey the themes, ideas, and characteristics of a product or event.
        - Attract target audience: Well-designed Key Visuals can grab the attention of the target audience and spark their interest.

        2. Design Elements:
        - Image or graphic: Usually represents the brand, product, or event.
        - Color: Color selection is crucial for Key Visuals as it directly impacts the visual communication effect and emotional expression.
        - Typography: Font choice is also a key factor in design; appropriate fonts can enhance the overall feel of the Key Visual and its message delivery.
        - Layout: How to arrange and organize elements such as images and text to achieve the best visual effect.
        - Iconic elements: Each brand or event may have its own unique iconic elements, such as logos, symbols, etc., which are often integrated into the Key Visual.
        <dim>
        Elements: {Elements},
        Interaction: {Interaction},
        Composition: {Composition},
        Lighting: {Lighting},
        Visual Hierarchy: {Visual_Hierarchy},
        Atmospheric Expression: {Atmospheric_Expression},
        Logo: {Logo}
        </dim>
        <dim_explain>
        Elements: Characters and Setting [Number of characters, who the main character is, what the setting is (weather, environment, architecture, vegetation, etc.)]
        Interaction: Character Actions [What the main character is doing, interactions between characters]
        Color: Generally, provide a main color tone.
        Composition: Scenery, Perspective, Viewpoint [Mid-shot, close-up/fisheye, wide-angle, human-eye perspective/overhead, looking up, looking straight]
        Lighting: 3D rendering will involve, referring to photographic terms.
        Visual Hierarchy: Typically centered around a specific character, but not necessarily in the center.
        Atmospheric Expression: Emotional atmosphere.
        Logo: A visual representation of the brand or event.
        </dim_explain>
        <example>
        {random_texts[0]}
        </example>
        <example>
        {random_texts[1]}
        </example>
        <example>
        {random_texts[2]}
        </example>
        <example>
        {random_texts[3]}
        </example>
        <prompt>
        {Elements},{Interaction},{Composition},{Lighting},{Visual_Hierarchy},{Atmospheric_Expression},{Logo},
        </prompt>
        你是一位提示词工程师，你仔细学习和模仿<example>标签中提示词的例子，这是一种描述图片构图的提示词。现在，你根据<prompt>中提供的提示词需求，要重点突出人物的位置与结构，人物与景物的空间关系,logo的文字内容和位置，仿照<example>的格式写一个提示词，不要把维度写出来，直接整体的，英文的自然语言，不超过150个英文单词。
        '''

        from openai import AzureOpenAI

        #faxing-gpt4,faxing-gpt35,faxing-gpt4-vp

        client = AzureOpenAI(
        azure_endpoint = "https://faxing.openai.azure.com/", 
        api_key="bb36bca535a54b1bafe3d9e6216a3c4f", 
        api_version="2024-02-15-preview"
        )

        message_text = [{"role":"user","content":control_prompt_for_gpt35}]
        completion = client.chat.completions.create(
        model="faxing-gpt35", # model = "deployment_name"
        messages = message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )

        print(completion.choices[0].message.content) 

        #return {"ui": {"tags": completion.choices[0].message.content}, "result": (completion.choices[0].message.content,)}
        return (completion.choices[0].message.content,)


class GptTag:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        
       return {"required": 
                    {"image": ("IMAGE", ),
                    },
                }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "Example"

    def test(self, image):
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        

            # 去掉批次维度
        image = image.squeeze(0)

        if image.ndim == 3 and image.shape[2] == 3:
            # 将图像数据类型转换为uint8 (因为原始数据在[0, 1]之间)
            np_image = (image * 255).byte().numpy()
            # 确保是HWC排列格式
            if np_image.shape[2] != 3:
                raise ValueError("Image is not in HWC format")
        else:
            raise ValueError("Input tensor must have shape (H, W, 3) after processing")
        
        # 将NumPy数组转换成Pillow图像对象
        pil_image = Image.fromarray(np_image)
        
        # 将Pillow图像保存到内存中，并转换成Base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")  # 你可以改成"JPEG"
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        #print(img_str)

        prompt = '''
        As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image.
        '''

    

        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. 不得包含sketch,black and white等形容图片风格和类型的单词.务必注意，每个tag仅仅是客观元素的内容描述，不得有风格、颜色、线条、氛围等。 Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image。"},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_str}",
                        "detail": "low"
                    },
                    },
                ],
                }
            ],
            max_tokens=300,
        )

        
        return (response.choices[0].message.content,)


class SegImage:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required": {"image": (sorted(files), {"image_upload": True})}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "seg_image"

    def seg_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.open_image(image_path)

        # 初始化 YOLOv8 模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")
        model = YOLO('yolo_model/yolov8s.pt').to(device)

        output_images = []
        output_masks = []
        
        for i in ImageSequence.Iterator(img):
            prev_value = None
            try:
                i = ImageOps.exif_transpose(i)
            except OSError:
                prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                i = ImageOps.exif_transpose(i)
            finally:
                if prev_value is not None:
                    ImageFile.LOAD_TRUNCATED_IMAGES = prev_value

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image_rgb = i.convert("RGB")  # 转换为 RGB 模式
            image_rgb = np.array(image_rgb)  # 转换为 NumPy 数组
            image_rgb = image_rgb.astype(np.float32) / 255.0  # 归一化

            # 进行目标检测
            results = model(image_rgb, conf=0.01)
            if results and len(results) > 0:
                result = results[0]
                detections = result.boxes.xyxy.cpu().numpy()
                if detections.shape[0] > 0:
                    x_min, y_min, x_max, y_max = map(int, detections[0][:4])
                    x_min = max(x_min - 100, 0)
                    y_min = max(y_min - 100, 0)
                    x_max = min(x_max + 100, image_rgb.shape[1])
                    y_max = min(y_max + 100, image_rgb.shape[0])

                    cropped_image = image_rgb[y_min:y_max, x_min:x_max]
                    output_images.append(cropped_image)
                    # 为了保持返回值与最开始规定的格式相同，这里添加了一个与裁剪后的图像相同大小的空掩码
                    mask_shape = cropped_image.shape[:-1] + (1,)
                    mask = np.zeros(mask_shape, dtype=np.float32)
                    output_masks.append(mask)

        # 将列表转换为张量，并添加 batch 维度
        output_image_tensor = torch.tensor(output_images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        output_mask_tensor = torch.tensor(output_masks, dtype=torch.float32, device=device).permute(0, 3, 1, 2)

        return output_image_tensor, output_mask_tensor


NODE_CLASS_MAPPINGS = {
    "Example": Example,
    "Luamoon": Luamoon,
    "Translate":Translate,
    "KV":KV,
    "GptTag":GptTag,
    "SegImage":SegImage
    
    
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Example": "LiuWei Node",
    "Luamoon": "Azure tag filter",
    "Translate":"Prompt zh",
    "KV":"Kv",
    "GptTag":"GptTag",
    "SegImage":"SegImage"
}
