import json
import requests
from PIL import Image
from io import BytesIO
import base64
import numpy as np
class ChatGPTImageQuestionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "Custom/ChatGPT"

    def process(self, image, question):
        # if not image:
        #     return ("Error: No image provided.",)
        # if not question:
        #     return ("Error: No question provided.",)

        # Convert image to base64      
        buffered = BytesIO()
        image = np.array(image.cpu())
        # 确保数组形状为 (height, width, 3)
        if image.ndim == 4:  # Batch size 维度存在时
            image = image[0]  # 移除 batch 维度
        if image.shape[0] == 1:  # 如果是单通道
            image = image.squeeze(0)
        # 确保数组的数据类型是 uint8
        image = np.clip(image, 0, 1) 
        image = (image * 255).astype(np.uint8)  # 缩放并转换为 uint8 类型
        # 将 numpy 数组转为 PIL 图像
        img = Image.fromarray(image)
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Call ChatGPT API
        api_key = ""  # Replace with your OpenAI API key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4",
            "messages": [
                
                {"role": "system", "content": "You are a helpful assistant for image analysis."},
                {"role": "user", "content": f"Here is an image in base64 format: {image_base64}. The question is: {question}"},
            ],
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
            response_data = response.json()
            answer = response_data["choices"][0]["message"]["content"].strip()
            print(answer)
            return (answer,)
        except Exception as e:
            print(f"Error: {str(e)}")
            return (f"Error: {str(e)}",)


# Register the node
NODE_CLASS_MAPPINGS = {
    "ChatGPTImageQuestionNode": ChatGPTImageQuestionNode
}

from zhipuai import ZhipuAI
import io

class ZhiPuImageQuestionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = "Custom/ChatGPT"

    def process(self, image, question):
        question = f"这有一张是汽车内饰的设计草稿图。我需要你根据下图帮我进行汽车内饰的设计。请你想象一个{question}风格的汽车内饰。首先请你从整体风格进行描述描述，包含整体的色调以及风格、材质等。然后再非常简短的描述你设计的座椅，方向盘，和显示器的详细样式，需要包含材质以及具体的颜色搭配等（一定注意，如果设计图出现了部件这些再描述，如果没有就不要描述。不需要描述具体功能。）。用英文描述。"

        image = np.array(image.cpu())
        if image.ndim == 4:  
            image = image[0]  # 移除 batch 维度
        print(image.shape)

        # 将np.array转换为图像
        image = np.clip(image, 0, 1)  # 将图像数据的值限制在0到1之间
        image = (image * 255).astype(np.uint8)  # 转换为uint8类型
        image_pil = Image.fromarray(image)
        
        # 创建一个字节流对象
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        img_base = base64.b64encode(img_byte_arr).decode('utf-8')
        client = ZhipuAI(api_key="c9bcc8c8eae1267231a9abc902d7c43d.PdJbzq90JNC8ngNT")
        response = client.chat.completions.create(
            model="glm-4v-plus",  # 填写需要调用的模型名称
            messages=[
            {
                "role": "user",
                "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_base
                    }
                },
                {
                    "type": "text",
                    "text": question
                }
                ]
            }
            ]
        )
        ans = 'Very high-tech. High-definition pictures. Bright and realistic interior lighting.' + response.choices[0].message.content
        
        return (ans,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "ChatGPTImageQuestionNode": ChatGPTImageQuestionNode,
    "ZhiPuImageQuestionNode": ZhiPuImageQuestionNode
}