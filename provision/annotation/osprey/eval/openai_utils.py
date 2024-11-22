from functools import partial
import os
import random
import time
import sys
from glob import glob
import traceback
from pydantic import BaseModel
from typing import Callable, List, Dict, Optional, Union, Literal
import requests
import io
import base64
import json

from tqdm import tqdm

from multiprocessing import Pool
import asyncio
import aiohttp

import ast
from PIL import Image

from openai import OpenAI
from openai import RateLimitError, APIConnectionError
from openai import BadRequestError
import backoff

def read_image(filepath) -> Image.Image:
    if os.path.isfile(filepath):
        raw_image = Image.open(filepath)
    else:
        raw_image = Image.open(requests.get(filepath, stream=True).raw)
    raw_image = raw_image.convert("RGB")
    
    return raw_image

def encode_image(image: Union[str, Image.Image]):
    if isinstance(image, str):  # if it's a file path
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, Image.Image):  # if it's a PIL Image object

        if image.mode == "P":
            if "transparency" in image.info:
                image = image.convert("RGBA")
            else:
                image = image.convert("RGB")
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            return base64.b64encode(output.getvalue()).decode('utf-8')
    else:
        raise ValueError("image_input must be a file path or a PIL Image object")

def is_json_schema(response_format) -> bool:
    if isinstance(response_format, dict):
        return response_format["type"] == "json_schema"
    try:
        return issubclass(response_format, BaseModel)
    except Exception:
        return False

class OpenaiAPI:

    def __init__(self, api_key=None, backoff_time=3, max_tries=3, max_time=60) -> None:
        # if api_key is None:
        #     api_key = open(os.environ['OPENAI_API_KEY']).read().strip()
        self.client = OpenAI(api_key=api_key)
        self.backoff_time = backoff_time
        self.max_tries = max_tries
        self.max_time = max_time
    
    @staticmethod
    def get_response_format(response_format: str | BaseModel) -> dict | BaseModel:
        ''' Returns the response format for the OpenAI API call. '''
        if response_format is None:
            return {"type": "text"}
        
        if isinstance(response_format, str):
            if response_format in ["json", "json_object"]:
                return {"type": "json_object"}
            else:
                raise ValueError(f"Invalid response format: {response_format}")
        
        return response_format

    def generate_image(self, text, n, model="dall-e-3", size=256):
        images = []
        c = 0
        while c < self.max_tries:
            try :
                image_resp = self.client.images.generate(model=model, prompt=text, n=n, size=f"{size}x{size}", quality="standard")
                for d in image_resp['data']:
                    image_url = d['url']
                    images.append(read_image(image_url))
                break
            except Exception:
                error = sys.exc_info()[0]
                if error == BadRequestError:
                    print(f"BadRequestError\nQuery:\n\n{text}\n\n")
                    break
                else:
                    print('Error:', sys.exc_info())
                    time.sleep(self.backoff_time)
                    c+=1
                
        return images

    @backoff.on_exception(backoff.expo, (RateLimitError, APIConnectionError), max_time=60)
    def _complete_chat(self, messages, model='gpt-3.5-turbo', max_tokens=256, response_format=None, 
                 top_p = 1.0, temperature=1.0, n=1, stop = '\n\n\n', 
                 ):

        # call GPT-3 API until result is provided and then return it
        response = None
        c = 0
        
        if is_json_schema(response_format):
            completions_fn = self.client.beta.chat.completions.parse
        else:
            completions_fn = self.client.chat.completions.create
        if 'o1' not in model:
            completions_fn = partial(completions_fn, stop=stop)
                    
        while c < self.max_tries:
            try:
                response = completions_fn(
                    messages=messages, model=model, max_completion_tokens=max_tokens, response_format=response_format,
                    temperature=temperature, top_p=float(top_p), n=n,
                )
                return response
            except Exception as e: 
                error = sys.exc_info()[0]
                if error == BadRequestError:
                    print(f"BadRequestError\nQuery:\n\n{messages}\n\n")
                    print(sys.exc_info())
                    break
                else:
                    print(e)
                    print('Error:', sys.exc_info())
                    time.sleep(self.backoff_time)
                    c+=1

        return response

    def call_chatgpt(
        self,
        model: str,
        sys_prompt:str=None,
        usr_prompt:str=None,
        examples: Optional[List[Dict[str,str]]]=None,
        response_format: str | BaseModel=None,
        max_tokens=256,
        top_p=1.0,
        temperature=1.0,
    ):  
        '''
        This function is used to generate a response from the OpenAI Chat model for a given user prompt. 
        
        Parameters:
        sys_prompt (str): The system prompt that sets the behavior of the assistant.
        usr_prompt (str): The user prompt that the model responds to.
        model (str): The model version to be used for generating the response.
        examples (List, optional): Examples for in context learning. Each example should be a dictionary with the following keys:
            - 'role' (str): The role of the message, either 'user' or 'system'.
            - 'content' (str): The content of the message.
            The examples should be provided in pairs of user and system responses.
        max_tokens (int, optional): The maximum length of the model's response. Default is 256.
        top_p (float, optional): The nucleus sampling parameter that controls the randomness of the model's output. Default is 1.0.
        temperature (float, optional): The parameter that controls the randomness of the model's output. Default is 1.0.
        
        Returns:
        tuple: 
            - A tuple containing the model's response and the total number of tokens used in the response. 
            - Returns None if the request fails or an error is raised.
        ''' 

        messages = []
        if sys_prompt is not None:
            messages.append({"role": "system", "content": sys_prompt})
        if usr_prompt is None:
            usr_prompt = ""
        user_message = [{"role": "user", "content": usr_prompt}]

        response_format = self.get_response_format(response_format)
        response_is_json_schema = is_json_schema(response_format)

        # Add in-context examples
        if examples is not None:
            # verify if example is valid message
            assert len(examples) % 2 == 0, "Examples should be in pairs of user and system response"
            for idx, example in enumerate(examples):
                assert 'role' in example, "Role should be provided in your example: {}".format(example)
                assert 'content' in example, "Content should be provided in your example: {}".format(example)
                if idx % 2 == 0:
                    expected_role = 'user'
                else:
                    expected_role = 'system'
                assert example['role'] == expected_role, "Expected role: {} but found role: {} in example {}".format(expected_role, example['role'], example)
            messages = messages +  examples + user_message
        else:
            messages = messages + user_message
        try:
            response = self._complete_chat(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                response_format=response_format,
                top_p=top_p,
                temperature=temperature,
                n=1
            )
            if response is None:
                return None
            if response_is_json_schema:
                result: BaseModel = response.choices[0].message.parsed
            else:
                result: str = response.choices[0].message.content
            return (result, response.usage.total_tokens)
            
        except AttributeError as e:
            traceback.print_exc()
            return None
    
    def call_chatgpt_vision(
        self,
        model: str,
        sys_prompt: str = None,
        usr_prompt: str = None,
        image_input: Union[str, List[str], Image.Image, List[Image.Image]] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        response_format: str | BaseModel = None,
        image_detail: str | list[str] = 'auto',
        max_tokens=256,
        top_p=1.0,
        temperature=1.0,
    ):  
        '''
        This function is used to generate a response from the OpenAI Chat model with optional image input.
        
        Parameters:
        sys_prompt (str): The system prompt that sets the behavior of the assistant.
        usr_prompt (str): The user prompt that the model responds to.
        image_input (str, Image.Image, list): URL or local path to image, or PIL image. Optionally takes in a list.
        model (str): The model version to be used for generating the response.
        examples (List, optional): Examples for in-context learning. Optionally can include image input.
        max_tokens (int, optional): The maximum length of the model's response. Default is 256.
        top_p (float, optional): The nucleus sampling parameter that controls the randomness of the model's output. Default is 1.0.
        temperature (float, optional): The parameter that controls the randomness of the model's output. Default is 1.0.
        
        Returns:
        tuple: 
            - A tuple containing the model's response and the total number of tokens used in the response. 
            - Returns None if the request fails or an error is raised.
        ''' 

        messages = []
        if sys_prompt is not None:
            messages.append({"role": "system", "content": sys_prompt})

        if usr_prompt is None:
            usr_prompt = ""

        # Prepare user message with optional image input
        def get_user_content(image_input, usr_prompt, image_detail):

            if not isinstance(image_input, list):
                image_input = [image_input]

            if not isinstance(image_detail, list):
                image_detail = [image_detail] * len(image_input)
            
            assert len(image_input) == len(image_detail), "Length of image_input and image_detail should be the same"
            
            content = []
            for idx,im in enumerate(image_input):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self.get_image_content(im),
                        "detail": image_detail[idx],
                    }
                })
            content.append({
                        "type": "text",
                        "text": usr_prompt,
                    })

            return content

        user_content = get_user_content(image_input, usr_prompt, image_detail)
        user_message = [{"role": "user", "content": user_content}]

        response_format = self.get_response_format(response_format)
        response_is_json_schema = is_json_schema(response_format)

        # Add in-context examples
        if examples is not None:
            vis_examples = []
            # verify if example is valid message
            assert len(examples) % 2 == 0, "Examples should be in pairs of user and system response"
            for idx, example in enumerate(examples):
                assert 'role' in example, f"Role should be provided in your example: {example}"
                assert 'content' in example, f"Content should be provided in your example: {example}"

                content = example['content']
                if idx % 2 == 0:
                    expected_role = 'user'
                    if 'image' in example:
                        content = user_content(example['image'], example['content'])
                else:
                    expected_role = 'system'
                    assert 'image' not in example, "Image should not be provided in system response."

                assert example['role'] == expected_role, f"Expected role: {expected_role} but found role: {example['role']} in example {example}"
                vis_examples.append({"role": example['role'], "content": content})
            messages = messages + vis_examples + user_message
        else:
            messages = messages + user_message

        try:
            response = self._complete_chat(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                response_format=response_format,
                top_p=top_p,
                temperature=temperature,
                n=1
            )
            if response is None:
                return None
            if response_is_json_schema:
                result: BaseModel = response.choices[0].message.parsed
            else:
                result: str = response.choices[0].message.content
            return (result, response.usage)

        except AttributeError as e:
            traceback.print_exc()
            return None

    @staticmethod
    def get_image_content(image_input: str | Image.Image) -> str:
        '''
        Transforms local image to bytes to send over vision api, else use the http url.
        '''
        if isinstance(image_input, str):
            if ('https://' in image_input or 'http://' in image_input):
                image_content = image_input
            else: # local file
                assert os.path.isfile(image_input)
                image_content = f"data:image/png;base64,{encode_image(image_input)}"
        else:
            image_content = f"data:image/png;base64,{encode_image(image_input)}"
        return image_content

    @staticmethod
    def parse_json_from_text(raw_text: str) -> dict:
        """
        Extracts and parses the first JSON block found in the given raw text.
        The JSON block can be marked by ```json or ```.
        
        :param raw_text: The raw text containing the JSON block.
        :return: A dictionary representation of the JSON block, if found and successfully parsed; otherwise, None.
        """

        # See if you can directly parse the data
        try:
            data = ast.literal_eval(raw_text)
            return data
        except Exception:
            pass

        # Try to find the start of the JSON block, considering both ```json and ``` markers
        start_markers = ["```json", "```"]
        start = -1
        for marker in start_markers:
            start = raw_text.find(marker)
            if start != -1:
                start += len(marker)
                break
        
        # If we found a start marker, then find the end of the JSON block
        if start != -1:
            end = raw_text.find("```", start)

            # Extract and strip the JSON string
            raw_text = raw_text[start:end]
        
        # Parse the JSON string into a Python dictionary
        try:
            json_data = json.loads(raw_text)
            return json_data
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            return None

class MultiProcessCaller:

    """" Helper function to call function over list of data with multiple processes while showing progress""" 

    @classmethod
    def call_multi_process(cls, fn: Callable, data: List, num_processes=8):
        result = []
        p = Pool(num_processes)
        pbar = tqdm(total=len(data))
        for i in p.imap_unordered(fn, data):
            if i is not None:
                result.append(i)
            pbar.update()

        return result
    
    @classmethod
    def batch_process_save(cls, data: List, openai_call: Callable, output_file: str, num_processes: int, 
                       batch_size=1000, 
                       sort_key: str=None, 
                       write_mode:Literal['a','w']='a'):
        """
        Process data in batches and save the results to a file in sorted order if a key is given.

        :param data: The list data to process.
        :param openai_call: The function to be called in multi-process.
        :param num_processes: The number of processes to use.
        :param output_file: The file to save the results to.
        :param batch_size: The size of each batch of data to process.
        :param key: The key to sort the results by, if not None.
        """
        all_result: List = []
        for idx in tqdm(range(0, len(data), batch_size)):
            
            # Process in batches
            batch_result: list = cls.call_multi_process(
                openai_call, data[idx:idx+batch_size], num_processes=num_processes
            )
            batch_result = [r for r in batch_result if r is not None]

            # Sort if a key is provided
            if sort_key is not None:
                batch_result = sorted(
                    batch_result,
                    key=lambda x: x[sort_key]
                )
            m = 'a' if write_mode == 'a' or idx > 0 else 'w'
            with open(output_file, m) as f:
                # Save batch results to file
                for r in batch_result:
                    f.write(json.dumps(r) + '\n')
                print(f'Save {len(batch_result)} batch data to {output_file}')

            all_result += batch_result
        
        return all_result

class GPTExamples:
    def __init__(self, cap_dir, conv_dir) -> None:

        caps = glob(os.path.join(cap_dir, '*_cap.txt')) # user prompt
        convs = glob(os.path.join(conv_dir, '*_conv.txt')) # assistant prompt
 
        self.idx_cap: dict = {self.get_index_from_file(d): open(d).read() for d in caps}
        self.idx_conv: dict = {self.get_index_from_file(d): open(d).read() for d in convs}

        assert len(self.idx_cap) == len(self.idx_conv), "Number of user prompts and assistant prompts should be the same"

        self.num_examples = len(self.idx_cap)
    
    def get_indices(self,):
        return list(self.idx_conv.keys())
    
    def get_index_from_file(self, file_name) -> int:
        return int(os.path.basename(file_name).split('_')[0])    
    
    def get_example(self, idx, image: str=None) -> List[Dict]:
        ''' image: path to image or url link'''

        user_content = self.idx_cap[idx]
        sys_content = self.idx_conv[idx]

        if image is not None:
            return [
                {'role': 'user', 'content': user_content, 'image': image}, 
                {'role': 'system', 'content': sys_content}
            ]
        else:
            return [
                {'role': 'user', 'content': user_content}, 
                {'role': 'system', 'content': sys_content}
            ]
    
    def __call__(self, n: int, random_sample: bool=False):
            """
            Randomly sample n examples.

            Parameters:
            - n (int): The number of examples to sample.
            - random_sample (bool, optional): If True, samples will be randomly selected. If False, samples will be selected in order.

            Returns:
            - examples (list): A list of sampled examples.

            Raises:
            - ValueError: If n is greater than the number of available indices.

            """
            if n > self.num_examples:
                raise ValueError("n cannot be greater than the number of indices")
            if random_sample:
                idxs = self.get_indices()
                idxs = random.sample(idxs, n)
            else:
                idxs = self.get_indices()[:n]
            
            examples = []
            for idx in idxs:
                examples += self.get_example(idx)
            return examples
