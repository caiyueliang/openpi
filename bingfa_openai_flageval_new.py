import io
import requests
import json
from PIL import Image
import numpy as np
import argparse
import threading
import time
import csv
from openai import OpenAI
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
from argparse import ArgumentParser

import argparse
import threading
import time
import csv
from openai import OpenAI
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
from argparse import ArgumentParser
import requests
import base64
import json
from PIL import Image
from io import BytesIO
import logging


# online_url = "https://cloud.zidongtaichu.com/maas/v1"
# online_url = "https://ai-maas.wair.ac.cn/maas/v1"
online_url = "https://platform-cloud.wair.ac.cn/api/v1/infer/11776/v1"

client = OpenAI(api_key='EMPTY', 
                base_url=online_url,
                # default_headers = {"Authorization": 'Bearer ryvsk3zz73419gkgubrnvufp'
                default_headers = {"Authorization": 'Admin'}
        )

def image_to_base64(image_path,timeout=8):
    """将图片转换为base64字符串"""
    if image_path.startswith('http'):
        response = requests.get(image_path, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    else: 
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# image_path = "/mnt/publish-data/train_data/mllm/local_media_dir/image_cache/75145b0a-12ef-48a6-9f9d-a213b6ed141a.jpg" 
# filename = '/home/huangrenhe/limin/think_with_image/think_with_image_new_3.json' # angle 80～159 
# with open(filename, 'r', encoding='utf-8') as file:
#     data_new = json.load(file)
img_contents = []
# ques = []
# for i in range(5,7):
#     img_contents.append(image_to_base64(data_new[i]["image_oss_2k"]))
#     ques.append(data_new[i]["question_cn"])


formaturl = False
formaturl = True
if formaturl:
    image_url = "https://zdtc-cdn.wair.ac.cn/assets/caption_2_file_1.jpeg"
    messages = [
    {
        'role':'user',
        'content': [{'type': 'text','text': '详细描述图片',}, 
                    {'type': 'image_url','image_url': {'url': image_url},},
                    ],
    },]
else:
    image_path =  "/home/huangrenhe/limin/think_with_image/log/flag_eva/caption_2_file_1.jpeg"
    # img_content = image_to_base64("https://zdtc-cdn.wair.ac.cn/assets/caption_2_file_1.jpeg")
    # image_to_base64("/home/huangrenhe/limin/think_with_image/log/flag_eva/821f0024-00d9-4b84-ae7d-d5d136d69f9a.jpg")
    image_path =  "/home/huangrenhe/limin/think_with_image/log/flag_eva/821f0024-00d9-4b84-ae7d-d5d136d69f9a.jpg"
    img_content = image_to_base64(image_path)
    messages = [
    {
        'role':'user',
        'content': [{'type': 'text','text': '详细描述图片',}, 
                    {'type': 'image_url','image_url': {'url': f"data:image/jpeg;base64,{img_content}"},},
                    ],
    },]  
    # f"data:image/jpeg;base64,{img_contents[0]}"
    # "https://zdtc-cdn.wair.ac.cn/assets/caption_2_file_1.jpeg"
    # f"data:image/jpeg;base64,{img_content}"
stream = True
def run_backend(
    i: int,
    input_token_lens: list,
    output_token_lens: list,
    first_token_times: list,
    next_token_times: list,
):
    next_token_times[i] = 0
    input_token_lens[i] = 0
    first_token_times[i] = 0
    output_token_lens[i] = 0

    dict_re  = {}
    t_1 = time.perf_counter() 

    response = client.chat.completions.create(
        # model='taichu_vl_new',
        model='taichumm',
        messages= messages,
        temperature=0.8,
        max_tokens=32,
        stream = stream ,
        stream_options = {"include_usage": True,"continuous_usage_stats":True},
        extra_body = { "min_tokens": 32}, 
        )
    first_token = True 
    t0 = time.perf_counter() 
    logging.warning(f'load img:{t0 - t_1}')
    if stream:  
        for dict_obj in response:
              
            if first_token:
                t1 = time.perf_counter()
                first_token = False
                first_token_times[i] = t1 - t_1
                logging.warning(first_token_times[i])
                # logging.warning(dict_obj)
            if dict_obj.usage is not None:
                t2 = time.perf_counter()
                 
                if dict_obj.usage.completion_tokens > 1:
                    next_token_times[i] = (t2-t1)/(dict_obj.usage.completion_tokens-1)
                # input_token_len = dict_obj.usage.prompt_tokens 
                # output_token_len = dict_obj.usage.completion_tokens  
                input_token_lens[i] = dict_obj.usage.prompt_tokens 
                output_token_lens[i] = dict_obj.usage.completion_tokens  
    
    

def send_requests(thread_num):
    threads = []
    input_token_lens = [None] * thread_num
    output_token_lens = [None] * thread_num
    first_token_times = [None] * thread_num
    next_token_times = [None] * thread_num

    t0 = time.perf_counter()
    for i in range(thread_num):
        t = threading.Thread(target=run_backend, args=(i,input_token_lens, output_token_lens, first_token_times, next_token_times))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    t1 = time.perf_counter()-t0

    batch_size = 1.0
    logging.warning(f'first_token_times:{first_token_times}')
    return_list = []
    return_list.append(thread_num)  # 并发请求数量
    return_list.append(batch_size)  # 1
    return_list.append(int(sum(input_token_lens)))      # 输入tokens数量
    return_list.append(int(sum(output_token_lens)))     # 输出tokens数量
    return_list.append(round(sum(first_token_times)/thread_num,3)) # 第一个token到达时间
    return_list.append(round(sum(next_token_times)/thread_num,3))  # decode阶段生成每个token间隔时间
    return_list.append(round(t1,3))                                # 总耗时 e2e时间
    return_list.append(round(thread_num/(sum(first_token_times)/thread_num),3))                     # 1秒可以响应得请求数量 首个tokne时间\
    return_list.append(round(thread_num/t1,3))                     # 1秒可以响应得请求数量 首个tokne时间
    return_list.append(round(sum(output_token_lens)/t1,3))         # 吞吐   输出token数量/总时间；
    return return_list
     
if __name__ == "__main__":
    parser = ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="round of test.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="round of test.",
    )

     
    # parser.add_argument(
    #     "--formaturl",
    #     type=bool,
    #     default=True,
    #     help="round of test.",
    # )    
    args = parser.parse_args()
    t_sum = 0
    cycyle_cnt = args.round
    num_client = args.concurrency # 
    cycyle_cnt = args.round

    # csv_path = "performance_lmdploy_fp16_a800_80g_torch_1.csv"
    csv_path = "taichu_vl_new_url.csv"
    # csv_path = "t_multi_1024_1_1.csv"
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'thread_num',
            'batch_size', 
            'input_token_len',
            'output_token_len',
            'first_token_time(s)',
            'next_token_time(s)',
            'time(s)',
            'qps_ttft(requests/s)',
            'qps(requests/s)',
            'throughput(tokens/s)',
        ])

        for i in range(0, cycyle_cnt):
            for j in range(0,num_client):
                return_list = send_requests(2 ** j) 
                # return_list = send_requests(4) 
                logging.warning(return_list)
                csv_writer.writerow(return_list)



                