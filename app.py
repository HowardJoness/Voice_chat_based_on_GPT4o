from __future__ import print_function
import pyaudio
import wave
import loguru
import numpy as np
import requests
import cv2
import base64
import os
import ctypes, sys
import threading
import platform
import psutil
import json
import pygame
import time
from openai import OpenAI

def get_system_info():
    # 获取操作系统信息
    os_info = platform.uname()
    os_name = os_info.system
    os_version = os_info.version
    device_name = os_info.node
    processor = os_info.processor
    
    # 获取CPU信息
    cpu_logical = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    
    # 获取内存信息
    virtual_memory = psutil.virtual_memory()
    total_memory = virtual_memory.total
    used_memory = virtual_memory.used
    free_memory = virtual_memory.available
    memory_usage = virtual_memory.percent
    
    # 获取硬盘信息
    disk_usage = psutil.disk_usage('/')
    total_disk = disk_usage.total
    used_disk = disk_usage.used
    free_disk = disk_usage.free
    disk_usage_percent = disk_usage.percent
    
    # 获取磁盘的完整信息和分区表的参数
    disk_partitions = psutil.disk_partitions()
    disk_io_counters = psutil.disk_io_counters()
    
    # 获取所有进程的数量
    processes_count = len(psutil.pids())
    
    # 格式化信息
    info = (
        f"操作系统: {os_name} {os_version}\n"
        f"设备名: {device_name}\n"
        f"处理器: {processor}\n"
        f"CPU逻辑个数: {cpu_logical}\n"
        f"CPU物理个数: {cpu_physical}\n"
        f"总内存: {total_memory / (1024**3):.2f} GB\n"
        f"已使用内存: {used_memory / (1024**3):.2f} GB\n"
        f"剩余内存: {free_memory / (1024**3):.2f} GB\n"
        f"内存使用率: {memory_usage}%\n"
        f"总硬盘空间: {total_disk / (1024**3):.2f} GB\n"
        f"已使用硬盘空间: {used_disk / (1024**3):.2f} GB\n"
        f"剩余硬盘空间: {free_disk / (1024**3):.2f} GB\n"
        f"硬盘使用率: {disk_usage_percent}%\n"
        f"磁盘的完整信息: {disk_partitions}\n"
        f"分区表的参数: {disk_io_counters}\n"
        f"硬盘IO总个数: 读={disk_io_counters.read_count}, 写={disk_io_counters.write_count}\n"
        f"所有进程的数量: {processes_count}"
    )
    
    return info


# 初始化日志记录器
logger = loguru.logger

# 清空 chat.json 文件
logger.info("正在清空 chat.json ...")
with open("chat.json", 'w', encoding='utf-8') as f:
    f.write("[]") 
logger.info("清空 chat.json 完毕")

# 检查是否具有管理员权限
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

# OpenAI API 客户端初始化
client = OpenAI(
    api_key='sk-xxx',
)

# 音频相关常量
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = 'test.mp3'  # 用户输入的音频储存文件
MIND_B = 2500  # 最小声音，大于则开始录音，否则结束
DELAY_TIME = 1.3  # 小声1.3秒后自动终止

# Google API 相关常量
GOOGLEapi_key = "xxx"
GOOGLEcse_id = "xxx"

# 初始化 PyAudio
p = pyaudio.PyAudio()

logger.info("打开音频流")
# 打开音频流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# 初始化摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.critical("无法打开摄像头")
    exit()

# 摄像头相关函数
def capture_camera_frame():
    """捕捉摄像头的当前帧并返回图像"""
    ret, frame = cap.read()
    if not ret:
        logger.error("无法读取摄像头内容")
        return None
    return frame

def convert_frame_to_base64(frame):
    """将图像从BGR格式转换为base64编码的JPEG格式"""
    _, buffer = cv2.imencode('.jpg', frame)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

def display_video():
    """显示视频流"""
    while True:
        frame = capture_camera_frame()
        if frame is None:
            break

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 音频播放函数
def play_audio(file_path):
    """播放音频文件"""
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# 上下文管理函数
def load_context():
    """加载上下文信息"""
    try:
        with open('chat.json', 'r', encoding='utf-8') as f:
            context = json.load(f)
        logger.success("上下文加载成功")
    except FileNotFoundError:
        context = []
        logger.error("FileNotFoundError: 文件NotFound")
    return context

def save_context(context):
    """保存上下文信息"""
    with open('chat.json', 'w', encoding='utf-8') as f:
        json.dump(context, f, ensure_ascii=False, indent=4)
    logger.success("保存上下文") 

def get_user_prompt():
    """获取用户输入的初始提示"""
    with open("prompt.txt", "r", encoding='utf-8') as f:
        prompt = f.read()
    return prompt

# 主程序
if __name__ == "__main__":
    # 获取用户的初始提示语，并添加到上下文
    context = load_context()
    initial_prompt = get_user_prompt()
    initial_message = {
        "role": "user",
        "content": initial_prompt
    }
    context.append(initial_message)
    save_context(context)

    # 创建并启动视频显示线程
    video_thread = threading.Thread(target=display_video)
    video_thread.start()

    ChatGPT_DoNotRunCode = True  # ChatGPT 是否执行了 cmd 命令
    CodeReceipt = ""  # cmd 命令回执

    while True:
        logger.debug(f"ChatGPT 没有运行脚本：{ChatGPT_DoNotRunCode}")
        logger.debug(f"ChatGPT 运行的脚本回执：{CodeReceipt}")

        # 若 ChatGPT 没有执行 cmd 命令
        if ChatGPT_DoNotRunCode:
            logger.info("正在监听...")

            # 录音逻辑
            frames = []
            flag = False  # 开始录音的标志
            stat = True   # 判断是否继续录音
            stat2 = False  # 判断声音小了
            tempnum = 0   # 时间计数器
            tempnum2 = 0

            while stat:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.short)
                temp = np.max(audio_data)

                if temp > MIND_B and not flag:
                    flag = True
                    FunctionStartTime = time.time()
                    logger.info("录音开始")
                    tempnum2 = tempnum

                if flag:
                    frames.append(data)

                    if temp < MIND_B and not stat2:
                        stat2 = True
                        tempnum2 = tempnum
                        logger.info("声音小，且之前是大的或刚开始，记录当前点")
                    if temp > MIND_B:
                        stat2 = False
                        tempnum2 = tempnum
                    if tempnum > tempnum2 + DELAY_TIME * 15 and stat2:
                        logger.info(f"间隔{DELAY_TIME:.2f}s后开始检测是否还是小声")
                        if stat2 and temp < MIND_B:
                            stat = False
                            logger.info("小声")
                        else:
                            stat2 = False
                            logger.info("大声")

                tempnum += 1

            logger.info("录音结束")

            # 停止音频流
            stream.stop_stream()
            stream.close()
            p.terminate()
            logger.info(f"录音 | 耗时：{str(time.time() - FunctionStartTime)}")

            # 保存录音文件
            with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))

        FunctionStartTime = time.time()

        # 捕捉摄像头帧
        frame = capture_camera_frame()
        if frame is None:
            break

        # 将摄像头内容转换为base64
        base64_str = convert_frame_to_base64(frame)
        logger.info(f"摄像头转base64 | 耗时：{str(time.time() - FunctionStartTime)}")

        # 语音识别
        url = "https://api.openai-proxy.org/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer sk-xxx"
        }

        if ChatGPT_DoNotRunCode:
            files = {
                'file': ('test.mp3', open(WAVE_OUTPUT_FILENAME, 'rb')),
                'model': (None, 'whisper-1'),
                'response_format': (None, 'json')
            }
            FunctionStartTime = time.time()

            response = requests.post(url, headers=headers, files=files)
            response_text = response.text
            logger.debug(response_text)
            logger.info(f"识别语言内容 | 耗时：{str(time.time() - FunctionStartTime)}")

            try:
                text = json.loads(response_text)["text"]
            except:
                text = "无法提取文本内容"
                logger.error("无法提取文本内容")
            logger.debug(f"识别结果：{text}")
        else:
            text = "[SYSTEM]" + CodeReceipt
            CodeReceipt = ""
            ChatGPT_DoNotRunCode = True

        FunctionStartTime = time.time()

        # 更新上下文
        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_str}",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_str}",
                    },
                },
            ],
        }
        context.append(user_message)
        save_context(context)

        # 调用 OpenAI API 进行对话生成
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=context,
            max_tokens=300,
        )
        logger.debug(f"返回：{response}")
        logger.info(f"对话生成 | 耗时：{str(time.time() - FunctionStartTime)}")

        # 输出对话结果
        assistant_message = {
            "role": "assistant",
            "content": response.choices[0].message.content
        }
        context.append(assistant_message)
        save_context(context)

        ChatGPTResponse = assistant_message["content"]
        
        # 处理不同类型的命令
        if "|cmd:" in ChatGPTResponse:
            command = ChatGPTResponse.split("|cmd:")[1]
            ContextResponse = ChatGPTResponse.split("|cmd:")[0]
            CommandType = 1
        elif "|admin" in ChatGPTResponse:
            ContextResponse = ChatGPTResponse.split("|admin")[0]
            CommandType = 2
        elif "|web:" in ChatGPTResponse:
            command = ChatGPTResponse.split("|web:")[1]
            ContextResponse = ChatGPTResponse.split("|web:")[0]
            CommandType = 3
        elif "|search:" in ChatGPTResponse:
            command = ChatGPTResponse.split("|search:")[1]
            ContextResponse = ChatGPTResponse.split("|search:")[0]
            CommandType = 4
        elif "|status" in ChatGPTResponse:
            ContextResponse = ChatGPTResponse.split("|status")[0]
            CommandType = 5
        else:
            ChatGPT_DoNotRunCode = True
            CommandType = 0
            ContextResponse = ChatGPTResponse

        # 生成语音
        FunctionStartTime = time.time()
        speech_file_path = f"{str(int(time.time()))}speech.mp3"
        response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=ContextResponse
        )

        response.stream_to_file(speech_file_path)
        logger.info(f"保存对话 | 耗时：{str(time.time() - FunctionStartTime)}")
        
        # 播放音频
        play_audio(speech_file_path)

        # 执行命令
        if CommandType == 1:
            # 执行 cmd 命令
            CodeReceipt = os.popen(command).read()
            logger.debug("命令行输出：", CodeReceipt)
            ChatGPT_DoNotRunCode = False

        elif CommandType == 2:
            # 获得管理员权限
            if is_admin():
                CodeReceipt = "ALREADY"
            else:
                try:
                    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
                    CodeReceipt = "OK" if is_admin() else "ERROR"
                except:
                    CodeReceipt = "ERROR"
            ChatGPT_DoNotRunCode = False

        elif CommandType == 3:
            # 获取网页
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.4472.124',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/png,image/jpeg,image/gif,image/x-icon,image/x-svg+xml;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            }
            CodeReceipt = requests.get(command, headers=headers).text
            logger.debug("返回：", CodeReceipt)
            ChatGPT_DoNotRunCode = False

        elif CommandType == 4:
            # 谷歌搜索
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "q": command,
                "key": GOOGLEapi_key,
                "cx": GOOGLEcse_id
            }

            response = requests.get(url, params=params)
            data = response.json()

            # 提取前5个搜索结果
            items = data.get('items', [])[:5]

            result = []
            # 格式化
            for i, item in enumerate(items, start=1):
                result.append({
                    "Result": i,
                    "Title": item.get('title'),
                    "Link": item.get('link'),
                    "Snippet": item.get('snippet')
                })
            CodeReceipt = str(result)
            logger.debug("返回：", CodeReceipt)
            ChatGPT_DoNotRunCode = False

        elif CommandType == 5:
            # 获取电脑运行状态
            CodeReceipt = get_system_info()
            ChatGPT_DoNotRunCode = False


        # 重新打开音频流以便于下次录音
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
