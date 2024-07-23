# 基于 GPT4o 的语音助手
无聊时随笔写下の产物
English | 简体中文
***

## 这个项目是什么

此项目允许GPT4o看东西、操控你的电脑。在这个项目里，GPT4o理论上能获得这台电脑的最高权限。所以在 Prompt 里植入一些~~奇怪の东西~~是非常不理智的。

## 这个项目能干什么

GPT4o可以通过这个项目：

- 执行cmd代码
- 提权
- 请求网页
- 搜索 Google（需要Google API和一些科学手段）
- 获取电脑状态

## 快速开始

### 第一步：安装库

在cmd中执行这段代码：

```bash
pip install -r requirements.txt
```

### 第二步：填入apikey

来到app.py

第94行：

```python
client = OpenAI(
    api_key='sk-1145141919810noheheaaaaaa',
)
```
（如果你使用了CloseAI的API代理，则代码可以调整为：
```python
client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',  # 不变
    api_key='sk-29tlFPZwOgqp9yFkY8cFDkn90UXV3NdnQO3AFrUD5ynGXQ5c',
)
```


第108和109行：

```python
GOOGLEapi_key = "niganmahahaeiyo"
GOOGLEcse_id = "nishiwodewoshinideshei"
```
在那儿填入你的GoogleAPIkey

注意: 你需要先申请 [Custom Search JSON API](https://developers.google.com/custom-search/v1/overview)

### 第三步：运行代码

使用这段cmd命令以启动项目

```bash
python app.py
```

### 第四步：可能存在的突发性屎山

在你运行过程中可能会出现一些突发性屎山，这时候你可以考虑提个Issues或者pr...
