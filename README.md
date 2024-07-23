# Voice chat based on GPT4o
When i have nothing to do, i wrote this project😂

***

## What is this program?

This program allows GPT4o to see things and operate your computer. In this program, GPT4o can theoretically obtain the highest permissions on your computer. Therefore, it is very dangerous to add some unreasonable content to the prompt.

## How can this program do?

In this program, GPT4o can:

- use cmd command
- get the administrator jurisdiction
- request web page
- search Google (need Google API)
- get the computer status

## Quick start

### Step 1. Install the packages

run this command in your CMD:

```bash
pip install -r requirements.txt
```



### Step 2. Write your API key

Go to app.py.

In line 94:

```python
client = OpenAI(
    api_key='put your OpenAI API key here',
)
```



In line 108 & 109:

```python
GOOGLEapi_key = "englishor69spanish"
GOOGLEcse_id = "whoevermovefirstisgay"
```

Put your Google API key there.

Attention: u should have the [Custom Search JSON API](https://developers.google.com/custom-search/v1/overview) first.

### Step 3. Run code

run this in your CMD or Powershell to run this project:

```bash
python app.py
```

### Step 4. Maybe some bugs

Maybe it will have some bugs in your computer. You can feedback it in the "Issues".

