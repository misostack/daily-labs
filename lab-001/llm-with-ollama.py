"""
Lab: Lab-001
Subject: LLM with Ollama
Author: sonnm
"""


import re

from constant import MODEL_NAME
from ollama import chat

USER_GREETING = "Xin chào bạn, tôi muốn hỏi về nextjsvietnam.com"
PROMPT = """
<persona>
{{persona}}
</persona>
<task>
{{task}}
</task>
<context>
{{context}}
</context>
<constraint>
{{constraint}}
</constraint>
<examples>
{{examples}}
</examples>
"""


def extract_json_content(raw_text):
    """
    Trích xuất nội dung JSON từ một chuỗi văn bản thô.
    Cách này bao phủ mọi trường hợp (có/không có ```, có \n, có text thừa ở cuối).
    Nó sẽ tự động tìm từ dấu ngoặc { hoặc [ đầu tiên đến dấu ngoặc } hoặc ] cuối cùng.
    """
    # Sử dụng re.DOTALL để dấu chấm (.) khớp với cả ký tự ngắt dòng \n
    match = re.search(r'(\{.*\}|\[.*\])', raw_text, re.DOTALL)

    if match:
        return match.group(0)

    # Trả về chuỗi gốc đã xóa khoảng trắng nếu không tìm thấy (để json.loads tự báo lỗi)
    return raw_text.strip()


class PromptBuilder:
    def __init__(self):
        self._prompt = PROMPT

    def add_persona(self, persona):
        self._prompt = self._prompt.replace("{{persona}}", persona)

    def add_task(self, task):
        self._prompt = self._prompt.replace("{{task}}", task)

    def add_context(self, context):
        self._prompt = self._prompt.replace("{{context}}", context)

    def add_constraint(self, constraint):
        self._prompt = self._prompt.replace("{{constraint}}", constraint)

    def add_examples(self, examples):
        self._prompt = self._prompt.replace("{{examples}}", examples)

    def get_prompt(self):
        return self._prompt


class OllamaService:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.messages = []

    def add_system_message(self, message):
        self.messages.append({"role": "system", "content": message})

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_assistant_message(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def parse_response(self, response):
        # parse the response to get the answer
        # in this example, we assume the response is in JSON format: {"answer": "the answer"}
        import json

        try:
            cleaned_text = extract_json_content(response.strip())
            data = json.loads(cleaned_text)
            return data.get("answer", "")
        except json.JSONDecodeError:
            return response

    def chat(self, message):
        self.add_user_message(message)
        response = chat(messages=self.messages, model=self.model_name)
        assistant_message = self.parse_response(response.message.content)
        self.add_assistant_message(assistant_message)
        return assistant_message


def exampleContext():
    return """
    1. nextjsvietnam.com là một trang web chuyên về javascript và nodejs.
    2. nextjsvietnam.com có nhiều bài viết về javascript và nodejs.
    3. nextjsvietnam.com được tạo ra bởi SONNM vào năm 2022.
    4. nextjsvietnam.com đã có nhiều từ khoá vào top 3 google: nestjs tutorial, nestjs boilerplate, ...
    5. nextjsvietnam.com cũng có cả bài viết về AWS
    6. nextjsvietnam.com có bài viết về nestjs, nextjs và reactjs
    7. nextjsvietnam.com được lập ra nhằm chia sẻ kiến thức.
    8. nextjsvietnam.com được xây dựng bằng hugo
    """


def exampleContraints():
    return """
    1. Reply in Vietnamese
    2. Reply in one sentence
    3. Reply in JSON format: {"answer": "the answer"}
    4. Follow context strictly, do not use any information outside the context.
    5. Follow the example strictly, do not use any information outside the example.
    """


def exampleExamples():
    return f"""
    <example>
        <user>
            {USER_GREETING}
        </user>
        <assistant>
            Xin chào bạn, tôi có thể giúp gì cho bạn?
        </assistant>
        <user>
            Trang web tạo ra nhằm mục đích gì?
        </user>
        <assistant>
        {{"answer": "Trang web được lập ra nhằm chia sẻ kiến thức. Cảm ơn bạn đã hỏi!"}}
        </assistant>
    </example>
    """


def main():
    # by default when ollama started, it will run the model "llama2-7b-chat" and listen on port 11434
    print("Lab 001: LLM with Ollama!")
    promptBuilder = PromptBuilder()
    promptBuilder.add_persona("You are a helpful assistant.")
    promptBuilder.add_task("Answer the question based on the given context.")
    promptBuilder.add_context(exampleContext())
    promptBuilder.add_constraint(exampleContraints())
    promptBuilder.add_examples(exampleExamples())
    chatService = OllamaService()
    chatService.add_system_message(promptBuilder.get_prompt())
    msg = USER_GREETING
    while msg.lower() != "exit":
        response = chatService.chat(msg)
        print("Assistant:", response)
        msg = input("User: ")


if __name__ == "__main__":
    main()
