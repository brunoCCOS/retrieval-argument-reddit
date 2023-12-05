from urllib import response
from openai import OpenAI

class Generator:
    def __init__(self) -> None:
        self.client = OpenAI()
    
    def generate(self,prompt):

        response = self.client.chat.completions.create(
        model="gpt-4",
        messages=[],
        temperature=0.7,
        max_tokens=64,
        top_p=1
        )

        return response