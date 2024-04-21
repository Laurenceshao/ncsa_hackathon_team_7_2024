import os
from openai import AzureOpenAI

AZURE_OPENAI_API_VERSION="2023-07-01-preview"
AZURE_OPENAI_API_KEY="dc528eaf83724782914e171f3bbdaeda"
AZURE_OPENAI_ENDPOINT="https://uiuc-chat-canada-east.openai.azure.com/"
AZURE_MODEL_VERSION="gpt-4-hackathon"

client = AzureOpenAI(
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_MODEL_VERSION)

def run_gpt4(prompt):
  completion = client.chat.completions.create(
    model="gpt-4-hackathon",
    messages=[
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": prompt}
    ]
  )
  return completion.choices[0].message.content