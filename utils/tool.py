import os
from typing import List

import langchain
from langchain.agents import load_tools
from langchain.agents.agent_toolkits.file_management.toolkit import FileManagementToolkit
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from utils.file_io import load_jsonl

AZURE_OPENAI_API_VERSION="2023-07-01-preview"
AZURE_OPENAI_API_KEY="dc528eaf83724782914e171f3bbdaeda"
AZURE_OPENAI_ENDPOINT="https://uiuc-chat-canada-east.openai.azure.com/"
AZURE_MODEL_VERSION="gpt-4-hackathon"
ROOT_DIR=os.path.join(os.getenv("REPO_DIR"), "files")

def get_human_input() -> str:
  """Placeholder for Slack/GH-Comment input from user."""
  print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
  contents = []
  while True:
    try:
      line = input()
    except EOFError:
      break
    if line == "q":
      break
    contents.append(line)
  return "\n".join(contents)

def get_tools(langsmith_run_id ):
  tools = []

  # GOOGLE SEARCH
  tools += load_tools(["serpapi"])

  # WOLFRAM ALPHA
  tools += load_tools(["wolfram-alpha"])

  # ARXIV
  tools += load_tools(["arxiv"])

  # DOCUMENT LOADER
  # file_path = "<filepath>"
  # loader = AzureAIDocumentIntelligenceLoader(
  #     api_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, file_path=file_path, api_model="prebuilt-layout"
  # )
  # tools += [loader]

  # FILE MANAGEMENT
  file_management = FileManagementToolkit(
    # If you don't provide a root_dir, operations will default to the current working directory
    root_dir=ROOT_DIR
  ).get_tools()
  tools += file_management

  # HUMAN TOOL
  llm = AzureChatOpenAI(
        temperature=0.1,
        model="gpt-4-1106-Preview",
    )
  tools += load_tools(["human"], llm=llm, input_func=get_human_input)
  return tools


if __name__ == "__main__":
  tools = get_tools()
  print(tools)
