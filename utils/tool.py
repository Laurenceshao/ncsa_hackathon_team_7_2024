import os
from typing import List
import uuid
import langchain
from langchain.agents import load_tools
from langchain.agents.agent_toolkits.file_management.toolkit import FileManagementToolkit
from langchain_community.tools import ShellTool
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit

ROOT_DIR = os.path.join(os.getenv("REPO_DIR"), "files")

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

def get_tools(langsmith_run_id):
  tools = []

  # GOOGLE SEARCH
  tools += load_tools(["serpapi"])

  # WOLFRAM ALPHA
  tools += load_tools(["wolfram-alpha"])

  # ARXIV
  tools += load_tools(["arxiv"])

  # # DOCUMENT LOADER
  # file_path = "<filepath>"
  # loader = AzureAIDocumentIntelligenceLoader(
  #     api_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'), 
  #     api_key=os.getenv('AZURE_OPENAI_API_KEY'), 
  #     file_path=file_path, 
  #     api_model="prebuilt-layout"
  # )
  # tools += [loader]

  

  # FILE MANAGEMENT
  file_management = FileManagementToolkit(root_dir=ROOT_DIR).get_tools()
  tools += file_management

  # SHELL TOOL
  shell_tool = ShellTool()
  tools += [shell_tool]

  # HUMAN TOOL
  llm = AzureChatOpenAI(
        temperature=0.1,
        model="gpt-4-1106-Preview",
    )
  tools += load_tools(["human"], llm=llm, input_func=get_human_input)
  return tools

if __name__ == "__main__":
  langsmith_run_id = str(uuid.uuid4())
  tools = get_tools(langsmith_run_id)
  for tool in tools:
    print(tool.args_schema)
  print(tools)