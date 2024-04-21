import inspect
import json
import asyncio
import logging
import traceback
import uuid

import ray
from utils.agent_2 import WorkflowAgent
from type.issue import Issue
from langchain import hub

# import nest_asyncio

# nest_asyncio.apply()

def main():
  """
  Executes the main workflow of the application.

  This function initializes the application, loads issue data from a JSON file, and processes it using a WorkflowAgent. If the issue data is valid, it runs the workflow to handle the issue. In case of any exceptions, it logs the error along with a traceback.

  The function also generates a unique run ID for each execution to track the workflow process. It ensures that the application gracefully handles errors and provides detailed logs for debugging purposes.

  Returns:
      tuple: An empty string and HTTP status code 200, indicating successful execution.
  """
    
  langsmith_run_id = str(uuid.uuid4())
  
  try:
    with open('issue.json') as f:
      issue_data = json.load(f)

    if issue_data:
      issue: Issue = Issue.from_json(issue_data)
    
    if not issue:
      raise ValueError(f"Missing the body of the webhook response. Response is {issue}")

    print("ABOUT TO CALL WORKFLOW AGENT on COMMENT OPENED")

    bot = WorkflowAgent(langsmith_run_id=langsmith_run_id)
    
    run_workflow(bot, issue)
  except Exception as e:
    logging.error(f"❌❌ Error in {inspect.currentframe().f_code.co_name}: {e}\nTraceback:\n", traceback.print_exc())
    err_str = f"Error in {inspect.currentframe().f_code.co_name}: {e}" + "\nTraceback\n```\n" + str(
        traceback.format_exc()) + "\n```"
    
    print(err_str)

  return '', 200

def run_workflow(bot: WorkflowAgent, issue: Issue):

  # Create final prompt for user
  prompt = f"""Here's your latest assignment: {issue.format_issue()}"""

  # RUN BOT
  result = bot.run(prompt)

  # FIN: Conclusion & results comment
  logging.info(f"✅✅ Successfully completed the issue: {issue}")
  logging.info(f"Output: {result}")

if __name__ == '__main__':
  bot = main()
