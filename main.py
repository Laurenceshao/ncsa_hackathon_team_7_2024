import gradio as gr
import json
import uuid
import logging
import traceback
from utils.tots_agent import WorkflowAgent as TOTsAgent
# from utils.agent_2 import WorkflowAgent as LATSAgent
from type.issue import Issue

# Function to handle user queries and run the workflow
def run_workflow_with_gradio(query):
    """
    Function to run the WorkflowAgent with Gradio.
    """
    langsmith_run_id = str(uuid.uuid4())
    
    try:
        # Load issue data from JSON file
        with open('issue.json') as f:
            issue_data = json.load(f)

        if not issue_data:
            raise ValueError("Missing the body of the webhook response.")

        # Create an issue instance from the loaded data
        issue = Issue.from_json(issue_data)

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
