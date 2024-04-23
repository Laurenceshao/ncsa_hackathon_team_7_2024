import gradio as gr
import json
import uuid
import logging
import traceback
from utils.agent_2 import WorkflowAgent
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

        # Create the WorkflowAgent and run the workflow
        bot = WorkflowAgent(langsmith_run_id=langsmith_run_id)
        prompt = f"Here's your latest assignment: {issue.format_issue()}"
        result = bot.run(prompt)

        return result  # Return the result from the workflow

    except Exception as e:
        error_message = f"Error: {e}\nTraceback:\n{traceback.format_exc()}"
        logging.error(error_message)
        return error_message  # Return the error message

# Create the Gradio interface
iface = gr.Interface(
    fn=run_workflow_with_gradio,  # Function to call when user submits a query
    inputs="text",  # Textbox for user input
    outputs="text",  # Textbox for output response
    title="Workflow Agent with Gradio",  # Title for the Gradio interface
    description="Submit a query to run the workflow and get the result.",  # Description
)

# Launch the Gradio interface
iface.launch()
