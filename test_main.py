import gradio as gr
import json
import uuid
import logging
import traceback
from utils.agent_2 import WorkflowAgent
from type.issue import Issue

def run_workflow_with_inputs(title, body, message, file_path):
    """
    Function to create/update issue.json based on Gradio inputs and run the WorkflowAgent.
    """
    langsmith_run_id = str(uuid.uuid4())
    
    issue_data = {
        "title": title,
        "body": body,
        "labels": ["enhancement"],
        "assignees": ["your-username"],  # Change to your desired assignees
        "milestone": None,
        "state": "open",
        "number": 1,
        "created_at": "2023-04-01T00:00:00Z",
        "updated_at": "2023-04-01T00:00:00Z",
        "closed_at": None,
        "author": "your-username",
        "comments": [
            {
                "user": "commenter-username",  # Change as needed
                "message": message,
            },
        ],
    }

    # If there's a file path, add it to the issue data
    if file_path:
        issue_data["file_path"] = file_path

    with open('issue.json', 'w') as f:
        json.dump(issue_data, f)

    try:
        issue = Issue.from_json(issue_data)  # Create Issue from the new data
        bot = WorkflowAgent(langsmith_run_id=langsmith_run_id)
        
        prompt = f"Here's your latest assignment: {issue.format_issue()}"
        
        result = bot.run(prompt)
        
        return result 

    except Exception as e:
        error_message = f"Error: {e}\nTraceback:\n{traceback.format_exc()}"
        logging.error(error_message)
        return error_message

iface = gr.Interface(
    fn=run_workflow_with_inputs,
    inputs=[
        gr.Textbox(label="Title", placeholder="Enter the issue title"),
        gr.Textbox(label="Body", placeholder="Describe the issue"),
        gr.Textbox(label="Message", placeholder="Enter your comment message"),
        gr.Textbox(label="File Path", placeholder="Enter file path if needed"),
    ],
    outputs="text",
    title="Workflow Agent with Multiple Inputs",
    description="Submit title, body, message, and optional file path to run the workflow.",
)

iface.launch()
