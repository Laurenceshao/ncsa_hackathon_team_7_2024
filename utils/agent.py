import getpass
import os
import platform
from typing import List, Tuple, Union, Annotated, TypedDict
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain_openai import AzureChatOpenAI
from langchain.chains.openai_functions import create_structured_output_runnable, create_openai_fn_runnable
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_agent_executor
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, BaseMessage
import operator

from utils.tool import get_tools

load_dotenv(override=True)
AZURE_OPENAI_API_VERSION=os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_API_KEY=os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT=os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_MODEL_VERSION=os.getenv('AZURE_MODEL_VERSION')


class Plan(BaseModel):
  """Plan to follow in future to complete the objective"""
  steps: List[str] = Field(description="Steps to follow in sorted order of execution.")


class Response(BaseModel):
  """Objective complete (or impossible), final response to user."""
  response: str


class State(TypedDict):
  input: str
  chat_history: list[BaseMessage]
  plan: List[str]
  past_steps: Annotated[List[Tuple], operator.add]
  response: str


def get_user_info_string():
  username = getpass.getuser()
  current_working_directory = os.getcwd()
  operating_system = platform.system()
  default_shell = os.environ.get("SHELL")

  return f"[User Info]\nName: {username}\nCWD: {current_working_directory}\nSHELL: {default_shell}\nOS: {operating_system}"


def get_llm():
    return AzureChatOpenAI(
        azure_deployment="gpt-4-128k",
        openai_api_version=os.getenv("AZURE_MODEL_VERSION"),
        temperature=0,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


class WorkflowAgent:

  @classmethod
  async def create(cls, langsmith_run_id):
    self = cls()
    print("Langgraph v2 agent initialized")
    self.langsmith_run_id = langsmith_run_id
    self.llm = get_llm()
    self.tools = await get_tools(langsmith_run_id)
    self.planner_prompt = ChatPromptTemplate.from_template(
        """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

{objective}""")
    self.replanner_prompt = ChatPromptTemplate.from_template(
        """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
    )

    self.executor_prompt = ChatPromptTemplate.from_template(
          """You are a world-class programmer and AI assistant capable of executing any goal related to software development, genAI, LLMs, and full-stack technologies.\
           For the given task, execute the task and return the result.\
           When you send a message containing code, it will be executed in a Docker container. You have been granted full permission to execute any code necessary to complete the task within this Docker environment using PythonRepl and shell tools as required.\
           You have access to a variety of tools, including browser, github_tools for interacting with GitHub, and multiple vectorstore instances. Utilize the browser for internet searches and github_tools for all interactions with GitHub repositories. For code execution, rely on PythonRepl and shell tools available in the Docker environment.\
           Before any execution task, prepare the development environment, whether that be a notebook, .sh, .py, .ipynb, .R, or other file types. Incrementally develop, execute, and debug the code, committing changes to GitHub regularly.\
           [User Info]: {user_info}\
           [Chat history]: {chat_history}\
           [Input]: {input}\
           [Agent scratchpad]: {agent_scratchpad}\
           """
    )
    # hub.pull("hwchase17/openai-functions-agent")

    self.agent_runnable = create_openai_functions_agent(self.llm, self.tools,
                                                        hub.pull("hwchase17/openai-functions-agent"))
    self.agent_executor = create_agent_executor(self.agent_runnable, self.tools)
    self.workflow = self.create_workflow()
    return self

  def create_workflow(self):
    workflow = StateGraph(State)

    async def execute_step(state: State):
      task = state["plan"][0]
      agent_response = await self.agent_executor.ainvoke({"input": task, "chat_history": []})
      return {"past_steps": (task, agent_response["agent_outcome"].return_values["output"])}

    async def plan_step(state: State):
      planner = create_structured_output_runnable(Plan, self.llm, self.planner_prompt)
      plan = await planner.ainvoke({"objective": state["input"]})
      return {"plan": plan.steps}

    async def replan_step(state: State):
      replanner = create_openai_fn_runnable([Plan, Response], self.llm, self.replanner_prompt)
      output = await replanner.ainvoke(state)
      if isinstance(output, Response):
        return {"response": output.response}
      else:
        return {"plan": output.steps}

    def should_end(state: State):
      if state["response"]:
        return True
      else:
        return False

    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges("replan", should_end, {True: END, False: "agent"})  #type: ignore

    return workflow.compile().with_config({"recursion_limit": 100})

  async def run(self, input_prompt):
    inputs = {"input": input_prompt}
    async for event in self.workflow.astream(inputs, config={"recursion_limit": 50}):
      for k, v in event.items():
        if k != "__end__":
          print(v)


# Example usage
# agent = WorkflowAgent()
# await agent.run("what is
