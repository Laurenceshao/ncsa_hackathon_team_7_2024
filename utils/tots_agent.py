from __future__ import annotations

import json
import logging
import math
from collections import defaultdict, deque
from typing import List, Optional

import os
import getpass
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.chains import create_structured_output_runnable
from langchain_core.prompt_values import ChatPromptValue
from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig, chain as as_runnable
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langgraph.graph import END, StateGraph

from utils.tool import get_tools


def get_llm():
    return AzureChatOpenAI(
        azure_deployment="gpt-4-128k",
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.7,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

class Node:
    def __init__(
        self,
        messages: List[BaseMessage],
        curr_step_generation = List[BaseMessage],
        evaluation: StateEvaluation = None,
        parent: Optional[Node] = None,
    ):
        self.messages = messages
        self.curr_step_generation = curr_step_generation
        self.parent = parent
        self.children = []
        self.evaluation = evaluation
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = evaluation.found_solution if evaluation else False
        self.best_child_idx = None
    
    def __repr__(self) -> str:
        return (
            f"<Node"
            f" solution={self.messages}/>"
        )
    
    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1
    
    def get_solution(self):
        """Return the solution."""
        reached_solution = False
        curr_step = self
        while not reached_solution:
            best_idx = curr_step.best_child_idx
            curr_step = curr_step.children[best_idx]
            reached_solution = curr_step._is_solved
            print(reached_solution)

        best_idx = curr_step.best_child_idx
        solution = curr_step.children[best_idx]
        return solution

class StateEvaluation(BaseModel):
    """
    Encapsulates the evaluation and voting of a response. This includes a textual critique, a candidate ID, and a flag indicating if the solution was found.
    """

    evaluations: str = Field(
        description="The critique and evaluations on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    best_candidate_id: int = Field(
        description="ID of the best candidate response"
    )
    found_solution: bool = Field(
        description="Wheter the user request was fully addressed"
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.evaluations}\nBest Candidate ID: {self.best_candidate_id}\nIs Solved: {self.found_solution}"
        )
    
    def get_best_candidate_id(self) -> int:
        return self.best_candidate_id

class Reflection(BaseModel):
    """
    Encapsulates the reflection and scoring of a response. This includes a textual critique, a numerical score, and a flag indicating if the solution was found.
    """

    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class TreeState(TypedDict):
    """
    Defines the structure of the state used in the tree search. It includes the root node of the search tree and the initial input string.
    """

    root: Node
    curr_step: Node
    input: str

class WorkflowAgent:
    """
    Manages the workflow of generating and evaluating responses to user queries. This includes initializing the language model, setting up the search tree, and executing the search algorithm to find the best response.
    """

    def __init__(self, langsmith_run_id, max_depth):
        """
        Initializes a new instance of the WorkflowAgent class.

        :param langsmith_run_id: A unique identifier for the run, used to fetch tools and configurations.
        """
        self.llm = get_llm()
        self.tools = get_tools(langsmith_run_id)
        self.tool_executor = ToolExecutor(tools=self.tools)
        self.parser = JsonOutputToolsParser(return_id=True)
        self.max_depth = max_depth

        # self.reflection_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             "You are a world-class programmer and AI assistant capable of executing any goal related to software development, genAI, LLMs, and full-stack technologies. Reflect and grade the assistant response to the user question below.",
        #         ),
        #         ("user", "{input}"),
        #         MessagesPlaceholder(variable_name="candidate"),
        #     ]
        # )

        # self.reflection_llm_chain = (
        #     self.reflection_prompt
        #     | self.llm.bind_tools(
        #         tools=[Reflection], tool_choice="Reflection"
        #     ).with_config(run_name="Reflection")
        #     | PydanticToolsParser(tools=[Reflection])
        # )
        
        self.state_evaluation_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="initial_input"),
                MessagesPlaceholder(variable_name="candidate0"),
                MessagesPlaceholder(variable_name="candidate1"),
                MessagesPlaceholder(variable_name="candidate2"),
                (
                    "system",
                    "You are a world-class programmer and AI assistant capable of executing any goal related to software development, genAI, LLMs, and full-stack technologies."
                    "Given three candidate responses to the user question below, reflect and choose the best candidate to proceed on."
                    "Pick any candidate if you can't find the best choice out of three responses."
                    "Also, concretly check if the user request was fully addressed with the latest best candidate respnse. You should not finish the task without completing it entirely.",
                ),
                ("user", "{input}"),
            ]
        )
        self.state_evaluation_chain = (
            self.state_evaluation_prompt
            | self.llm.bind_tools(
                tools=[StateEvaluation], tool_choice="StateEvaluation"
            )
            | PydanticToolsParser(tools=[StateEvaluation])
        )

        self.initial_answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world-class programmer and AI assistant capable of executing any goal related to software development, genAI, LLMs, and full-stack technologies."
                    "The main objective is to plan and execute the workflow efficiently to complete the user request."
                    "Given the user request, generate the next step to complete the user request, execute the current step, and return the result each time. Projress with only a single step instead of the whole plan. Your step should be descriptive and well-explained."
                    "You have access to a variety of tools, including browser, wolfram for numerical computations, arxiv for scientific article access, and interaction with the user. Utilize the browser for internet searches and rely on file management tools for saving and loading the local files needed.",
                ),
                ("user", "{input}"),
            ]
            )
        self.initial_answer_chain = (
            self.initial_answer_prompt
            | self.generate_candidates
            # | self.llm.bind_tools(tools=self.tools).with_config(
            #     run_name="GenerateInitialCandidate"
            # )
        )
        
        self.expansion_prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder(variable_name="messages"),
                    ("system", 
                        "You are a world-class programmer and AI assistant capable of executing any goal related to software development, genAI, LLMs, and full-stack technologies."
                        "The main objective is to plan and execute the workflow efficiently to complete the user request."
                        "Given the user request and chat history, generate the next step to proceed on the latest best candidate response, execute the next step, and return the result. Projress with only a single step instead of the whole plan. Your step should be descriptive and well-explained."
                        "You have access to a variety of tools, including browser, wolfram for numerical computations, arxiv for scientific article access, and interaction with the user. Utilize the browser for internet searches and rely on file management tools for saving and loading the local files needed.",),
                    ("user", "{input}")
                ]
        )

        #summary prompting
        self.summarization_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You're an assistant that summarizes data in a clear, easy-to-understand format."),
                ("user", "Generate a natural language summary from the given messages."),
                MessagesPlaceholder(variable_name="final_trajectory"),
                MessagesPlaceholder(variable_name="final_response")
            ]
        )

<<<<<<< Updated upstream


=======
>>>>>>> Stashed changes
        self.expansion_chain = (
            self.expansion_prompt
            | self.generate_candidates
        )

        self.graph = self.create_graph()

    def generate_candidates(self, messages: ChatPromptValue, config: RunnableConfig):
        n = config["configurable"].get("N", 3)
        bound_kwargs = self.llm.bind_tools(tools=self.tools).kwargs
        chat_result = self.llm.generate(
            [messages.to_messages()],
            n=n,
            callbacks=config["callbacks"],
            run_name="GenerateCandidates",
            **bound_kwargs,
        )
        return [gen.message for gen in chat_result.generations[0]]

    def create_graph(self):
        builder = StateGraph(TreeState)

        def generate_initial_response(state: TreeState) -> TreeState:
            # Thoughts Generation
            # logging.info(f"Generating initial response for: {state['input']}")
            print(f"Generating initial response for: {state['input']}")

            initial_answer_prompt = self.initial_answer_prompt.invoke({"input": state["input"]})
            new_candidates = self.initial_answer_chain.invoke({"input": state["input"]})
            parsed = self.parser.batch(new_candidates)
            flattened = [
                (i, tool_call)
                for i, tool_calls in enumerate(parsed)
                for tool_call in tool_calls
            ]
            tool_responses = self.tool_executor.batch(
                [
                    ToolInvocation(tool=tool_call["type"], tool_input=tool_call["args"])
                    if "__arg1" not in tool_call["args"] else
                    ToolInvocation(tool=tool_call["type"], tool_input=tool_call["args"]["__arg1"])
                    for _, tool_call in flattened
                ]
            )
            collected_responses = defaultdict(list)
            for (i, tool_call), resp in zip(flattened, tool_responses):
                collected_responses[i].append(
                    ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
                )
            
            for i, candidate in enumerate(new_candidates):
                candidate.content = f"Candidate {i}: " + candidate.content
            
            output_messages = [
                [candidate] + collected_responses[i]
                for i, candidate in enumerate(new_candidates)
            ]

            state_evaluation_input = {"initial_input": initial_answer_prompt.messages, "input": state["input"], "candidate0": output_messages[0], "candidate1": output_messages[1], "candidate2": output_messages[2]}
            state_evaluation_prompt = self.state_evaluation_prompt.invoke(state_evaluation_input)
            state_evaluation = self.state_evaluation_chain.invoke(state_evaluation_input)[0]
            print(state_evaluation.found_solution)

            # Add Root Node
            root = Node([], evaluation=state_evaluation)
            root.best_child_idx = root.evaluation.get_best_candidate_id()

            messages = state_evaluation_prompt.messages
            messages.append(state_evaluation.as_message())
            # Add Children Nodes
            child_nodes = [
                Node(messages, curr_step_generation=output_messages[root.best_child_idx] ,parent=root)
                for _ in output_messages
            ]
            root.children.extend(child_nodes)
    
            curr_step = root.children[root.best_child_idx]
            return {
                "root" : root,
                "curr_step": curr_step,
                "input": state["input"]
            }

        def expand(state: TreeState, config: RunnableConfig) -> TreeState:
            best_candidate =  state["curr_step"]
            chat_history = best_candidate.messages
            expansion_prompt = self.expansion_prompt.invoke({"input": state["input"], "messages": chat_history})
            new_candidates = self.expansion_chain.invoke(
                {"input": state["input"], "messages": chat_history}, config
            )
            parsed = self.parser.batch(new_candidates)
            flattened = [
                (i, tool_call)
                for i, tool_calls in enumerate(parsed)
                for tool_call in tool_calls
            ]
            tool_responses = self.tool_executor.batch(
                [
                    ToolInvocation(tool=tool_call["type"], tool_input=tool_call["args"])
                    if "__arg1" not in tool_call["args"] else
                    ToolInvocation(tool=tool_call["type"], tool_input=tool_call["args"]["__arg1"])
                    for _, tool_call in flattened
                ]
            )
            collected_responses = defaultdict(list)
            for (i, tool_call), resp in zip(flattened, tool_responses):
                collected_responses[i].append(
                    ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
                )
            output_messages = [
                [candidate] + collected_responses[i]
                for i, candidate in enumerate(new_candidates)
            ]

            state_evaluation_input = {"initial_input": expansion_prompt.messages, "input": state["input"], "candidate0": output_messages[0], "candidate1": output_messages[1], "candidate2": output_messages[2]}
            state_evaluation_prompt = self.state_evaluation_prompt.invoke(state_evaluation_input)
            state_evaluation = self.state_evaluation_chain.invoke(state_evaluation_input)[0]
            print(state_evaluation.found_solution)

            best_candidate.evaluation = state_evaluation
            best_candidate.best_child_idx = best_candidate.evaluation.get_best_candidate_id()

            messages = state_evaluation_prompt.messages
            messages.append(state_evaluation.as_message())
            # Add Children Nodes
            child_nodes = [
                Node(messages, curr_step_generation=output_messages[best_candidate.best_child_idx], parent=best_candidate)
                for _ in output_messages
            ]
    
            best_candidate.children.extend(child_nodes)
            curr_step = best_candidate.children[best_candidate.best_child_idx]

            state["curr_step"] = curr_step
            return state

        def should_loop(state: TreeState):
            curr_step = state["curr_step"]
            if curr_step.parent.is_solved:
                print("Solved!")
                return "__end__"
            if curr_step.height > self.max_depth:
                print("Reached max depth!")
                return "__end__"
            print("Expanding...")
            return "expand"

        builder.add_node("start", generate_initial_response)
        builder.add_node("expand", expand)
        builder.set_entry_point("start")
        builder.add_conditional_edges(
            "start",
            should_loop,
        )
        builder.add_conditional_edges(
            "expand",
            should_loop,
        )
        return builder.compile()
       
    #summary
    def generate_summary(self, trajectory: List[BaseMessage], final_response: List[BaseMessage]) -> str:
        prompt = self.summarization_prompt.invoke(
            {"final_trajectory": trajectory, "final_response": final_response}
        )
        summary_result = self.llm.generate([prompt.to_messages()])

<<<<<<< Updated upstream
=======
        #summary
    def generate_summary(self, trajectory: List[BaseMessage], final_response: List[BaseMessage]) -> str:
        prompt = self.summarization_prompt.invoke(
            {"final_trajectory": trajectory, "final_response": final_response}
        )
        summary_result = self.llm.generate([prompt.to_messages()])

>>>>>>> Stashed changes
        # print(dir(summary_result.generations[0]))
        print(summary_result.generations[0][0])
        print(summary_result.generations[0][0].to_json())
        
        summary_result = summary_result.generations[0][0].to_json()

        if 'kwargs' in summary_result and 'message' in summary_result['kwargs']:
            ai_message = summary_result['kwargs']['message']  # Get the AIMessage object
            if hasattr(ai_message, 'content'):  # Check if content is accessible
                message_content = ai_message.content  # Extract the content
                return message_content  # Output the extracted message content
        else:
            return "Error: 'message' key not found in 'kwargs'"
    
    def run(self, question: str) -> BaseMessage:
        """
        Executes the workflow to generate and evaluate responses to a given question.

        :param question: The user's question to respond to.
        :return: The best response message after evaluating various candidates.
        """
        state = TreeState(input=question)
        for step in self.graph.stream(state):
            step_name, step_state = next(iter(step.items()))
            # print(f"Step Name: {step_name}, Step State: {step_state}")
            # print(f"Depth: {step_name.depth}")
        solution_node = step["__end__"]["curr_step"]
        full_trajectory = solution_node.messages
        final_response = solution_node.curr_step_generation

        summary = self.generate_summary(full_trajectory, final_response) # added

        return summary
<<<<<<< Updated upstream

        # return full_trajectory, final_response
    



=======
>>>>>>> Stashed changes
