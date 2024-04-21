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


class Node:
    """
    Represents a node in the search tree. Each node contains messages, a reflection on those messages, and links to parent and child nodes.
    Nodes are used to track the state of the search, including the depth of the search, whether a solution has been found, and the score of the solution.
    """

    def __init__(
        self,
        messages: List[BaseMessage],
        reflection: Reflection,
        parent: Optional[Node] = None,
    ):
        """
        Initializes a new instance of the Node class.

        :param messages: A list of messages associated with this node.
        :param reflection: A reflection object containing feedback on the messages.
        :param parent: An optional reference to the parent node in the search tree.
        """
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages} reflection={self.reflection}/>"
        )

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child(self):
        """Select the child with the highest UCT to search next."""
        if not self.children:
            return None
        all_nodes = self._get_all_children()
        return max(all_nodes, key=lambda child: child.upper_confidence_bound())

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> List[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent


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
    input: str


def get_llm():
    print(repr(os.getenv("AZURE_OPENAI_ENDPOINT")))
    return AzureChatOpenAI(
        azure_deployment="gpt-4-128k",
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
        # azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_endpoint='https://uiuc-chat-canada-east.openai.azure.com/',
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )


class WorkflowAgent:
    """
    Manages the workflow of generating and evaluating responses to user queries. This includes initializing the language model, setting up the search tree, and executing the search algorithm to find the best response.
    """

    def __init__(self, langsmith_run_id):
        """
        Initializes a new instance of the WorkflowAgent class.

        :param langsmith_run_id: A unique identifier for the run, used to fetch tools and configurations.
        """
        self.llm = get_llm()
        self.tools = get_tools(langsmith_run_id)
        self.tool_executor = ToolExecutor(tools=self.tools)

        self.reflection_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a world-class programmer and AI assistant capable of executing any goal related to software development, genAI, LLMs, and full-stack technologies. Reflect and grade the assistant response to the user question below.",
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="candidate"),
            ]
        )
        self.reflection_llm_chain = (
            self.reflection_prompt
            | self.llm.bind_tools(
                tools=[Reflection], tool_choice="Reflection"
            ).with_config(run_name="Reflection")
            | PydanticToolsParser(tools=[Reflection])
        )

        self.initial_answer_chain = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a world-class programmer and AI assistant capable of executing any goal related to software development, genAI, LLMs, and full-stack technologies.

First, write a step-by-step plan for the task. The plan should be descriptive and well-explained. 

The main objective is to plan and execute the workflow efficiently. Break down the execution into small, informed steps rather than attempting everything in one go.

You have access to a variety of tools, including browser, github_tools for interacting with GitHub, and multiple vectorstore instances. Utilize the browser for internet searches and github_tools for all interactions with GitHub repositories. For code execution, rely onand shell tools available in the Docker environment to create and execute/test files.

Use shell and file management tools to always execute the code and iterate on the plan based on the output.""",
                    ),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="messages", optional=True),
                ]
            )
            | self.llm.bind_tools(tools=self.tools).with_config(
                run_name="GenerateInitialCandidate"
            )
        )
        self.parser = JsonOutputToolsParser(return_id=True)

        self.expansion_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", "You are an AI assistant."),
                    ("user", "{input}"),
                    MessagesPlaceholder(variable_name="messages", optional=True),
                ]
            )
            | self.generate_candidates
        )

        self.graph = self.create_graph()

    def generate_candidates(self, messages: ChatPromptValue, config: RunnableConfig):
        n = config["configurable"].get("N", 5)
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
        
        @as_runnable
        def reflection_chain(inputs) -> Reflection:
            logging.info(f"Reflection inputs in reflection chain: {inputs}")
            tool_choices = self.reflection_llm_chain.invoke(inputs)
            reflection = tool_choices[0]
            if not isinstance(inputs["candidate"][-1], AIMessage):
              reflection.found_solution = False
            return reflection

        def generate_initial_response(state: TreeState) -> dict:
            logging.info(f"Generating initial response for: {state['input']}")
            print(f"Generating initial response for: {state['input']}")
            res = self.initial_answer_chain.invoke({"input": state["input"]})
            parsed = self.parser.invoke(res)
            tool_responses = self.tool_executor.batch(
                [ToolInvocation(tool=r["type"], tool_input=r["args"]) for r in parsed]
            )
            output_messages = [res] + [
                ToolMessage(content=json.dumps(resp), tool_call_id=tool_call["id"])
                for resp, tool_call in zip(tool_responses, parsed)
            ]
            # print(f"Reflection inputs: {inputs}")
            reflection = reflection_chain.invoke({"input": state["input"], "candidate": output_messages})
            root = Node(output_messages, reflection=reflection)
            print(f"Initial response generated: {root}")
            return {
                **state,
                "root": root,
            }

        def expand(state: TreeState, config: RunnableConfig) -> TreeState:
            root = state["root"]
            best_candidate: Node = root.best_child if root.children else root
            messages = best_candidate.get_trajectory()
            new_candidates = self.expansion_chain.invoke(
                {"input": state["input"], "messages": messages}, config
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
            reflections = reflection_chain.batch(
                [
                    {"input": state["input"], "candidate": msges}
                    for msges in output_messages
                ],
                config,
            )
            child_nodes = [
                Node(cand, parent=best_candidate, reflection=reflection)
                for cand, reflection in zip(output_messages, reflections)
            ]
            best_candidate.children.extend(child_nodes)
            print(f"Expanded node: {best_candidate}")
            return state

        def should_loop(state: TreeState):
            root = state["root"]
            if root.is_solved:
                print("Solved!")
                return "__end__"
            if root.height > 5:
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

    def run(self, question: str) -> BaseMessage:
        """
        Executes the workflow to generate and evaluate responses to a given question.

        :param question: The user's question to respond to.
        :return: The best response message after evaluating various candidates.
        """
        state = TreeState(input=question)
        for step in self.graph.stream(state):
            step_name, step_state = next(iter(step.items()))
            print(f"Step Name: {step_name}, Step State: {step_state}")
        solution_node = step["__end__"]["root"].get_best_solution()
        best_trajectory = solution_node.get_trajectory(include_reflections=False)
        return best_trajectory[-1]
