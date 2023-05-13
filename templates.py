from langchain.callbacks.base import BaseCallbackHandler
from colorama import init as colorama_init
from colorama import Fore
import time
import utils as u

colorama_init()

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  mins = mins % 60
  return "{0}m:{1:.0f}s".format(int(mins),sec)

class Templates:
    def Wikipedia():
        return """What is the subject of this sentence: {question}?
        The subject is:"""
    
    def Search():
        return """You are an assistant that makes search using the input
        
        examples:
        xxx xx xxxx
        xx xxx xx
        
        What should be a possible search for this question: {question}?
            
        search:"""
        
    def ChatTest():
        return """Answer the following questions as best you can. You have access to the following tools:

        wikipedia, google, youtube

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [wikipedia, google, youtube]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question"""
        
    def Generate():
        return """Based on this context: 
        {result}
    
        Answer this question: {question}"""
        
    def Thinking():
        return """Based on this input: {input}, you have to take an action, it should be only one of this: {tools}
        
        Examples:
        Action: wikipedia
        Action: google
        Action: youtube
        
        Action: Choose an action"""
    
    def Chat():
        return """You are an helpful assistant, your name is {name}:
        Based on this user message: {message}
        And on this history: {history}
        Write an answer:"""
    
    def Code():
        return """You are an assistant that writes code based on user input
        This is the input: {input}
        Code:"""
        
    def Document():
        return """Generate an answer based on an input text and a question.
        The input text to search for an answer in:
            {input}
        
        The question to answer:
            {action}
        
        Answer:"""
        
"""Callback Handler streams to stdout on new llm token."""
import sys
from typing import Any, Dict, List, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

class TokenHandler(BaseCallbackHandler):
    def __init__(self, bot_name, time_start, time_end, total_time):
        self.bot_name = bot_name
        self.time_start = time_start
        self.time_end = time_end
        self.total_time = total_time
    
    """Callback handler for streaming. Only works with LLMs that support streaming."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.start_time = time.time()
        sys.stdout.write(Fore.RED + self.bot_name + ": " + Fore.WHITE)
        sys.stdout.flush()
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.time_end = time.time()
        self.total_time = self.time_end - self.start_time
        #For test only
        # print("\n" + Fore.CYAN + "Elapsed time: " + Fore.WHITE + f"{time_convert(self.total_time)}")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""