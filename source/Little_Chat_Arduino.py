"""
Little_Chat_Arduino is a simple AI Agent chatbot that lets you
chat with your LLM but also offers an Arduino agent.
With the Arduino agent you can ask your LLM to control your
Arduino device and do things in the real world.
The Arduino agent builds your sketch and runs it, and lets your
LLM know if something went wrong — e.g. device not connected or
device errors.

Supports triple provider mode:
  - Local Ollama LLM   (default, no API key needed)
  - Network Ollama LLM (LAN server, e.g. 192.168.1.58:11434)
  - Anthropic Claude   (requires --api-key)
"""
VERSION = "0.1.0 triple-provider - text/graph option" 


import requests
import json
import warnings
import os
import argparse
from dotenv import load_dotenv

# --- Pydantic ---
from pydantic import Field

# --- LangChain Core & Agent Imports ---
from langchain_core.tools import BaseTool

# --- LangGraph for Agent ---
from langgraph.prebuilt import create_react_agent

# --- Specific Integration Packages ---
from langchain_ollama import ChatOllama

import gradio as gr

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# =================================================================
# CONSTANTS
# =================================================================

SERVER_URL = "http://127.0.0.1:8000"

# Default models
DEFAULT_OLLAMA_MODEL    = "qwen3:4b"
DEFAULT_CLAUDE_MODEL    = "claude-sonnet-4-5"   # great balance of speed & quality

# Default network Ollama server
DEFAULT_OLLAMA_NET_HOST = "192.168.1.58"  # <<== WARNING: SET THIS ADDRESS TO YOUR NETWORK
DEFAULT_OLLAMA_NET_PORT = 11434


# =================================================================
# LLM FACTORY
# =================================================================

def get_llm(
    provider: str,
    api_key: str = None,
    model: str = None,
    temperature: float = 0.1,
    ollama_host: str = None,
    ollama_port: int = None,
):
    """
    Return a LangChain chat model based on the chosen provider.

    Providers
    ---------
    "ollama"     — local Ollama on 127.0.0.1 (no key needed)
    "ollama-net" — Ollama on a LAN server, e.g. 192.168.1.58:11434
    "anthropic"  — Anthropic Claude (requires api_key)
    """
    if provider == "anthropic":
        if not api_key:
            raise ValueError(
                "An Anthropic API key is required when using provider='anthropic'.\n"
                "Pass it with --api-key sk-ant-..."
            )
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is not installed.\n"
                "Run:  pip install langchain-anthropic"
            )
        resolved_model = model or DEFAULT_CLAUDE_MODEL
        print(f"[LLM Factory] Using Anthropic Claude — model: {resolved_model}")
        return ChatAnthropic(
            model=resolved_model,
            temperature=temperature,
            api_key=api_key,
        )

    elif provider == "ollama-net":
        host = ollama_host or DEFAULT_OLLAMA_NET_HOST
        port = ollama_port or DEFAULT_OLLAMA_NET_PORT
        base_url = f"http://{host}:{port}"
        resolved_model = model or DEFAULT_OLLAMA_MODEL
        print(f"[LLM Factory] Using network Ollama @ {base_url} — model: {resolved_model}")
        return ChatOllama(
            model=resolved_model,
            temperature=temperature,
            base_url=base_url,
        )

    else:  # default: local ollama
        resolved_model = model or DEFAULT_OLLAMA_MODEL
        print(f"[LLM Factory] Using local Ollama — model: {resolved_model}")
        return ChatOllama(model=resolved_model, temperature=temperature)


# =================================================================
# FastMCPTool
# =================================================================

class FastMCPTool(BaseTool):
    """A LangChain tool that calls the MCP server API."""
    name: str = Field()
    description: str = Field()
    function_name: str = Field()

    def _run(self, query: str) -> str:
        try:
            endpoint_url = f"{SERVER_URL}/{self.function_name}"
            params = {"myParam": query.strip()}
            response = requests.get(endpoint_url, params=params)
            response.raise_for_status()
            return json.dumps(response.json())
        except requests.exceptions.RequestException as e:
            return f"Network error calling function {self.function_name}: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"


# =================================================================
# Main Client Application
# =================================================================

class FastMCPLangChainClient:
    def __init__(
        self,
        provider: str = "ollama",
        api_key: str = None,
        model: str = None,
        show_thinking: bool = False,
        ollama_host: str = None,
        ollama_port: int = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.show_thinking = show_thinking
        self.agent_executor = None
        self.chat_history = []

        # Build LLM instance
        self.llm = get_llm(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=0.1,
            ollama_host=ollama_host,
            ollama_port=ollama_port,
        )

    def initialize(self):
        """Initialize the LangChain agent with MCP tools."""

        Arduino_config_file = "Arduino_configuration.txt"
        try:
            with open(Arduino_config_file, "r") as f:
                Arduino_config = f.read()
        except FileNotFoundError:
            Arduino_config = "Arduino UNO R3"

        mcp_tools_config = [
            {
                "name": "get_datetime",
                "description": "Use this tool to find the current date and time for any city. Input args should be the city name, like 'Paris' or 'Tokyo, Japan'.",
                "function_name": "get_datetime",
            },
            {
                "name": "get_weather",
                "description": "Use this tool to get the current weather for a city. Input args should be the city name, like 'London, UK'.",
                "function_name": "get_weather",
            },
            {
                "name": "get_calc",
                "description": "Use this tool to get the result of arithmetic operations. Input should be OPERATION, NUM-ONE, NUM-TWO.\
                    Allowed operations are : ADD, SUB, MUL, DIV",
                "function_name": "get_calc",
            },
            {
                "name": "run_Arduino",
                "description": """Compile and running Aruino sketch."
    Use this tool whenever the user asks to running anything you can do using Arduino device.
    You have ready to go an: """ + Arduino_config +  """
    Write your Arduino source-code, then call this tool using this syntax: Args: {'query': '... here your source-code.....'}
    This tool will create for you a sketch, will compile it and run it .
    Example:
void setup() {
  // Initialize serial communication at 9600 bits per second
  Serial.begin(9600);
}
void loop() {
  // Print a message to the command line
  Serial.println("Hello from your Elegoo Uno R3!");
  delay(1000); // Wait for 1 second
}""",
                "function_name": "run_Arduino",
            },
        ]

        langchain_tools = [FastMCPTool(**config) for config in mcp_tools_config]

        self.agent_executor = create_react_agent(self.llm, langchain_tools)

        print("\nFastMCP LangChain Client initialized successfully!")
        print("Tools available:", [tool.name for tool in langchain_tools])

    def chat(self, message: str) -> str:
        if not self.agent_executor:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        try:
            messages = list(self.chat_history)
            messages.append({"role": "user", "content": message})

            final_response = ""

            # --- THINKING MODE (stream) ---
            if self.show_thinking:
                print("\n" + "─" * 30 + " THINKING PROCESS " + "─" * 30)

                for event in self.agent_executor.stream({"messages": messages}, stream_mode="values"):
                    current_message = event["messages"][-1]

                    if hasattr(current_message, "tool_calls") and current_message.tool_calls:
                        for tool in current_message.tool_calls:
                            print(f"\nThought: I need to use tool '{tool['name']}'")
                            print(f"   Args: {tool['args']}")

                    elif current_message.type == "tool":
                        preview = (
                            current_message.content[:200] + "..."
                            if len(current_message.content) > 200
                            else current_message.content
                        )
                        print(f"\nObservation ({current_message.name}):")
                        print(f"   {preview}")

                    elif current_message.type == "ai" and not current_message.tool_calls:
                        final_response = current_message.content

                print("─" * 80 + "\n")

            # --- SILENT MODE (invoke) ---
            else:
                result = self.agent_executor.invoke({"messages": messages})
                final_response = result["messages"][-1].content

            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": final_response})

            return final_response

        except Exception as e:
            return f"Error processing message: {str(e)}"


# =================================================================
# CLI Entry Point
# =================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Little_Chat_Arduino — triple provider (Ollama / Ollama-Net / Claude)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["graph", "text"],
        help="Interface mode:\n  graph — Gradio web UI\n  text  — terminal chat",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "ollama-net", "anthropic"],
        default="ollama",
        help=(
            "LLM provider to use:\n"
            "  ollama     — local Ollama model      (default, no key needed)\n"
            "  ollama-net — Ollama on a LAN server  (use with --ollama-host / --ollama-port)\n"
            "  anthropic  — Anthropic Claude         (requires --api-key)\n"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for the chosen provider (or set ANTHROPIC_API_KEY env var).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Model name override.\n"
            f"  Ollama default    : {DEFAULT_OLLAMA_MODEL}\n"
            f"  Anthropic default : {DEFAULT_CLAUDE_MODEL}\n"
        ),
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help=(
            f"Hostname or IP of the network Ollama server (ollama-net only).\n"
            f"Default: {DEFAULT_OLLAMA_NET_HOST}"
        ),
    )
    parser.add_argument(
        "--ollama-port",
        type=int,
        default=None,
        help=(
            f"Port of the network Ollama server (ollama-net only).\n"
            f"Default: {DEFAULT_OLLAMA_NET_PORT}"
        ),
    )
    parser.add_argument(
        "--think",
        action="store_true",
        default=False,
        help="Show the agent's thinking / tool-use process (streaming mode).",
    )

    return parser.parse_args()


def clean_agent_response(agent_output_string):
    """Strip any <think>...</think> reasoning block some models emit, keeping only the final answer."""
    if "</think>" in agent_output_string:
        clean_output = agent_output_string.split("</think>", 1)[1]
    else:
        clean_output = agent_output_string
    return clean_output.strip()


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")

    # Resolve display model name for the banner
    if args.provider == "anthropic":
        display_model = args.model or DEFAULT_CLAUDE_MODEL
    else:
        display_model = args.model or DEFAULT_OLLAMA_MODEL

    # Resolve display server address for the banner
    if args.provider == "ollama-net":
        host = args.ollama_host or DEFAULT_OLLAMA_NET_HOST
        port = args.ollama_port or DEFAULT_OLLAMA_NET_PORT
        display_server = f"{host}:{port}"
    else:
        display_server = "127.0.0.1:11434"

    client = FastMCPLangChainClient(
        provider=args.provider,
        api_key=api_key,
        model=args.model,
        show_thinking=args.think,
        ollama_host=args.ollama_host,
        ollama_port=args.ollama_port,
    )

    try:
        client.initialize()

        print(f"\n{'=' * 55}")
        print(f"  Little_Chat_Arduino  —  v{VERSION}")
        print(f"  Provider : {args.provider.upper()}")
        print(f"  Server   : {display_server}")
        print(f"  Model    : {display_model}")
        print(f"  Mode     : {'THINKING' if args.think else 'SILENT'}")
        print(f"{'=' * 55}")
        print("Example questions:")
        print("What's the weather and time in Sydney now?")
        print("Print loop message \"Hello my Friend\" to your Arduino serial port")
        print("Print loop message \"hello world\" and progressive counter to your Arduino LCD module")
        print("Type 'quit' to exit.\n")
        print("Your Assistant is Ready!\n")

        if args.mode == "graph":
            def chat_with_agent(message, history):
                try:
                    response = client.chat(message)
                    return clean_agent_response(response)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return "Sorry, I encountered an error while processing your request."

            demo = gr.ChatInterface(
                fn=chat_with_agent,
                title="Little_Chat_Arduino",
                description="If you want to modify your Arduino configuration, please edit Arduino_configuration.txt",
                examples=[
                    "What is the weather in London, UK?",
                    "Calculate 15 * 3 + 7",
                    "Print loop message \"Hello my Friend\" to your Arduino serial port",
                    "Print loop message \"hello world\" and progressive counter to your Arduino LCD module"
                ],
                textbox=gr.Textbox(placeholder="Ask your agent a question...", scale=7),
            )
            demo.launch(share=False, debug=True)

        elif args.mode == "text":
            print("Type your question and press Enter. Type 'quit' or 'exit' to end.")
            print("-" * 50)
            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() in ["quit", "exit", "bye"]:
                        print("Goodbye!")
                        break
                    if not user_input:
                        continue
                    print("\nAssistant: ", end="", flush=True)
                    response = client.chat(user_input)
                    print(clean_agent_response(response))
                    print("-" * 50)
                except (KeyboardInterrupt, EOFError):
                    print("\n\nExiting chat. Goodbye!")
                    break

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
