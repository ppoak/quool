from openai import OpenAI
from quool import setup_logger
from rich.console import Console
from rich.markdown import Markdown


class Terminal:

    def __init__(
        self,
        model: str = None,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        prompt: str = "",
        log_file: str = ".quool_history",
        log_level: str = "INFO"
    ):
        self.console = Console()
        self.setup_client(base_url, api_key)
        self.setup_model(model)
        self.setup_prompt(prompt)
        self._lino = 1
        self.logger = setup_logger("App", stream=False, level=log_level, file=log_file, clear=True)
        self.console.clear()
        self.console.print(Markdown(
            "# Quool\n\n"
            "Welcom to Quool Terminal\n\n"
            "Try to talk ask anything you want to know\n\n"
        ))
        
    def setup_client(self, base_url: str, api_key: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
    
    def setup_model(self, model: str):
        available = sorted([model.id for model in self.client.models.list()])
        if model and model not in available:
            raise ValueError(f"Model {model} not available. Available models: {available}")
        self.model = model or available[0]
    
    def setup_prompt(self, prompt: str):
        self.prompt = prompt
        self._messages = [{"role": "system", "content": prompt}]

    def process_input(self, user_input: str):
        if user_input.startswith("%"):
            self.process_command(user_input)
        else:
            self.process_chat(user_input)
    
    def process_command(self, user_input: str):
        commands = user_input[1:].split(" ")
        command = getattr(self, f"command_{commands[0]}", None)
        if command:
            command(*commands[1:])
        else:
            self.console.print(f"Unknown command: {commands[0]}")
    
    def command_show(self, name: str):
        self.console.print(Markdown(f"**{name}**\n\n{getattr(self, name)}"))

    def process_chat(self, user_input: str):
        self._messages.append({"role": "user", "content": user_input})
        self.logger.info(f"Q[{self._lino}]: {user_input}")
        response = self.client.chat.completions.create(
            messages=self._messages,
            model=self.model,
            stream=True,
        )
        answer = ""
        for chunck in response:
            content = chunck.choices[0].delta.content
            answer += content
            self.console.print(content, end="")
        self.console.print()
        self._messages.append({"role": "assistant", "content": answer})
        self.logger.info(f"A[{self._lino}]: {answer}")

    def run(self):
        while True:
            try:
                user_input = self.console.input(f"[green]Q[{self._lino}]: [/green]")
                if user_input:
                    self.process_input(user_input)
            except EOFError:
                self.console.print("Exiting...")
                break
            except KeyboardInterrupt:
                self.console.print("Interrupting...")
            except Exception as e:
                self.console.print_exception(show_locals=True)
            else:
                self._lino += 1
