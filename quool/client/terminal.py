import re
from pathlib import Path
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm


class Terminal:

    def __init__(
        self,
        model: str = None,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        prompt: str = "",
        file: str = None,
    ):
        self.console = Console(record=True)
        self.setup_client(base_url, api_key)
        self.setup_model(model)
        self.setup_prompt(prompt)
        self.setup_file(file)
        self._lino = 1
        self.console.clear()
        self.console.print(Markdown(
            "# Quool\n\n"
            "Welcom to Quool Terminal\n\n"
            "Try to ask anything you want to know\n\n"
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
    
    def setup_file(self, file: str):
        self.file = file

    def process_input(self, user_input: str):
        if user_input.startswith("%"):
            self.console.print(self.process_command(user_input))
        else:
            self.console.print(self.process_chat(user_input))
    
    def process_command(self, user_input: str):
        commands = user_input[1:].split(" ")
        command = getattr(self, f"command_{commands[0]}", None)
        if command:
            command(*commands[1:])
        else:
            self.console.print(f"Unknown command: {commands[0]}")
    
    def process_chat(self, user_input: str):
        self._messages.append({"role": "user", "content": user_input})
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
        self.console.rule()
        self._messages.append({"role": "assistant", "content": answer})
        clear_answer = re.sub(r"<think>\s+", ">", answer)
        clear_answer = re.sub(r"\s+</think>", "\n\n", clear_answer)
        return Markdown(clear_answer)

    def run(self):
        while True:
            try:
                user_input = Prompt.ask(f"\n[green]Q[{self._lino}][/green]", console=self.console)
                if user_input:
                    self.process_input(user_input)
            except EOFError:
                self.console.print("[red]Exiting...[/red]")
                self.command_save()
                break
            except KeyboardInterrupt:
                self.console.print("[yellow]Interrupting...[/yellow]")
            except Exception as e:
                self.console.print_exception(show_locals=True)
            else:
                self._lino += 1

    def command_save(self):
        if Confirm.ask("Do you want to export the conversation to a HTML file?", console=self.console, default='Y'):
            if self.file is None:
                self.file = Prompt.ask("Enter the file name", console=self.console)
            Path(self.file).write_text(self.console.export_html(), encoding="utf-8")
