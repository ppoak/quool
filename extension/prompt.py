from IPython.terminal.prompts import Prompts, Token


class QuoolPrompts(Prompts):
    def in_prompt_tokens(self):
        return [
            (Token.Prompt, self.vi_mode() ),
            (Token.Prompt, 'Quool#'),
            (Token.PromptNum, str(self.shell.execution_count)),
            (Token.Prompt, '<<< '),
        ]

    def out_prompt_tokens(self):
        return [
            (Token.OutPrompt, 'Quool#'),
            (Token.OutPromptNum, str(self.shell.execution_count)),
            (Token.OutPrompt, '>>> '),
        ]
