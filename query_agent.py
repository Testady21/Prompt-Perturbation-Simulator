from langchain_community.llms.ollama import Ollama


class QueryAgent:
    """S
    A class to manage interactions with a self-hosted Ollama LLM,
    including optional conversation history.
    """
    def __init__(self, model: str, system_prompt: str = ""):
        # Fixed Base URL for your Docker container
        self.base_url: str = "http://localhost:11437"
        
        # Use ChatOllama instead of undefined OllamaLLM
        self.llm = Ollama(model=model, base_url=self.base_url)

        self.history_lst: list[tuple[str, str]] = []
        self.system_prompt: str = system_prompt

    def _generate_query(self, query: str, history: bool = False) -> str:
        prompt = ""
        if history and self.history_lst:
            fmt_history = "\n".join(
                f"User: {q}\nAssistant: {a}" for q, a in self.history_lst
            )
            prompt = f"""
                --------------
                System prompt: {self.system_prompt},
                --------------
                Conversation History: {fmt_history},
                --------------
                User query: {query}
                --------------
            """
        else:
            prompt = f"""
                --------------
                System prompt: {self.system_prompt},
                --------------
                User query: {query}
                --------------
            """

        return " ".join(prompt.split())

    def query(self, query: str, history: bool = False) -> str:
        prompt = self._generate_query(query, history)
        response = self.llm.invoke(prompt)  # call LLM
        if history:
            self.history_lst.append((query, response))
        return response

    def clear_history(self):
        self.history_lst.clear()
