from langchain.chat_models import init_chat_model


class DEEPSEEKLLMClient:
    def __init__(self, cfg: dict,secret: dict):
        l = cfg["llm"]
        self.model = init_chat_model(
            model=l["model1"],
            api_key=secret["llm1_api_key"],
            base_url=secret["llm1_base_url"],
        )

    def invoke(self, messages):
        return self.model.invoke(messages).content

class CLOSEAILLMClient:
    def __init__(self, cfg: dict,secret: dict):
        l = cfg["llm"]
        self.model = init_chat_model(
            model=l["model2"],
            api_key=secret["llm2_api_key"],
            base_url=secret["llm2_base_url"],
        )

    def invoke(self, messages):
        return self.model.invoke(messages).content