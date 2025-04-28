# ========= Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2024 @ CAMEL-AI.org. All Rights Reserved. =========

import os
import uuid
from typing import Any, Dict, List, Optional, Union

import dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from camel.configs import WATSONX_API_PARAMS, WatsonXConfig
from camel.models.base_model import BaseModelBackend
from camel.types import ChatCompletion, ModelType
from camel.utils import BaseTokenCounter, OpenAITokenCounter, api_keys_required


class WatsonXModel(BaseModelBackend):
    """Backend for IBM Watson X LLM API integration.

    Args:
        model_type (Union[ModelType, str]): Model for which a backend is
            created, supported model can be found in IBM Watson X documentation.
        model_config_dict (Optional[Dict[str, Any]], optional): A dictionary
            that will be fed into the API call. If :obj:`None`,
            :obj:`WatsonXConfig().as_dict()` will be used.
                (default: :obj:`None`)
        api_key (Optional[str], optional): The API key for authenticating with
            the Watson X service. (default: :obj:`None`)
        url (Optional[str], optional): The url to the Watson X service.
            If not provided, "https://us-south.ml.cloud.ibm.com/ml/v1/generation/text"
            will be used. (default: :obj:`None`)
        token_counter (Optional[BaseTokenCounter], optional): Token counter to
            use for the model. If not provided, :obj:`OpenAITokenCounter(
            ModelType.GPT_4O_MINI)` will be used. (default: :obj:`None`)
        timeout (Optional[float], optional): The timeout value in seconds for
            API calls. If not provided, will fall back to the MODEL_TIMEOUT
            environment variable or default to 180 seconds.
            (default: :obj:`None`)
    """

    @api_keys_required(
        [
            ("api_key", 'WATSONX_API_KEY'),
        ]
    )
    def __init__(
        self,
        model_type: Union[ModelType, str],
        model_config_dict: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        project_id: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        timeout: Optional[float] = None,
    ) -> None:
        # 加载环境变量
        dotenv.load_dotenv()

        if model_config_dict is None:
            model_config_dict = WatsonXConfig().as_dict()

        # 初始化凭证和客户端
        self.credentials = Credentials(
            url=url or os.getenv("URL"),
            api_key=api_key or os.getenv("WATSONX_API_KEY"),
        )
        self.client = APIClient(self.credentials)
        self.project_id = project_id or os.getenv("PROJECT_ID")

        timeout = timeout or float(os.environ.get("MODEL_TIMEOUT", 180))

        super().__init__(
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=api_key,
            url=url,
            token_counter=token_counter,
            timeout=timeout,
        )

        # 初始化模型推理实例
        self.model = ModelInference(
            model_id=str(model_type),
            api_client=self.client,
            project_id=self.project_id,
            # params=model_config_dict,
        )

    @property
    def token_counter(self) -> BaseTokenCounter:
        r"""Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(self.model_type)
        return self._token_counter

    def _format_messages_for_watsonx(
        self, messages: List[Dict[str, Any]]
    ) -> str:
        """Format messages for Watson X API.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries.

        Returns:
            str: Formatted prompt string for Watson X.
        """
        formatted_prompt = ""

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                formatted_prompt += f"System: {content}\n\n"
            elif role == "user":
                formatted_prompt += f"User: {content}\n\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n\n"
            else:
                formatted_prompt += f"{content}\n\n"

        # Add final assistant prompt
        formatted_prompt += "Assistant: "

        return formatted_prompt

    def _run(self, messages: List[Dict[str, Any]], **kwargs) -> ChatCompletion:
        """Run the Watson X model with the given messages.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            ChatCompletion: The model's response.

        Raises:
            Exception: If the API call fails.
        """
        # 格式化消息为提示语
        formatted_prompt = self._format_messages_for_watsonx(messages)

        # 使用ModelInference生成文本
        try:
            response = self.model.generate(formatted_prompt, **kwargs)
            generated_text = response.get("results", [{}])[0].get(
                "generated_text", ""
            )

            # 创建ChatCompletion对象
            completion = ChatCompletion(
                id=f"watsonx-{uuid.uuid4()}",
                object="chat.completion",
                created=response.get("created_at", 0),
                model=str(self.model_type),
                choices=[
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                usage={
                    "prompt_tokens": response.get("usage", {}).get(
                        "input_tokens", 0
                    ),
                    "completion_tokens": response.get("usage", {}).get(
                        "generated_tokens", 0
                    ),
                    "total_tokens": response.get("usage", {}).get(
                        "total_tokens", 0
                    ),
                }
                if "usage" in response
                else {},
            )

            return completion

        except Exception as e:
            raise Exception(f"Error calling Watson X API: {e!s}")

    async def _arun(
        self, messages: List[Dict[str, Any]], **kwargs
    ) -> ChatCompletion:
        """Run the Watson X model asynchronously with the given messages.

        This is a placeholder implementation that calls the synchronous method.
        For a true async implementation, you would use aiohttp or similar.

        Args:
            messages (List[Dict[str, Any]]): List of message dictionaries.
            **kwargs: Additional arguments to pass to the API.

        Returns:
            ChatCompletion: The model's response.
        """
        # For now, just call the synchronous method
        # In a real implementation, you would use aiohttp or similar
        return self._run(messages, **kwargs)

    def check_model_config(self) -> None:
        """Check whether the model configuration contains any
        unexpected arguments to Watson X API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to Watson X API.
        """
        for param in self.model_config_dict:
            if param not in WATSONX_API_PARAMS:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Watson X model backend."
                )
