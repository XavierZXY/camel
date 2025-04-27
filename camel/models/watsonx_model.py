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

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Union

import requests

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
            :obj:`WatsonXConfig().as_dict()` will be used. (default: :obj:`None`)
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
        token_counter: Optional[BaseTokenCounter] = None,
        timeout: Optional[float] = None,
    ) -> None:
        if model_config_dict is None:
            model_config_dict = WatsonXConfig().as_dict()

        api_key = api_key or os.environ.get("WATSONX_API_KEY")
        url = url or os.environ.get(
            "WATSONX_API_BASE_URL",
            "https://us-south.ml.cloud.ibm.com/ml/v1/generation/text",
        )
        timeout = timeout or float(os.environ.get("MODEL_TIMEOUT", 180))

        super().__init__(
            model_type=model_type,
            model_config_dict=model_config_dict,
            api_key=api_key,
            url=url,
            token_counter=token_counter,
            timeout=timeout,
        )

        # Initialize session for API calls
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }
        )

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
        # Prepare the request payload
        payload = self.model_config_dict.copy()
        payload.update(kwargs)

        # Format messages for Watson X
        formatted_prompt = self._format_messages_for_watsonx(messages)

        # Prepare the request body
        request_body = {
            "model": str(self.model_type),
            "input": formatted_prompt,
            "parameters": {
                "decoding_method": payload.get("decoding_method", "greedy"),
                "temperature": payload.get("temperature", 0.7),
                "max_new_tokens": payload.get("max_new_tokens", 1024),
                "min_new_tokens": payload.get("min_new_tokens", 0),
                "stop_sequences": payload.get(
                    "stop_sequences", ["User:", "\n\nUser:"]
                ),
            },
        }

        # Add optional parameters if present
        for param in ["top_p", "top_k", "repetition_penalty", "random_seed"]:
            if param in payload:
                request_body["parameters"][param] = payload[param]

        # Make the API call
        try:
            response = self.session.post(
                self._url, json=request_body, timeout=self._timeout
            )
            response.raise_for_status()
            response_data = response.json()

            # Extract the generated text
            generated_text = response_data.get("results", [{}])[0].get(
                "generated_text", ""
            )

            # Create a ChatCompletion object
            completion = ChatCompletion(
                id=f"watsonx-{uuid.uuid4()}",
                object="chat.completion",
                created=int(response_data.get("created_at", 0)),
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
                    "prompt_tokens": response_data.get("usage", {}).get(
                        "input_tokens", 0
                    ),
                    "completion_tokens": response_data.get("usage", {}).get(
                        "generated_tokens", 0
                    ),
                    "total_tokens": response_data.get("usage", {}).get(
                        "total_tokens", 0
                    ),
                }
                if "usage" in response_data
                else {},
            )

            return completion

        except Exception as e:
            raise Exception(f"Error calling Watson X API: {str(e)}")

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

    def token_counter(self) -> BaseTokenCounter:
        """Get a token counter for this model.

        Returns:
            BaseTokenCounter: A token counter for this model.
        """
        if self._token_counter is None:
            # Use OpenAI's token counter as a fallback
            self._token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        return self._token_counter

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
