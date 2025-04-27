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

from typing import Any, Dict, Optional

from camel.configs import BaseConfig

# WatsonX API parameters
# Reference: https://cloud.ibm.com/apidocs/watsonx-ai
WATSONX_API_PARAMS = [
    "model",
    "api_key",
    "input",
    "parameters",
    "project_id",
    "version",
    "space_id",
    "return_options",
    "stop_sequences",
    "max_new_tokens",
    "min_new_tokens",
    "decoding_method",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "random_seed",
]


class WatsonXConfig(BaseConfig):
    r"""Configuration for IBM Watson X LLM API.

    Reference: https://cloud.ibm.com/apidocs/watsonx-ai

    Args:
        include_usage (bool, optional): Whether to include usage information
            in the response. (default: :obj:`True`)
    """

    include_usage: bool = True

    def __init__(self, include_usage: bool = True, **kwargs):
        kwargs_with_usage = {"include_usage": include_usage, **kwargs}
        super().__init__(**kwargs_with_usage)

    def as_dict(self) -> Dict[str, Any]:
        r"""Returns the configuration as a dictionary.

        Returns:
            Dict[str, Any]: The configuration dictionary.
        """
        config_dict = super().as_dict()
        return config_dict
