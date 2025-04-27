# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
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
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
from __future__ import annotations

from typing import Optional, Sequence, Union

from pydantic import Field

from camel.configs.base_config import BaseConfig


class WatsonXConfig(BaseConfig):
    r"""Defines the parameters for generating chat completions using the
    IBM WatsonX API.

    Args:
        temperature (float, optional): Sampling temperature to use, between
            :obj:`0` and :obj:`2`. Higher values make the output more random,
            while lower values make it more focused and deterministic.
            (default: :obj:`None`)
        top_p (float, optional): An alternative to sampling with temperature,
            called nucleus sampling, where the model considers the results of
            the tokens with top_p probability mass. So :obj:`0.1` means only
            the tokens comprising the top 10% probability mass are considered.
            (default: :obj:`None`)
        n (int, optional): How many chat completion choices to generate for
            each input message. (default: :obj:`None`)
        response_format (object, optional): An object specifying the format
            that the model must output. Setting to
            {"type": "json_object"} enables JSON mode, which guarantees the
            message the model generates is valid JSON.
        stream (bool, optional): If True, partial message deltas will be sent
            as data-only server-sent events as they become available.
            (default: :obj:`None`)
        stop (str or list, optional): Up to :obj:`4` sequences where the API
            will stop generating further tokens. (default: :obj:`None`)
        max_tokens (int, optional): The maximum number of tokens to generate
            in the chat completion. The total length of input tokens and
            generated tokens is limited by the model's context length.
            (default: :obj:`None`)
        presence_penalty (float, optional): Number between :obj:`-2.0` and
            :obj:`2.0`. Positive values penalize new tokens based on whether
            they appear in the text so far, increasing the model's likelihood
            to talk about new topics. (default: :obj:`None`)
        frequency_penalty (float, optional): Number between :obj:`-2.0` and
            :obj:`2.0`. Positive values penalize new tokens based on their
            existing frequency in the text so far, decreasing the model's
            likelihood to repeat the same line verbatim. (default: :obj:`None`)
        logit_bias (dict, optional): Modify the likelihood of specified tokens
            appearing in the completion. (default: :obj:`{}`)
        user (str, optional): A unique identifier representing your end-user,
            which can help monitor and detect abuse.
            (default: :obj:`None`)
    """

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[Union[str, Sequence[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[dict] = None
    frequency_penalty: Optional[float] = None
    logit_bias: dict = Field(default_factory=dict)
    user: Optional[str] = None


WATSONX_API_PARAMS = {param for param in WatsonXConfig.model_fields.keys()}
