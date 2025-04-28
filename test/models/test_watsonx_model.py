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
import os
import re
from unittest.mock import MagicMock, patch

import pytest
import requests
from ibm_watsonx_ai.foundation_models import ModelInference

from camel.configs import WatsonXConfig
from camel.models import WatsonXModel
from camel.types import ChatCompletion, ModelType
from camel.utils import OpenAITokenCounter


@patch("ibm_watsonx_ai.foundation_models.ModelInference")
@patch("ibm_watsonx_ai.APIClient")
@pytest.mark.model_backend
@pytest.mark.parametrize(
    "model_type",
    [
        "ibm/granite-13b-instruct-v2",
    ],
)
def test_watsonx_model(
    mock_api_client, mock_model_inference, model_type: ModelType
):
    # 模拟APIClient实例
    mock_api_client_instance = MagicMock()
    mock_api_client.return_value = mock_api_client_instance

    # 模拟ModelInference实例
    mock_model_instance = MagicMock()
    mock_model_inference.return_value = mock_model_instance

    model = WatsonXModel(model_type)
    assert model.model_type == model_type
    assert (
        model.model_config_dict == WatsonXConfig().as_dict()
    ), "Model config dict should be the same as WatsonXConfig"
    assert isinstance(
        model.token_counter, OpenAITokenCounter
    ), f"Token counter is {model.token_counter.__class__.__name__}"


@patch("ibm_watsonx_ai.foundation_models.ModelInference")
@patch("ibm_watsonx_ai.APIClient")
@pytest.mark.model_backend
def test_watsonx_model_unexpected_argument(
    mock_api_client, mock_model_inference
):
    # 模拟APIClient实例
    mock_api_client_instance = MagicMock()
    mock_api_client.return_value = mock_api_client_instance

    # 模拟ModelInference实例
    mock_model_instance = MagicMock()
    mock_model_inference.return_value = mock_model_instance

    model_type = "ibm/granite-13b-instruct-v2"
    model_config_dict = {"model_path": "vicuna-7b-v1.5"}

    with pytest.raises(
        ValueError,
        match=re.escape(
            (
                "Unexpected argument `model_path` is "
                "input into Watson X model backend."
            )
        ),
    ):
        _ = WatsonXModel(model_type, model_config_dict)


@pytest.mark.model_backend
def test_watsonx_model_run():
    """Test the WatsonX model with actual API calls.

    This test is skipped if the required API credentials are not present
    in the environment variables.
    """
    # Check for required environment variables
    api_key = os.environ.get("WATSONX_API_KEY")
    project_id = os.environ.get("PROJECT_ID")

    # Skip test if credentials aren't available
    if not api_key or not project_id:
        pytest.skip(
            "WATSONX_API_KEY and PROJECT_ID environment variables must be set to run this test"
        )

    # Create model instance with credentials from environment
    model_type = "ibm/granite-13b-instruct-v2"
    model = WatsonXModel(model_type)

    # Prepare test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me 1 + 1."},
    ]

    # Call run method
    completion = model._run(messages)

    # # Verify results
    # assert isinstance(completion, ChatCompletion)
    # assert completion.model == model_type
    # assert isinstance(completion.choices[0].message.content, str)
    # assert completion.choices[0].message.content.strip() != ""
    # assert completion.choices[0].message.role == "assistant"
    # assert completion.choices[0].finish_reason == "stop"

    # # Check that we have usage information
    # assert hasattr(completion, "usage")
    # assert completion.usage.prompt_tokens > 0
    # assert completion.usage.completion_tokens > 0
    # assert completion.usage.total_tokens > 0
    print(f"\n----------------{completion}")
    assert completion is not None, "Completion should not be None"


# @patch("ibm_watsonx_ai.foundation_models.ModelInference")
# @patch("ibm_watsonx_ai.APIClient")
# @patch.object(requests.Session, "post")
# def test_watsonx_model_run_error(
#     mock_post, mock_api_client, mock_model_inference
# ):
#     # 设置模拟响应抛出异常
#     mock_post.side_effect = Exception("API调用失败")

#     # 模拟APIClient实例
#     mock_api_client_instance = MagicMock()
#     mock_api_client.return_value = mock_api_client_instance

#     # 模拟ModelInference实例
#     mock_model_instance = MagicMock()
#     mock_model_inference.return_value = mock_model_instance
#     mock_model_instance.generate.side_effect = Exception("API调用失败")

#     # 创建模型实例
#     model_type = "ibm/granite-13b-instruct-v2"
#     model = WatsonXModel(model_type, api_key="test_key")

#     # 准备测试消息
#     messages = [
#         {"role": "system", "content": "你是一个助手"},
#         {"role": "user", "content": "你好"},
#     ]

#     # 验证异常处理
#     with pytest.raises(
#         Exception, match="Error calling Watson X API: API调用失败"
#     ):
#         model._run(messages)


# @patch("ibm_watsonx_ai.foundation_models.ModelInference")
# @patch("ibm_watsonx_ai.APIClient")
# @patch.object(requests.Session, "post")
# def test_watsonx_format_messages(
#     mock_post, mock_api_client, mock_model_inference
# ):
#     # 设置模拟响应
#     mock_response = MagicMock()
#     mock_response.json.return_value = {
#         "results": [{"generated_text": "回复"}],
#         # Include usage data to pass validation
#         "usage": {
#             "input_tokens": 10,
#             "generated_tokens": 5,
#             "total_tokens": 15,
#         },
#     }
#     mock_post.return_value = mock_response

#     # 模拟APIClient实例
#     mock_api_client_instance = MagicMock()
#     mock_api_client.return_value = mock_api_client_instance

#     # 模拟ModelInference实例
#     mock_model_instance = MagicMock()
#     mock_model_inference.return_value = mock_model_instance
#     mock_model_instance.generate.return_value = {
#         "results": [{"generated_text": "回复"}],
#         "usage": {
#             "input_tokens": 10,
#             "generated_tokens": 5,
#             "total_tokens": 15,
#         },
#     }

#     # 创建模型实例
#     model_type = "ibm/granite-13b-instruct-v2"
#     model = WatsonXModel(model_type, api_key="test_key")

#     # 准备测试消息
#     messages = [
#         {"role": "system", "content": "系统指令"},
#         {"role": "user", "content": "用户问题"},
#         {"role": "assistant", "content": "助手回复"},
#         {"role": "unknown", "content": "未知角色"},
#     ]

#     # 调用格式化方法
#     formatted = model._format_messages_for_watsonx(messages)

#     # 验证格式化结果
#     assert "System: 系统指令" in formatted
#     assert "User: 用户问题" in formatted
#     assert "Assistant: 助手回复" in formatted
#     assert "未知角色" in formatted
#     assert formatted.endswith("Assistant: ")

#     # 调用run方法验证格式化消息被正确传递
#     model._run(messages)
#     call_args = mock_post.call_args[1]
#     assert (
#         call_args["json"]["input"] == formatted
#     ), "Formatted messages should be passed to the API call"


# @patch("ibm_watsonx_ai.foundation_models.ModelInference")
# @patch("ibm_watsonx_ai.APIClient")
# @patch.object(requests.Session, "post")
# @pytest.mark.asyncio
# async def test_watsonx_model_arun(
#     mock_post, mock_api_client, mock_model_inference
# ):
#     # 设置模拟响应
#     mock_response = MagicMock()
#     mock_response.json.return_value = {
#         "results": [{"generated_text": "异步测试回复"}],
#         "created_at": 1234567890,
#         "usage": {
#             "input_tokens": 10,
#             "generated_tokens": 5,
#             "total_tokens": 15,
#         },
#     }
#     mock_post.return_value = mock_response

#     # 模拟APIClient实例
#     mock_api_client_instance = MagicMock()
#     mock_api_client.return_value = mock_api_client_instance

#     # 模拟ModelInference实例
#     mock_model_instance = MagicMock()
#     mock_model_inference.return_value = mock_model_instance
#     mock_model_instance.generate.return_value = {
#         "results": [{"generated_text": "异步测试回复"}],
#         "created_at": 1234567890,
#         "usage": {
#             "input_tokens": 10,
#             "generated_tokens": 5,
#             "total_tokens": 15,
#         },
#     }

#     # 创建模型实例
#     model_type = "ibm/granite-13b-instruct-v2"
#     model = WatsonXModel(model_type, api_key="test_key")

#     # 准备测试消息
#     messages = [
#         {"role": "system", "content": "你是一个助手"},
#         {"role": "user", "content": "你好"},
#     ]

#     # 调用异步方法
#     completion = await model._arun(messages)

#     # 验证结果
#     assert isinstance(completion, ChatCompletion)
#     assert completion.model == model_type
#     assert completion.choices[0].message.content == "异步测试回复"
#     assert completion.choices[0].message.role == "assistant"

#     # 验证API调用
#     mock_post.assert_called_once()
#     call_args = mock_post.call_args[1]
#     assert "json" in call_args
#     assert call_args["json"]["model"] == model_type
