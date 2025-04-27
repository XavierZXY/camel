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

"""
This example demonstrates how to use the IBM Watson X LLM API integration in CAMEL.

Before running this example, make sure you have set the WATSONX_API_KEY
environment variable or pass it directly to the model.

Example usage:
    export WATSONX_API_KEY="your_api_key_here"
    python watsonx_model_example.py
"""

import os
from typing import List

from camel.models import ModelFactory
from camel.types import ModelPlatformType


def main():
    # Set your API key here or use environment variable
    api_key = os.environ.get("WATSONX_API_KEY")
    
    if not api_key:
        print("Please set the WATSONX_API_KEY environment variable or provide it directly.")
        return
    
    # Create a WatsonX model instance
    # You can specify a specific model name as a string
    model = ModelFactory.create(
        model_platform=ModelPlatformType.WATSONX,
        model_type="ibm/granite-13b-chat-v2",  # Example model, replace with an actual WatsonX model
        api_key=api_key,
        # Optional: custom endpoint URL
        # url="https://your-custom-endpoint.ibm.com/ml/v1/generation/text",
        # Optional: custom configuration
        model_config_dict={
            "temperature": 0.5,
            "max_new_tokens": 500,
            "top_p": 0.9,
        }
    )
    
    # Example messages for a simple chat
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Tell me about IBM Watson X AI platform in 3 sentences."}
    ]
    
    # Get a response from the model
    try:
        response = model.run(messages)
        print("\nResponse from Watson X:")
        print(response.choices[0].message.content)
        
        # Print usage information if available
        if hasattr(response, "usage") and response.usage:
            print("\nToken Usage:")
            print(f"  Prompt tokens: {response.usage.get('prompt_tokens', 'N/A')}")
            print(f"  Completion tokens: {response.usage.get('completion_tokens', 'N/A')}")
            print(f"  Total tokens: {response.usage.get('total_tokens', 'N/A')}")
    
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()