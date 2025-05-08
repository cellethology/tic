# file: tic/annotation/llm_caller.py
"""
This file contains the code for calling the LLM to annotate the cell types.

prompt file is stored in: tic/annotation/prompt.txt
"""
import os
import json
from typing import Optional, Dict
from .llm_backends import get_llm

current_dir = os.path.dirname(os.path.abspath(__file__))
PROMPT_TEMPLATE = open(os.path.join(current_dir, "prompt.txt"), "r").read()

class LLMPredictor:
    """
    Unified interface for different LLM backends.
    """
    def __init__(
        self,
        model_name: str = "openai",
        model_kwargs: Optional[Dict] = None,
        prompt_template: str = PROMPT_TEMPLATE,
    ):
        self.llm = get_llm(model_name, model_kwargs or {})
        self.prompt_template = prompt_template

    def generate(self, user_input: str, system_prompt: str = "") -> str:
        return self.llm.generate(user_input=user_input, system_prompt=system_prompt)
    
    def _parse_response(self, response: str, return_json: bool = True) -> dict | str:
        """
        Parse the response from LLM.

        Mostly, the response looks like:

        ```json
        {
            "cluster_id": {
                "assigned_cell_type": "Cell Type Name",
                "reasoning": "Brief justification based on marker gene expression."
        ```
        {some other text}

        First, we need to find the first ```json and the last ```.
        Then, we parse the json content.
        """
        if return_json:
            start = response.find("```json")
            end = response.rfind("```")
            if start == -1 or end == -1 or start >= end:
                raise ValueError("Could not find valid JSON block in the response.")
            
            # Strip the ```json prefix
            json_content = response[start + len("```json"):end].strip()
            try:
                return json.loads(json_content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse LLM response as JSON.\nExtracted content:\n{json_content}\nOriginal error: {e}")
        else:
            return response
        
    def annotate_clusters(self, top_genes: dict, dataset_description: str = "", return_json: bool = True) -> dict | str:
        """
        Annotate clusters given top marker genes.

        Args:
            top_genes: dict mapping cluster_id to {"genes": [...], "scores": [...]}.
            dataset_description: str, description of the dataset.
        Returns:
            dict, annotated clusters in the format of :
            {
                "cluster_id": {
                    "assigned_cell_type": "Cell Type Name",
                    "reasoning": "Brief justification based on marker gene expression."
                },
                ...
            }
            or str, annotated clusters from LLM.
        """
        # Prepare user input by combining template and JSON payload
        payload = json.dumps(top_genes, ensure_ascii=False, indent=2)
        user_input = f"{self.prompt_template}\n\nProvided Results:\n{payload}\n"
        if dataset_description:
            user_input += f"\n\nDataset Description:\n{dataset_description}\n"
        # Call LLM
        response = self.generate(user_input)

        return self._parse_response(response, return_json)

