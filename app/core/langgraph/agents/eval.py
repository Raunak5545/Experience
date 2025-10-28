import json
import os
from typing import (
    Any,
    Dict,
    Optional,
)

from google import genai
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.schema.experience import TravelPlan


class EvalAgent:
    """Agent that evaluates extracted travel information against the raw input for quality and compliance."""

    def __init__(self):
        self.eval_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=0.2,
            google_api_key=settings.LLM_API_KEY,
        )

        self.prompt = """
        ## 🎯 Travel Information Evaluation Prompt
        You are a **Travel Information Evaluation Specialist**.
        Your job is to **evaluate the quality and reliability** of a travel information extraction result.

        ### 🧩 Inputs
        You are given:
        1. **Original Input Files** — the user's initial travel data or documents.
        2. **Extracted Experience (Final Output)** — the model-generated narrative of travel details.

        ### 🧠 Evaluation Goals
        You must evaluate how accurate, faithful, and concise the extraction is.

        ### 📊 Evaluation Parameters
        Provide the following metrics (JSON format only):

        - **hallucination** → Value between 0.0 and 1.0  
          (How much information is added but not present or supported in the original input)
        - **accuracy** → Value between 0.0 and 1.0  
          (How correctly the output matches facts and details from the input)
        - **conciseness** → Value between 0.0 and 1.0  
          (How well the output avoids redundancy while covering essential details)
        - **structure_compliance** → "Pass" or "Fail"  
          (Whether it follows the required narrative format — detailed but structured logically)
        - **overall_score** → Integer 0–100  
          (Weighted aggregation of the above factors, representing total quality)
        - **validation_required** → Boolean (true/false)  
          (Whether human validation is recommended based on the evaluation)
        - **validation_reason** → String  
          (Brief explanation of why validation is needed, if applicable)

        ### ⚙️ Evaluation Logic
        - Compare extracted content with original input.  
        - Identify any **fabricated or hallucinated details**.  
        - Check if **key information** is missing or misinterpreted.  
        - Assess **clarity, completeness, and logical structure**.
        - Ensure the narrative feels **human-like but faithful** to input facts.
        - Recommend validation if overall_score < 80, hallucination > 0.15, or structure_compliance is "Fail"

        ### ✅ Output Format
        Return **strictly in JSON format** like this:
        ```json
        {
          "hallucination": 0.01,
          "accuracy": 0.97,
          "conciseness": 0.89,
          "structure_compliance": "Pass",
          "overall_score": 93,
          "validation_required": false,
          "validation_reason": ""
        }
        ```

        Do NOT include explanations or additional commentary.
        """

    def evaluate_experience(self, experience: Dict[str, Any], raw_input: Optional[str] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the extracted experience against the original input"""
        
        # Prepare the evaluation prompt with original input and extracted experience
        original_input_text = raw_input or "No text input provided."
        
        eval_prompt = f"""
        ## 🎯 Travel Information Evaluation Prompt
        You are a **Travel Information Evaluation Specialist**.
        Your job is to **evaluate the quality and reliability** of a travel information extraction result.

        ### 🧩 Inputs
        **Original Input**: {original_input_text}
        **Extracted Experience (Final Output)**: {json.dumps(experience, indent=2)}

        ### 🧠 Evaluation Goals
        You must evaluate how accurate, faithful, and concise the extraction is.

        ### 📊 Evaluation Parameters
        Provide the following metrics (JSON format only):

        - **hallucination** → Value between 0.0 and 1.0  
          (How much information is added but not present or supported in the original input)
        - **accuracy** → Value between 0.0 and 1.0  
          (How correctly the output matches facts and details from the input)
        - **conciseness** → Value between 0.0 and 1.0  
          (How well the output avoids redundancy while covering essential details)
        - **structure_compliance** → "Pass" or "Fail"  
          (Whether it follows the required narrative format — detailed but structured logically)
        - **overall_score** → Integer 0–100  
          (Weighted aggregation of the above factors, representing total quality)
        - **validation_required** → Boolean (true/false)  
          (Whether human validation is recommended based on the evaluation)
        - **validation_reason** → String  
          (Brief explanation of why validation is needed, if applicable)

        ### ⚙️ Evaluation Logic
        - Compare extracted content with original input.  
        - Identify any **fabricated or hallucinated details**.  
        - Check if **key information** is missing or misinterpreted.  
        - Assess **clarity, completeness, and logical structure**.
        - Ensure the narrative feels **human-like but faithful** to input facts.
        - Recommend validation if overall_score < 80, hallucination > 0.15, or structure_compliance is "Fail"

        ### ✅ Output Format
        Return **strictly in JSON format** like this:
        ```json
        {{
          "hallucination": 0.01,
          "accuracy": 0.97,
          "conciseness": 0.89,
          "structure_compliance": "Pass",
          "overall_score": 0.93,
          "validation_required": false,
          "validation_reason": ""
        }}
        ```

        Do NOT include explanations or additional commentary.
        """
        
        try:
            response = self.eval_llm.invoke([HumanMessage(content=eval_prompt)])
            content = response.content.strip()
            
            # Try to parse as JSON
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            evaluation = json.loads(content)
            return evaluation
        except Exception as e:
            return {
                "error": f"Failed to evaluate: {str(e)}",
                "hallucination": 0.0,
                "accuracy": 0.0,
                "conciseness": 0.0,
                "structure_compliance": "Fail",
                "overall_score": 0,
                "validation_required": True,
                "validation_reason": "Evaluation failed due to error"
            }

    def extract_from_text(self, text: str) -> str:
        response = self.text_llm.invoke([HumanMessage(content=self.prompt + f"Text To Analyze from : {text}")])
        return response.content

    def extract_from_file(self, file_path: str, extra_prompt: Optional[str] = None) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        print(extra_prompt)
        task_prompt = extra_prompt  or  "Extract key travel information (dates, destinations, travelers, etc.) from this file."
        full_prompt = f"{self.prompt}\n\n{task_prompt}"
        print(full_prompt)
        
        uploaded_file = self.multimodal_client.files.upload(file=file_path)
        response = self.multimodal_client.models.generate_content(
            model=settings.LLM_MODEL,
            contents=[uploaded_file, full_prompt]
        )
        return response.text

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        
        # Get the final combined output (experience)
        experience = state.get("experience", {})
        basic_detail=state.get("basic_information",{})
        TravelPlan=state.get("travel_plan",{})
        
        # Print the extracted final combined output
        print("\n" + "=" * 80)
        print("EXTRACTED FINAL COMBINED OUTPUT:")
        print("=" * 80)
        print(json.dumps(experience, indent=2))
        print("=" * 80)
        
        # Get original input for evaluation
        raw_input: Optional[str] = state.get("raw_input")
        file_path: Optional[str] = state.get("input_file_path")
        
        # if not experience:
        #     return {
        #         "evaluation": {"error": "No experience data to evaluate"},
        #     }
        
        # Evaluate the experience against original input
        evaluation = self.evaluate_experience({"basic_information": basic_detail, "travel_plan": TravelPlan}, raw_input, file_path)
        
        return {
            "evaluation": evaluation,
        }