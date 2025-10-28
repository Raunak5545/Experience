from typing import Any, Dict
from click import prompt
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from app.core.config import settings
from app.core.langgraph.agents.globalstate import TravelAgentState
from app.core.langgraph.schema.experience import BasicInfo, ExperienceTagsOutputScehma
from langchain_core.tools import tool
from app.core.langgraph.data.experience_taxonomy import TAXONOMY
from app.core.langgraph.tools.experience_types_tags import get_full_experience_taxonomy
import time


class BasicInfoAgent:
    """
    Agent that extracts basics info from the text in a structured JSON format.
    """
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.BASIC_INFO_MODEL,
            temperature=0.4,
            google_api_key=settings.LLM_API_KEY
        )

        self.prompt = """ 
                    You are an expert at extracting and structuring information from unstructured text sources (such as extracted content from PDFs, video 
                    transcripts, image descriptions, or raw text files) into a standardized JSON format for tour or experience listings. 
                    Your output must be exactly valid JSON matching the structure below—no extra fields, no markdown, no explanations. 
                    Use the input text to populate fields intelligently: infer or extract details where possible, but leave as empty strings
                    or empty arrays if not explicitly mentioned. For location, if coordinates are not in the text, use approximate ones based on 
                    the place (e.g., via known geography). For Inclusion and Exclusion, create arrays of strings listing key inclusions/exclusions mentioned.
                    For FAQ, generate any number of question-answer pairs (0 or more): questions must be relatable to the scenario (e.g., suitability, 
                    best options, logistics) and directly inspired by details in the source text; answers must be concise paraphrases or direct quotes from the 
                    text supporting the question.
                    
                    Input text: {text}
                    
                    Output JSON structure:
                    {{
                        "caption": "A short, engaging one-line caption summarizing the experience.",
                        "summary": ["Bullet point 1 summary from text.", "Bullet point 2 summary from text."],
                        "location": {{
                            "city": "Extracted or inferred city",
                            "state": "Extracted or inferred state",
                            "country": "Extracted or inferred country",
                            "placeName": "Full place or tour name from text",
                            "coordinates": {{
                                "type": "Point",
                                "coordinates": [longitude, latitude]
                            }}
                        }},
                        "inclusion": ["List item 1 from text.", "List item 2 from text."],
                        "exclusion": ["List item 1 from text.", "List item 2 from text."],
                        "faq": [
                            {{
                                "question": "Relatable question 1 based on text details.",
                                "answer": "Concise answer drawn from text."
                            }},
                            {{
                                "question": "Relatable question 2 based on text details.",
                                "answer": "Concise answer drawn from text."
                            }}
                        ]
                    }}
                    
                    Note: Add more FAQ entries as needed based on text, or use an empty array if no relevant FAQs can be formed.
                    Ensure the JSON is complete and parseable. If the text lacks details for a field, use empty string or empty array but keep the structure intact.
                    """
        self.tags_prompt = """
                You are an expert travel experience analyzer specialized in understanding travel descriptions
                    and structuring them into a taxonomy of categories, types, and subtypes.
                    Your goal is to produce an accurate, structured JSON representation of the given travel experience.

### PLAN & EXECUTION RULES (ANTI-HALLUCINATION EDITION)

                    1. **Tool Usage (MANDATORY)**
                       - You have access to the following tool to query the taxonomy:
                         - `get_full_experience_taxonomy` → returns the **complete, authoritative taxonomy** (categories → types → subtypes).
                       - **You MUST call `get_full_experience_taxonomy` at least once** before selecting any item.
                       - **Every category, type, and subtype you output MUST exist verbatim in the taxonomy returned by the tool.**
                       - If a plausible item is **not** in the taxonomy, **do not use it**—choose the closest valid match or omit it.
                       - After tool use, **you MUST provide your final answer as a JSON object only**.

                    2. **Determine Experience Category**
                       - If `experienceCategory` is provided in the input, **use it exactly** (validate it exists in taxonomy).
                       - If not provided, infer **up to 3 categories** that:
                         - Are **explicitly mentioned** or **directly implied** by concrete nouns/verbs in the text.
                         - Have **at least 2 supporting phrases** in the input.
                       - **Never infer a category from a single ambiguous word.**

                    3. **Extract Experience Types (Evidence-Based)**
                       - For each selected category, list **all types** returned by the taxonomy.
                       - Score each type 0–3 based on **direct evidence** in the text:
                           - 3 = Explicitly named.
                           - 2 = Strongly implied by 2+ specific details.
                           - 1 = Weakly implied by 1 detail.
                           - 0 = No evidence.
                       - Select the **top 2 types with score ≥ 2**. If none reach 2, select the **highest-scoring valid type** (max 2).

                    4. **Extract Experience Subtypes (Strict Matching)**
                       - For each selected type, list **all subtypes** from the taxonomy.
                       - Only include a subtype if **at least one concrete detail** in the text matches its definition **exactly**.
                       - Max **4 subtypes per type**. If fewer than 4 qualify, output only the valid ones.

                    5. **Generate Experience Tags (Grounded in Text + Taxonomy)**
                       - Produce **exactly 8 tags**.
                       - **Source rule**: 
                           - 50% (4 tags) must be **direct noun/verb phrases** from the input text (≤ 3 words each).
                           - 50% (4 tags) must be **valid subtypes** from step 4 or **taxonomy-defined attributes** of selected types.
                       - **No synonyms, no rephrasing, no invented themes.**

                    6. **Generate Secondary Suggestions (Always Include, Same Constraints)**
                       - **Secondary Experience Types**: The next 2 types with highest evidence score **below** the primary ones (score ≥ 1).
                       - **Secondary Experience Subtypes**: 1–3 subtypes per secondary type with **exact text evidence**.
                       - **Secondary Experience Tags**: 5 tags following the same 50/50 source rule as primary tags.
                    ### INPUT = {text}
                    ### ANTI-HALLUCINATION GUARDRAILS
                    - **Zero tolerance for fabrication**: If no evidence exists for a field, use an empty array `[]` instead of guessing.
                    - **Confidence logging (internal only)**: Before final JSON, note `[Confidence: High/Medium/Low]` for each array based on evidence strength.
                    - **Image input rule**: If input is an image, **only use visually identifiable objects/actions**. Ignore assumptions (e.g., a boat with people ≠ "bird watching" unless birds are visible and focused).

### CRITICAL: FINAL OUTPUT REQUIREMENT
                    After using `get_full_experience_taxonomy`, respond with **ONLY** a valid JSON object (no markdown, no explanation, no code blocks).

                    ```json
                    {{
                      "experienceCategory": ["Category1", "Category2", "Category3"],
                      "experienceTypes": ["Type1", "Type2"],
                      "experienceSubTypes": ["Subtype1", "Subtype2", "Subtype3", "Subtype4"],
                      "experienceTags": ["Tag1", "Tag2", "Tag3", "Tag4", "Tag5", "Tag6", "Tag7", "Tag8"],
                      "secondaryTags": {{
                        "experienceTypes": ["SecondaryType1", "SecondaryType2"],
                        "experienceSubTypes": ["SecondarySubtype1", "SecondarySubtype2", "SecondarySubtype3"],
                        "experienceTags": ["SecondaryTag1", "SecondaryTag2", "SecondaryTag3", "SecondaryTag4", "SecondaryTag5"]
                      }}
                    }}
                            """

        self.tools = [
            get_full_experience_taxonomy
        ]

        self.agent_executor = create_react_agent(
            self.llm,
            self.tools
        )

    def execute(self, state: TravelAgentState) -> Dict[str, Any]:
        extracted_text = state.get("extracted_text")
        if not extracted_text:
            return {"next": "extraction"}
        
        llm_structured = self.llm.with_structured_output(BasicInfo)
        start = time.time()
        response = llm_structured.invoke([
            HumanMessage(self.prompt.format(text=extracted_text))
        ])
        duration = time.time() - start
        print(f"[Timing] BasicInfoAgent LLM call finished in {duration:.2f} seconds.")
        tags_info = self.extract_tags(state)
        print("Returning from basic-info")
        return {"basic_info": response,"tags_info":tags_info}

    def extract_tags(self, state: TravelAgentState) -> Dict[str, Any]:
        extracted_text = state.get("extracted_text")
        if not extracted_text:
            return {"next": "extraction"}
        start = time.time()
        # First invocation - let agent use tools
        result = self.agent_executor.invoke(
                        {
                            "messages": [HumanMessage(content=self.tags_prompt.format(text=extracted_text))]
                        },
                        config={
                            "max_iterations": 1,   # single reasoning step
                            "timeout": 60,
                            "stream": False
                        }
                    )        
        print("Agent result messages:")
        for msg in result["messages"]:
            print(f"Type: {type(msg).__name__}, Content: {msg.content[:200] if msg.content else 'EMPTY'}")
        
        final_message = result["messages"][-1]
        
        # If final message is empty, the agent might have finished with a tool call
        # We need to prompt it again to generate the final JSON
        if not final_message.content or final_message.content.strip() == "":
            print("Final message empty, requesting JSON output...")
            result = self.agent_executor.invoke(
                        {
                            "messages": [HumanMessage(content=self.tags_prompt.format(text=extracted_text))]
                        },
                        config={
                            "max_iterations": 1,   # single reasoning step
                            "timeout": 60,
                            "stream": False
                        }
                    )        

            final_message = result["messages"][-1]
        
        print(f"Final message content: {final_message.content}")
        
        llm_structured = self.llm.with_structured_output(ExperienceTagsOutputScehma)
        tag_start = time.time()
        tags_info = llm_structured.invoke([
            HumanMessage(content=f"Convert this to structured format: {final_message.content}")
        ])
        tag_duration = time.time() - tag_start
        print(f"[Timing] BasicInfoAgent tags LLM call finished in {tag_duration:.2f} seconds.")
        print(tags_info)
        total_duration = time.time() - start
        print(f"[Timing] BasicInfoAgent extract_tags total finished in {total_duration:.2f} seconds.")
        return tags_info
