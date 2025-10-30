Analyze the following travel information and determine if it contains:
        1. Destination or City (specific location)
        2. Activities or Attractions

        Prompt user and ask for missing information if any of the above are absent.

        **Return validated if we have everything that we need**
        Extracted Information:

        {extracted_text}

        Respond in JSON format:
        {{
        "validated": false,
        "prompt": ""
        }}