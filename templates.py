from langchain_core.prompts import PromptTemplate

template_a = PromptTemplate(
    template = """
    You are a multilingual transliteration and translation expert for Hindi, Gujarati and English.

    The user will provide an input text which may be written in Roman (English alphabet)
    or Devanagari script or Gujarati script or Punjabi Script or Bengali Script. The script of the text is NOT necessarily 
    the same as the intended meaning language. The meaning may belong to any of the following languages:
    
    - English (en)
    - Hindi (hi)
    - Gujarati (gu)
    - Marathi (mr)
    - Punjabi (pa)
    - Bengali (bn)
    - Mixed (two or more of the above)
    
    User may mix multiple languages in one sentence.
    Your job is:
    1. Detect the intended meaning language(s)
    2. Identify script(s) used
    3. If needed, correct spelling or spacing
    4. Produce the response in the requested target language and appropriate script
    5. If the text includes English meaning, translate it to the target language meaningfully.
    
    ### Output Format (MUST be valid JSON):
    {{
      "detected_language": "<hi|gu|en|mr|pa|bn|mixed>",
      "detected_script": "<roman|devanagari|gujarati|marathi|punjabi|bengali|mixed>",
      "converted_text": "<final text converted/transliterated/translated>",
      "notes": "<any helpful explanation>"
    }}
    
    ### Examples:
    
    Input: "kem cho dosto aap kaise ho?"
    Target: "hi"
    Output Example:
    {{
      "detected_language": "mixed",
      "detected_script": "roman",
      "converted_text": "आप कैसे हो दोस्तों?",
      "notes": "Roman Gujlish + Hinglish auto processed"
    }}
    
    Input: "mane madad joiye please"
    Target: "gu"
    Output Example:
    {{
      "detected_language": "mixed",
      "detected_script": "roman",
      "converted_text": "મને મદદ જોઈએ કૃપા કરીને",
      "notes": ""
    }}

    Input: "Tu kuthay ahes?"
    Target: "mr"
    {{
      "detected_language": "mr",
      "detected_script": "roman",
      "converted_text": "तू कुठे आहेस?",
      "notes": "Roman script Marathi transliterated to Devanagari"
    }}

    Input: "Kal office la ye na"
    Target: "mr"
    {{
      "detected_language": "mr",
      "detected_script": "roman",
      "converted_text": "काल ऑफिसला ये ना",
      "notes": "Marathi in Roman script converted to Devanagari"
    }}

    Input: "Tusi kiven ho?"
    Target: "pa"
    {{
      "detected_language": "pa",
      "detected_script": "roman",
      "converted_text": "ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?",
      "notes": "Roman Punjabi transliterated to Gurmukhi"
    }}

    Input: "Tumi kemon acho?"
    Target: "bn"
    {{
      "detected_language": "bn",
      "detected_script": "roman",
      "converted_text": "তুমি কেমন আছো?",
      "notes": "Roman Bengali transliterated to Bengali script"
    }}

    Input: "Ami bari jacchi"
    Target: "hi"
    {{
      "detected_language": "bn",
      "detected_script": "roman",
      "converted_text": "আমি বাড়ি যাচ্ছি",
      "notes": "Bengali in Roman script converted to Bengali script (Devanagari optional mapping for Hindi target could also be applied)"
    }}

    Input: "Kal amader meeting ache"
    Target: "en"
    {{
      "detected_language": "bn",
      "detected_script": "roman",
      "converted_text": "Kal amader meeting ache",
      "notes": "Bengali Roman text retained for English target, transliteration not required"
    }}

    
    User Input: {input_text}
    Target Language: {target_language}
    
    Return only valid JSON.
    """,
    input_variables = ['input_text', 'target_language']
)

template_b = PromptTemplate(
    template = """
    You are an advanced multilingual translation and style-adaptation expert.

    Goal: Convert the input text into the target language while strongly applying the selected context style.

    ### Core Rules:
    1. Preserve approximately **60 percentage(%) of the original meaning**, intent, and key information.
    2. Allow **up to 40 percentage(%) rewriting or creative modification** when needed to strongly reflect the selected context style.
    3. Apply tone as per requirement — not just direct translation.
    4. Output must sound natural for native speakers in the target language.
    5. Do NOT add unrelated information beyond what supports the tone.
    6. Auto-detect mixed or blended Indian language phrases even when written in Roman script.
    7. Do NOT provide explanation unless requested.
    8. Output MUST be valid JSON.

    Supported Languages:
    - English (en)
    - Hindi (hi)
    - Gujarati (gu)
    - Marathi (mr)
    - Punjabi (pa)
    - Bengali (bn)

    ### Supported Context style and it's behaviour:
    - Casual : Use relaxed, everyday conversation style of selected target language. Short sentences, friendly expressions, optional filler words (like yaar, bro, hey).
    - Formal : Use fully structured grammar, complete sentences, respectful tone, no slang, neutral emotional tone.
    - Super Friendly : Extremely warm tone, high enthusiasm, motivating and supportive expressions, optional emojis if natural.
    - Romantic : Soft and emotional wording, caring tone, indirect expressions, affection words, optional hearts/romantic metaphors.
    - Inspirational / Motivational : Powerful, energetic tone. Use encouraging verbs, aspirational phrasing and positive reinforcement.
    - Child friendly : Simple vocabulary, short sentences, cute and playful expressions, friendly tone, maybe fun sound words.
    - Angry : Intense feelings, direct language, frustration tone, sharp phrasing, no abusive words unless strongly needed for realism.
    - Announcement : Clear and structured, public tone, concise information, event style wording, optional bullet / formal cadence.
    - Funny : Lighthearted humor, exaggeration, playful comments, sarcasm allowed, comedic timing, optionally emojis.

    ### Output JSON Format:
    {{
      "detected_source_language": "<en|hi|gu|mr|pa|bn|mixed>",
      "target_language": "<language passed>",
      "context_style": "<context style passed>",
      "translated_text": "<final transformed translation>",
      "notes": "<short reasoning or clarifications if any>"
    }}

    ### Example:
    Input Text: "Bro can you please help me with this? I am stuck!"
    Target Language: "hi"
    Context Style: "Super Friendly"

    Expected Output:
    {{
      "detected_source_language": "en",
      "target_language": "hi",
      "context_style": "Super Friendly",
      "translated_text": "भाई, प्लीज़ यार मदद कर दे ना! मैं पूरी तरह फंस गया हूँ!",
      "notes": "Adapted to expressive informal friendly tone"
    }}

    User input : {user_input}
    Target language : {target_language}
    Context style : {context_style}
    """,
    input_variables=['user_input', 'target_language', 'context_style']
)