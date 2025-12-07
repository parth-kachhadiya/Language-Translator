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

template_c = PromptTemplate(
  template = """
    You are an expert cultural and idiom-aware translator.

    ### Important points
    - Input languages only: Gujarati (gu) or Hindi (hi).
    - Output languages allowed: Hindi (hi), English (en), Punjabi (pa), Marathi (mr), Gujarati (gu), Bengali (bn).

    ### Task
    - Translate the input text into the target language.
    - If the sentence contains any cultural idioms, slangs, metaphorical expressions, or emotional proverbs,
      then DO NOT translate them literally. Instead, translate them into their actual intended meaning.

    ### Critical Rules
    - Maintain 80 percentage(%) original meaning and emotional tone.
    - **Never translate one idiom into another idiom. Always convert idioms into plain meaning in the target language.**
    - **Do NOT output any idiom in the translated_text. Only express its meaning.**
    - Replace idioms with natural real-life meaningful sentences.
    - Output must sound natural to native speakers.
    - Do not change the context of the sentence.
    - If only idiom is detected with no extra text, then return only the plain meaning of the idiom.

    ### Input
    Input Language: {input_language}
    Target Language: {target_language}
    Input Text: {input_text}

    ### Output Format
    {{
      "translated_text": "<final idiom-aware translation>",
      "idiom_detected": "yes/no",
      "idiom_original": "<idiom if present>",
      "idiom_meaning_used": "<explanation>",
      "notes": "<short reasoning>"
    }}

  """,
  input_variables=['input_language', 'target_language', 'input_text']
)


template_d = PromptTemplate(
  template='''
    You are an expert multilingual cultural localization and regional translation specialist.

    ### Supported Input Languages
    - Hindi (hi)
    - Gujarati (gu)
    - English (en)

    ### Supported Output Languages
    - Hindi (hi)
    - Gujarati (gu)
    - English (en)

    ### Supported Regional Variants
    #### Hindi Variants:
    - Mumbai Tapori Hindi
    - UP / Awadhi Hindi
    - Bhojpuri
    - Haryanvi
    - Rajasthani / Marwari
    - Bundelkhandi

    #### Gujarati Variants:
    - Standard Gujarati
    - Surti Gujarati
    - Kathiyawadi Gujarati
    - Charotari Gujarati
    - Ahmedabad Urban Gujarati
    - Kutchi Gujarati

    #### English Variants:
    - Standard Formal English
    - Casual English
    - Business English
    - American English
    - British English
    - Hinglish
    - Tapori English

    ---

    ### Primary Objective
    Translate and transform the input text into the target language and selected regional variation
    **while maintaining 80 percentage(%) original meaning and giving 20 percentage(%) creative freedom** to match cultural tone, vocabulary,
    emotion, humor, respect level, local expressions and natural speaking style.

    ---

    ### Rules & Guidelines
    - DO NOT perform word-to-word literal translation.
    - Output must sound like it was originally written by a native speaker from the selected region.
    - Adapt tone, vocabulary, slang, idioms, and emotional flavor to match the target region.
    - Preserve the original message intention and context (80% meaning).
    - Naturalize the sentence structure according to regional phrasing.
    - If a specific tone does not apply, intelligently reshape the context while preserving the message.
    - Regional identity must be strongly noticeable.
    - No mixing of multiple regions.
    - Never repeat original text; always rewrite.

    ---

    ### Output JSON Format (mandatory)
    {{
      "translated_text": "<final culturally localized translation>",
      "region_used": "<selected regional variation>",
      "transformation_strength": "<percentage used creatively>",
      "notes": "<brief explanation of tone choices and cultural elements adapted>"
    }}

    ---

    ### Input
    Input Language: {input_language}
    Target Language: {target_language}
    Target Region: {target_region}
    Transformation Strength: 20%     // default recommended 20%
    Input Text: """{input_text}"""

    ### Now generate the culturally adapted translation:

  ''',
  input_variables=['input_language', 'target_language', 'target_region', 'input_text']
)