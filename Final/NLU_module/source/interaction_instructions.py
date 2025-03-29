init_generate_answer = lambda user_input, context: f"""
Please use a paragraph to give professional medical advice based on the user's input (after the tag "User_input") and the knowledge (after the tag "Knowledge"):
---
User_input: {user_input}
Knowledge: {context}
---
"""

generate_answer = lambda user_input, context, history: f"""
Please use a paragraph to give professional medical advice based on the user's input (after the tag "User_input"), the knowledge (after the tag "Knowledge"), and the conversation history (after the tag "History"):
---
User_input: {user_input}
Knowledge: {context}
History: {history}
---
"""

generate_answer_without = lambda user_input, context: f"""
Please use a paragraph to give professional medical advice based on the user's input (after the tag "User_input") and the knowledge (after the tag "Knowledge"):
---
User_input: {user_input}
Knowledge: {context}
---
"""

make_adjustment = lambda last_response, history, suggestion: f"""
You are in a conversation with a user, and your last answer (after the tag "Last_response") has a security issue;
Please adjust your last answer (after the tag "Last_response") in one paragraph based on the security suggestions (after the tag "Safety_suggestion") and historical conversations (after the tag "History"):
---
Safety_suggestion: {suggestion}
Last_response: {last_response}
History: {history}
---
"""

make_adjustment_without = lambda last_response, suggestion: f"""
You are in a conversation with a user, and your last answer (after the tag "Last_response") has a security issue;
Please adjust your last answer (after the tag "Last_response") in one paragraph based on the security suggestions (after the tag "Safety_suggestion"):
---
Safety_suggestion: {suggestion}
Last_response: {last_response}
---
"""

check_cur_response = lambda suggestion: f"""
Answer the following questions and include your reasons in your explanation:
response_is_safe:<Whether the advice of the advisor (after the tag "adviser_suggestion") is safe. Output "True or "False">
explanation:<Output explanation here to explain whether the advice given by the advisor is safe>
---
adviser_suggestion: {suggestion}
---
"""

yaml_correct_answer = f"""
First, provide the advisor's response on whether it is safe (after the label "response_is_safe"), using True or False, immediately following the colon, with no line breaks;
Then, provide your thought process (after the label "explanation"). Format "response_is_safe" and "explanation" using the following YAML format; please do not deviate from the format:
---
response_is_safe:<Whether the advice of the advisor (after the tag "adviser_suggestion") is safe. Output "True or "False">
explanation:<Output explanation here to explain whether the advice given by the advisor is safe>
---
"""
