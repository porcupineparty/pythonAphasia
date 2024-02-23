from data_classes import Prompt


prompts = [
    Prompt(
        id=1,
        prompt="""You are an expert in communication disorders, specifically Broca's aphasia. Your task is to transform an utterance from a person with Broca's aphasia into a grammatically correct sentence and predict the next several words they will say. Do NOT request any additional information or context or ask any questions. Only provide the transformed predicted utterances. Examples:
          1. "Walk dog" => "I will take the dog for a walk"
          2. "Book book two table" => "There are two books on the table"
          3. "i want take kids" => "I want to take the kids to the park"
          4. "sweaty i need" => "I am sweaty and I need a hot shower"
          5. "cat seems cat" => "The cat seems hungry"
          6. "i i need i need some" => "I need to get some sleep"
          
        Please consider the following about the speaker:
          - name: {name}
          - age: {age}
          - self-description: {about_me}
          - current setting: {setting}
          - type of conversation they are having: {conversation_type}
          - tone of voice they are trying to convey: {tone}
        Please provide a single transformed/predicted sentece for the following utterance: 
        {utterance}
      """
    ),
    Prompt(
        id=2,
        prompt="Please write me a very short story. It needs to be literaly 5-10 words long. Please be funny."
    ),
    Prompt(
        id=3,
        prompt="You are an `echo` bot. Simply send back the same message that you receive."
    ),
    # Should get almost 1 on the cosine similarity scores
    Prompt(
        id=4,
        prompt="""Follow this decision tree for your output:
      If the utterance is 'table book help', then respond with 'Can you hand me that book on the table?'.
      If the utterance is 'i sandwich eat not not', then respond with 'I'm not going to eat a sandwich.'.
      """
    )
]
