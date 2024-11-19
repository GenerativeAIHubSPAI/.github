#Basic LM invocation and prediction code example
#dspy==2.5.31

import dspy
import os

turbo = dspy.LM(model=os.getenv('OPENAI_MODEL_NAME'),
                api_key=os.getenv('OPENAI_API_KEY'),
                api_base=os.getenv('OPENAI_ENDPOINT'))

dspy.configure(lm=turbo)

class BasicQA(dspy.Signature):
  """Answer questions with short factoid answers"""

  question = dspy.InputField()
  answer = dspy.OutputField(desc="often between 1 and 5 words",
                            prefix="Question’s Answer:")


generate_response  = dspy.Predict(BasicQA)

pred = generate_response(question="When was the last Solar Eclipse in the United States, and what states were covered in total darkness?")
print(f"Answer: {pred.answer}")
