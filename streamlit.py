import streamlit as st
#import os
#import keras
import keras_nlp
from keras_nlp import tokenizers

class ModelInterface(object):
    def __init__(self):
        self.instruction = """
            You are an AI agent tasked to answer nutrition questions in
            a simple and short way.
        """
        self.path_to_model = "kaggle:/favouryahdii/gemma-nutritionx/keras/gemma-nutritionx-2b"
        self.max_new_tokens = 128
        self.initialize_model()

    def initialize_model(self):
        #start_time = time()
        self.tokenizer = tokenizers.GemmaTokenizer.from_preset(self.path_to_model)
        #tok_time = time()
        # print(f"Load tokenizer: {round(tok_time-start_time, 1)} sec.")
        self.model = keras_nlp.models.GemmaCausalLM.from_preset(
            self.path_to_model,
        )
        #mod_time = time()
        # print(f"Load model: {round(mod_time-tok_time, 1)} sec.")

    def prompt(self, input_text):
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        prompt_text = template.format(
            instruction=input_text,
            response="",
        )
        return prompt_text

    def get_message_response(self, input_text):
        #start_time = time()
        prompt_text = self.prompt(input_text)
        outputs = self.model.generate(
            prompt_text,
            max_length=self.max_new_tokens,
        )
        #end_time = time()
        answer = outputs
        ##print(f"Total response time: {round(end_time-start_time, 1)} sec.")
        return {
            "input": input_text,
            "response": answer,
            #"response_time": f"{round(end_time-start_time, 1)} sec."
        }


# Show title and description.
st.title("💬 NutritionX")
st.write(
    "This is a simple chatbot that uses Google's gemma model to provide answers to Nutrition based questions."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "model_interface" not in st.session_state:
    st.session_state.model_interface = ModelInterface()

model_interface = st.session_state.model_interface

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("Ask any question"):

    # Store and display the current prompt.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a response using the ModelInterface.
    
    response_data = model_interface.get_message_response(prompt)
    response = response_data["response"]

    # Stream the response to the chat using `st.write`, then store it in session state.
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
