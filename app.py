from transformers import pipeline
import gradio as gr

# Initialize the text generation pipeline
text_generator = pipeline(model="facebook/blenderbot-400M-distill")

def vanilla_chatbot(message, history):
    # Generate text based on the message
    generated_text = text_generator(message, max_length=50, num_return_sequences=1)
    
    return generated_text[0]['generated_text']

# Create the Gradio interface
demo_chatbot = gr.ChatInterface(vanilla_chatbot, title="James", description="Enter text to start chatting.")

demo_chatbot.launch(share=True)
