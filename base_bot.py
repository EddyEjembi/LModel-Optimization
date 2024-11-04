from transformers import AutoModelForCausalLM
from chatbot import Pirate

# Load Model
model = AutoModelForCausalLM.from_pretrained("<PATH-TO-MODEL>")

# Initialize the Pirate Class
pirate = Pirate(model)

# Test the chatbot
while True:
    input_text = input("Enter your Message: ")
    response = pirate.generate(input_text)
    print(f"Jack Sparrow 🏴‍☠️: {response}")
