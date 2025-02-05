import ollama


# model_name="phi"
# model_details = ollama.show(model_name)
# modelfile_content = model_details['modelfile']
# with open('ocen.modelfile', 'w') as f:
#     f.write(modelfile_content)




import base64

# Read the file as binary and encode it in base64
with open('ocen.modelfile', 'rb') as f:
    modelfile_content = base64.b64encode(f.read()).decode('utf-8')  # Convert binary data to base64 string

# Create the model with the base64-encoded content
response = ollama.create(model='myocen', files={'modelfile': modelfile_content})






# delete model
# response = ollama.delete(model='myocen')


print(response)