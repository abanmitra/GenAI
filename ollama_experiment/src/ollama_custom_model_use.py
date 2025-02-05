import ollama

model_name='myocen'

prompt = """
who are you? what is your name?
""".strip()

response = ollama.generate(model=model_name, prompt=prompt)

output = response.get('response')
print(output.strip())