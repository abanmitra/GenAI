from pydantic import BaseModel, ValidationError
import ollama

# Initialize Ollama client
client = ollama.Client()

class Query(BaseModel):
    text: str
    

def process_query(data: dict) -> str:
    try:
        # Validate input data using Pydantic model
        query = Query(**data)

        # Generate response using DeepSeek-R1 via Ollama
        response = client.generate(
            model="deepseek-r1:14b",
            prompt=query.text,
        )
        # if different from the default model
        # response = client.generate(
        #     model="deepseek-r1",
        #     messages=[{
        #         "role": "user",
        #         "content": query.text
        #     }],
        #     temperature=query.temperature,
        #     max_tokens=query.max_tokens
        # )

        return response.choices[0].message.content

    except ValidationError as e:
        raise ValueError(f"Invalid input: {e}")

if __name__ == "__main__":
    # Example usage
    sample_input = {
        "text": "Explain quantum computing to me in simple terms.",
        "temperature": 0.7,
        "max_tokens": 512
    }

    result = process_query(sample_input)
    print("Response:", result)