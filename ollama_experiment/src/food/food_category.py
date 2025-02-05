import ollama
import os
import write_to_msword as wtf

def read_file(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"File '{file_path}' does not exist.")
            exit(1)
            
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
    return content

def prepare_prompt(items):
    prompt = f"""
    You are a food expert. Yuor are an assistant that categorizes food items into different categories and sorts them alphabetically.

    Here is a list of food items:
    {items}

    Please:
     1. categorize the food items into appropriate categories such as "fruits", "vegetables", "grains", "proteins", "dairy", "fats", "sweets", "snacks", "beverages", "condiments", "spices", "herbs", "other".
     2. sort the categories alphabetically within each category.
     3. Present the catgorized list in a clear and oraganize manner, using bullet points or numbered lists.
    """

    return prompt

def category_food_items(food_items):

    items = "\n".join(food_items)
    prompt = prepare_prompt(items)

    model="llama3.2"
    response = ollama.generate(model=model, prompt=prompt)
    return response.response
    
def write_to_file(file_path, content):
    try:
        #with open(file_path, 'w', encoding='utf-8') as file:
        #    file.write(content.strip())
        wtf.write_to_msword(file_path, content)

        print(f"Content written to '{file_path}' successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():

    input_file="D:/work/GPT/ollama_experiment/data/food_list.txt"
    output_file="D:/work/GPT/ollama_experiment/data/food_category.docx"


    items = read_file(input_file)
    response = category_food_items(items)
    write_to_file(output_file, response)

if __name__ == "__main__":
    main()