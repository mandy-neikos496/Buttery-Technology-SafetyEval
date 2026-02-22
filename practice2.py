import json

class Characters:
    def __init__(self, name, path, element, role, age):
        self.name = name
        self.path = path
        self.element = element
        self.role = role
        self.age = age
    
    def get_info(self):
        return f"Name: {self.name} | Path: {self.path} | Element: {self.element} | Role: {self.role} | Age: {self.age}"
    
def process_characters():
    try:
        with open('data2.json', 'r') as file:
            data = json.load(file)

        print("--- Best HSR Characters ---")
        for item in data:
            char = Characters(item['name'], item['path'], item['element'], item['role'], item['age'])
            print(char.get_info())

    except FileNotFoundError:
        print("Error: the data2.json file was not found lmao")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    process_characters()