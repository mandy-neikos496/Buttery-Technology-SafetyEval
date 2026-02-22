import json

class Chrysos:
    def __init__(self, name, role):
        self.name = name
        self.role = role
    
    def get_info(self):
        return f"Member: {self.name} | Role: {self.role}"

def process_team():
    try:
        with open('data.json', 'r') as file:
            data = json.load(file)

        print("--- Report ---")
        for item in data:
            member = Chrysos(item['name'], item['role'])
            print(member.get_info())

    except FileNotFoundError:
        print("Error: the data.json file was not found :/")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    process_team()