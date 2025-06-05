def find_start_end(text, entity):
    start = text.find(entity)
    end = start + len(entity)
    return start, end

text = "El paciente tiene dolor de cabeza"

entity = "dolor de cabeza"

# we need to find the char start and end index of the entity in the text
start, end = find_start_end(text, entity)
print(start, end)