def randomIdFunc(stringLength=4):
    import random
    import string
    characters = string.ascii_letters + string.digits
    
    return ''.join(random.choices(characters, k=stringLength))