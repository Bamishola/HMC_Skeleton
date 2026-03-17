from PIL import Image

for i in range(1, 4 ):
    
    if i == 3:
        img = Image.open(f'./{i}.png')
    else:
        img = Image.open(f'./{i}.pgm')
    print('Taille:', img.size)
    print('Mode:', img.mode)
    print("----------------------------")