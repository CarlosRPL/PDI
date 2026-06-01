from PIL import Image
import math
#normalizando logo 
def fdp_alvo(i):
    return (math.exp(i / 255) - 1) / (math.e - 1)
img = Image.open('imagens/equalizada.jpg').convert('L')
pixels = list(img.getdata())
largura, altura = img.size

hist_alvo = [fdp_alvo(i) for i in range(256)]
soma = sum(hist_alvo)
hist_alvo = [v / soma for v in hist_alvo]

cdf_alvo = [0.0] * 256
cdf_alvo[0] = hist_alvo[0]
for i in range(1, 256):
    cdf_alvo[i] = cdf_alvo[i - 1] + hist_alvo[i]

cdf_alvo_inv = [0] * 256
for u_idx in range(256):
    u = u_idx / 255.0
    melhor_j, menor_diff = 0, float('inf')
    for j in range(256):
        diff = abs(cdf_alvo[j] - u)
        if diff < menor_diff:
            menor_diff = diff
            melhor_j = j
    cdf_alvo_inv[u_idx] = melhor_j

pixels_novos = [cdf_alvo_inv[p] for p in pixels]

img_final = Image.new('L', (largura, altura))
img_final.putdata(pixels_novos)
img_final.save('imagens/imagem_final.jpg')
img_final.show()