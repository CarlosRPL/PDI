from PIL import Image

#abrir e converter para escala de cinza
img = Image.open('imagens/imagem.jpg').convert('L')
pixels = list(img.getdata())          # lista com todos os valores (0–255)
largura, altura = img.size
total = largura * altura

histograma = [0] * 256
for p in pixels:
    histograma[p] += 1

cdf = [0] * 256
cdf[0] = histograma[0]
for i in range(1, 256):
    cdf[i] = cdf[i - 1] + histograma[i]
#acha o pixel logo depois do 0 
cdf_min = next(v for v in cdf if v > 0)

#normalizando
tabela = [0] * 256
for i in range(256):
    tabela[i] = round((cdf[i] - cdf_min) / (total - cdf_min) * 255)

#remapeando
pixels_novos = [tabela[p] for p in pixels]

img_nova = Image.new('L', (largura, altura))
img_nova.putdata(pixels_novos)
img_nova.save('imagens/equalizada.jpg')
img_nova.show()