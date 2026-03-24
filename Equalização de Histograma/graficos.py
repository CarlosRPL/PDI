from PIL import Image, ImageDraw

img = Image.open('imagem_final.jpg').convert('L')
pixels = list(img.getdata())

histograma = [0] * 256
for p in pixels:
    histograma[p] += 1

graf_w, graf_h = 512, 300
margem = 40

img_graf = Image.new('RGB', (graf_w + margem, graf_h + margem * 2), (255, 255, 255))
draw = ImageDraw.Draw(img_graf)

draw.line([(margem, margem), (margem, graf_h + margem)], fill=(0, 0, 0), width=2)
draw.line([(margem, graf_h + margem), (graf_w + margem, graf_h + margem)], fill=(0, 0, 0), width=2)

max_val = max(histograma)
barra_w = graf_w / 256

for i, valor in enumerate(histograma):
    barra_h = int((valor / max_val) * graf_h) if max_val > 0 else 0
    x0 = int(margem + i * barra_w)
    x1 = int(margem + (i + 1) * barra_w)
    y0 = graf_h + margem - barra_h
    y1 = graf_h + margem
    draw.rectangle([x0, y0, x1, y1], fill=(80, 80, 80))

for v in [0, 64, 128, 192, 255]:
    x = int(margem + v * (graf_w / 256))
    draw.line([(x, graf_h + margem), (x, graf_h + margem + 5)], fill=(0, 0, 0))
    draw.text((x - 8, graf_h + margem + 7), str(v), fill=(0, 0, 0))

img_graf.save('final.jpg')
img_graf.show()