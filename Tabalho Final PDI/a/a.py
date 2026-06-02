from pathlib import Path
from PIL import Image
from hufman_acc import save_acc
import numpy as np


GLYPHS_RAW = {
    ' ':  [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00],
    '.':  [0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18],
    '·':  [0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x00],
    '`':  [0x10,0x08,0x00,0x00,0x00,0x00,0x00,0x00],
    ',':  [0x00,0x00,0x00,0x00,0x00,0x18,0x08,0x10],
    ':':  [0x00,0x00,0x18,0x18,0x00,0x18,0x18,0x00],
    ';':  [0x00,0x00,0x18,0x18,0x00,0x18,0x08,0x10],
    '!':  [0x18,0x18,0x18,0x18,0x18,0x00,0x18,0x00],
    "'":  [0x08,0x10,0x00,0x00,0x00,0x00,0x00,0x00],
    '-':  [0x00,0x00,0x00,0x00,0x7E,0x00,0x00,0x00],
    '~':  [0x00,0x00,0x00,0x32,0x4C,0x00,0x00,0x00],
    '+':  [0x00,0x00,0x18,0x18,0x7E,0x18,0x18,0x00],
    '*':  [0x00,0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00],
    '=':  [0x00,0x00,0x7E,0x00,0x7E,0x00,0x00,0x00],
    '?':  [0x3C,0x42,0x02,0x0C,0x18,0x00,0x18,0x00],
    'c':  [0x00,0x00,0x3C,0x42,0x40,0x42,0x3C,0x00],
    'o':  [0x00,0x00,0x3C,0x42,0x42,0x42,0x3C,0x00],
    'x':  [0x00,0x42,0x24,0x18,0x18,0x24,0x42,0x00],
    'n':  [0x00,0x00,0x5C,0x62,0x42,0x42,0x42,0x00],
    'u':  [0x00,0x00,0x42,0x42,0x42,0x46,0x3A,0x00],
    'P':  [0x78,0x44,0x44,0x78,0x40,0x40,0x40,0x00],
    'R':  [0x78,0x44,0x44,0x78,0x28,0x24,0x42,0x00],
    'M':  [0x3C,0x66,0x6E,0x76,0x66,0x66,0x3C,0x00],
    'O':  [0x3C,0x42,0x42,0x42,0x42,0x42,0x3C,0x00],
    'B':  [0x78,0x44,0x44,0x78,0x44,0x44,0x78,0x00],
    '#':  [0x24,0x24,0x7E,0x24,0x24,0x7E,0x24,0x24],
    '@':  [0x3C,0x42,0x5A,0x5E,0x5E,0x40,0x3C,0x00],
    '█':  [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF],
}

ORDER = [' ', '.', ':', '-', '+', 'o', '#', '█',
         '`', '*', 'x', 'c', 'O', '=', '@', '?',
         ',', ';', '!', "'", '~', 'n', 'u', 'P',
         'R', 'M', 'B', '·']

EXTENSOES_VALIDAS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}

# ---------------------------------------------------------------------
# Gaussian blur / DoG genérico (usado tanto na imagem inteira quanto
# nos templates isolados 8x8, com borda replicada -- 'edge')
# ---------------------------------------------------------------------

def gaussian_kernel1d(sigma):
    radius = max(1, int(np.ceil(sigma * 2.5)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return k

def gaussian_blur_2d(arr, sigma):
    """Blur gaussiano separável, borda replicada (edge). Funciona tanto
    pra imagem inteira quanto pra blocos pequenos (8x8)."""
    if sigma <= 0.01:
        return arr.copy()
    k = gaussian_kernel1d(sigma)
    radius = len(k) // 2

    padded = np.pad(arr, radius, mode='edge')
    tmp = np.zeros_like(arr, dtype=np.float64)
    for i, w in enumerate(k):
        tmp += w * padded[radius:radius + arr.shape[0], i:i + arr.shape[1]]

    padded2 = np.pad(tmp, radius, mode='edge')
    out = np.zeros_like(arr, dtype=np.float64)
    for i, w in enumerate(k):
        out += w * padded2[i:i + arr.shape[0], radius:radius + arr.shape[1]]
    return out

def compute_dog(arr, sigma_fine=0.6, sigma_coarse=1.6):
    """DoG = blur fino - blur grosso. Realça estrutura/bordas,
    suprime brilho absoluto e ruído de alta frequência."""
    fine = gaussian_blur_2d(arr, sigma_fine)
    coarse = gaussian_blur_2d(arr, sigma_coarse)
    return fine - coarse  # aprox faixa [-1, 1]


def bits_to_block(rows):
    block = np.zeros((8, 8), dtype=np.float64)
    for y, row in enumerate(rows):
        for x in range(8):
            block[y, x] = (row >> x) & 1
    return block


def build_glyph_set(n_chars):
    """Cada glifo carrega também seu próprio mapa DoG (calculado
    isoladamente, com borda replicada -- consistente com o gerador
    de dataset da rede)."""
    chars = ORDER[:n_chars]
    glyphs = {}
    for ch in chars:
        block = bits_to_block(GLYPHS_RAW[ch])
        mean = block.mean()
        std = block.std()

        dog = compute_dog(block)
        dog_std = dog.std()

        glyphs[ch] = {
            "block": block,
            "mean": mean,
            "std": std,
            "flat": std < 1e-4,         # ' ' e '█' -- sem estrutura de intensidade
            "dog": dog,
            "dog_std": dog_std,
            "dog_flat": dog_std < 1e-4, # também vale pra DoG (ex: ' ' tem DoG ~0)
        }
    return glyphs


def compute_grid_from_original(orig_w, orig_h):
    cols = max(1, round(orig_w / 8))
    rows_out = max(1, round(orig_h / 8))
    return cols, rows_out


def load_and_prepare(path, cols, rows_out):
    """Carrega em escala de cinza, redimensiona (Image.BOX), e já
    calcula o mapa DoG da IMAGEM INTEIRA -- assim os pixels de borda
    de cada bloco 8x8 usam vizinhos REAIS (blocos adjacentes),
    em vez de bordas artificiais."""
    img = Image.open(path).convert("L")
    target_w, target_h = cols * 8, rows_out * 8
    img = img.resize((target_w, target_h), resample=Image.BOX)

    intensity = np.asarray(img, dtype=np.float64) / 255.0
    dog_full = compute_dog(intensity)

    return intensity, dog_full


def normalize_block(block):
    mean, std = block.mean(), block.std()
    if std < 1e-6:
        return np.zeros_like(block), mean, std
    return (block - mean) / std, mean, std


FLAT_THRESHOLD = 0.03
EPS = 1e-6
DOG_WEIGHT = 0.4      # peso do canal DoG na pontuação final (0 = ignora DoG, 1 = só DoG)
INTENSITY_WEIGHT = 1.0 - DOG_WEIGHT

def zncc(block_norm, template, t_mean, t_std):
    template_norm = (template - t_mean) / (t_std + EPS)
    return np.sum(block_norm * template_norm) / 64.0


def match_glyph(intensity_block, dog_block, glyphs):
    mean, std = intensity_block.mean(), intensity_block.std()

    if std < FLAT_THRESHOLD:
        flat_glyphs = {ch: g for ch, g in glyphs.items() if g["flat"]}
        if not flat_glyphs:
            flat_glyphs = glyphs
        return min(flat_glyphs, key=lambda ch: abs(flat_glyphs[ch]["mean"] - mean))

    intensity_norm, _, _ = normalize_block(intensity_block)

    dog_mean, dog_std = dog_block.mean(), dog_block.std()
    dog_norm = np.zeros_like(dog_block) if dog_std < EPS else (dog_block - dog_mean) / dog_std

    structured = {ch: g for ch, g in glyphs.items() if not g["flat"]}
    if not structured:
        structured = glyphs

    best_ch, best_score = None, -np.inf
    for ch, g in structured.items():
        score_intensity = zncc(intensity_norm, g["block"], g["mean"], g["std"])

        if g["dog_flat"] or dog_std < EPS:
            score_dog = 0.0
        else:
            score_dog = zncc(dog_norm, g["dog"], g["dog"].mean(), g["dog_std"])

        score = INTENSITY_WEIGHT * score_intensity + DOG_WEIGHT * score_dog

        if score > best_score:
            best_score, best_ch = score, ch

    return best_ch


def image_to_charmap(path, n_chars):
    with Image.open(path) as im:
        orig_w, orig_h = im.size

    cols, rows_out = compute_grid_from_original(orig_w, orig_h)
    glyphs = build_glyph_set(n_chars)
    intensity_img, dog_img = load_and_prepare(path, cols, rows_out)

    charmap = []
    for r in range(rows_out):
        row_chars = []
        for c in range(cols):
            i_block = intensity_img[r*8:(r+1)*8, c*8:(c+1)*8]
            d_block = dog_img[r*8:(r+1)*8, c*8:(c+1)*8]
            row_chars.append(match_glyph(i_block, d_block, glyphs))
        charmap.append(row_chars)
    return charmap, glyphs, cols, rows_out


def render_charmap_to_image(charmap, glyphs, fg=255, bg=0):
    rows_out = len(charmap)
    cols = len(charmap[0]) if rows_out > 0 else 0

    out_h = rows_out * 8
    out_w = cols * 8
    canvas = np.full((out_h, out_w), bg, dtype=np.uint8)

    for r in range(rows_out):
        for c in range(cols):
            ch = charmap[r][c]
            block = glyphs[ch]["block"]
            pixel_block = np.where(block > 0.5, fg, bg).astype(np.uint8)
            canvas[r*8:(r+1)*8, c*8:(c+1)*8] = pixel_block

    return Image.fromarray(canvas, mode="L")


# ---------------------------------------------------------------------
# 6. Processamento em lote de um diretório
# ---------------------------------------------------------------------

def process_directory(input_dir=".", output_dir="s", sets=(32)):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"Diretório '{input_dir}' não encontrado.")
        return

    imagens = sorted(
        p for p in in_path.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSOES_VALIDAS
    )

    if not imagens:
        print(f"Nenhuma imagem encontrada em '{input_dir}'.")
        return

    print(f"Encontradas {len(imagens)} imagem(ns) em '{input_dir}'.\n")

    for img_path in imagens:
        print(f"Processando: {img_path.name}")
        try:
            for n in sets:
                charmap, glyphs, cols, rows_out = image_to_charmap(img_path, n)
                out_img = render_charmap_to_image(charmap, glyphs)

                out_name = f"{img_path.stem}_set{n}.png"
                out_img.save(out_path / out_name)
                print(f"  -> set {n}: {out_img.width}x{out_img.height}px "
                      f"salvo em {out_path / out_name}")
                acc_name = f"{img_path.stem}_set{n}.acc"
                size = save_acc(out_path / acc_name, charmap)
                print(f"  -> set {n}: {out_img.width}x{out_img.height}px "
                      f"salvo em {out_path / out_name} | .acc: {size} bytes")
        except Exception as e:
            print(f"  Erro ao processar {img_path.name}: {e}")

    print("\nConcluído.")


if __name__ == "__main__":
    process_directory(input_dir=".", output_dir="s", sets=(1,1,32))

