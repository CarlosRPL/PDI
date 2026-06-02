"""
pipeline1_debug.py
Pipeline 1 (contraste -> suavizacao -> DoG -> Sobel -> NMS -> histerese
-> remocao de fragmentos -> fechamento morfologico direcional ->
agregacao por bloco) processando UMA imagem, salvando uma imagem de
cada etapa intermediaria (pasta de saida), e gerando 3 renders finais:

  <nome>_00_original.png
  <nome>_01_suavizada.png
  <nome>_02_contraste.png
  <nome>_03_dog.png
  <nome>_04_sobel_magnitude.png
  <nome>_05_nms.png
  <nome>_06_histerese.png
  <nome>_07_fragmentos_removidos.png
  <nome>_08_fechamento_direcional.png
  <nome>_apenas_densidade.png   (ascii so por brilho/preenchimento)
  <nome>_apenas_bordas.png      (ascii so com as bordas detectadas)
  <nome>_final_completo.png     (ascii combinando as duas camadas)

Uso:
    python pipeline1_debug.py caminho/para/imagem.png [pasta_saida] [paleta]

    paleta: 10, 20 ou 40 (default 20)
"""

import sys
import os
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, convolve, binary_closing, label, median_filter

def cor_para_brilho(bri):
    """Mapeia o brilho (0-255) do bloco original para a mesma paleta de
    tons usada na densidade -- assim a borda herda a cor da regiao que
    esta substituindo, em vez de uma cor fixa."""
    t = bri / 255.0
    t = max(0.0, min(1.0, t))
    idx = int(round(t * (len(CORES) - 1)))
    return CORES[idx]
# ---------------------------------------------------------------------
# Glifos (mesmo alfabeto do al.py)
# ---------------------------------------------------------------------

FILL_BITMAPS = {
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
    '0':  [0x3C,0x66,0x6E,0x76,0x66,0x66,0x3C,0x00],
    'O':  [0x3C,0x42,0x42,0x42,0x42,0x42,0x3C,0x00],
    'B':  [0x78,0x44,0x44,0x78,0x44,0x44,0x78,0x00],
    '#':  [0x24,0x24,0x7E,0x24,0x24,0x7E,0x24,0x24],
    '@':  [0x3C,0x42,0x5A,0x5E,0x5E,0x40,0x3C,0x00],
    '█':  [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF],
}

DENSIDADES = {}
for ch, bmp in FILL_BITMAPS.items():
    bits = sum(bin(byte).count('1') for byte in bmp)
    DENSIDADES[ch] = bits

EDGE_BITMAPS = {
    '-':  [0x00,0x00,0x00,0xFF,0xFF,0x00,0x00,0x00],
    '/':  [0x01,0x03,0x06,0x0C,0x18,0x30,0x60,0xC0],
    '\\': [0xC0,0x60,0x30,0x18,0x0C,0x06,0x03,0x01],
    '|':  [0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x18],
}

CORES = [
    (88, 68, 70), (102, 78, 80), (116, 88, 90), (130, 98, 100),
    (144, 108, 110), (158, 120, 122), (174, 136, 138), (190, 154, 156),
]
BG = (0, 0, 0)
EDGE_COLOR =  (88, 68, 70)


def criar_paleta(tamanho):
    chars = sorted(FILL_BITMAPS.keys(), key=lambda c: DENSIDADES[c])
    if chars[0] != ' ' or chars[-1] != '█':
        chars = [' '] + [c for c in chars if c not in (' ', '█')] + ['█']
    if tamanho >= len(chars):
        return chars
    indices = [0] + list(np.linspace(1, len(chars) - 2, tamanho - 2, dtype=int)) + [len(chars) - 1]
    return [chars[i] for i in indices]


# ---------------------------------------------------------------------
# Etapas do pipeline (copiadas do al.py)
# ---------------------------------------------------------------------

def aumentar_contraste_forte(img_f, cutoff=0):
    im = Image.fromarray(np.clip(img_f, 0, 255).astype(np.uint8), 'L')
    im = ImageOps.equalize(im)
    im = ImageOps.autocontrast(im, cutoff=cutoff)
    return np.array(im, dtype=np.float32)


def suavizar_pre_dog(img_f):
    return median_filter(img_f, size=5, mode="reflect")


def dog(img_f, sigma1=3.0, k=1.6):
    sigma2 = sigma1 * k
    g1 = gaussian_filter(img_f, sigma1)
    g2 = gaussian_filter(img_f, sigma2)
    return g1 - g2


def sobel(img_f):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    Gx = convolve(img_f, kx)
    Gy = convolve(img_f, ky)
    return np.hypot(Gx, Gy), np.arctan2(Gy, Gx)


def non_max_suppression(mag, ang):
    deg = np.degrees(ang) % 180
    viz_0_a = np.roll(mag, 1, axis=1)
    viz_0_b = np.roll(mag, -1, axis=1)
    viz_45_a = np.roll(np.roll(mag, -1, axis=0), 1, axis=1)
    viz_45_b = np.roll(np.roll(mag, 1, axis=0), -1, axis=1)
    viz_90_a = np.roll(mag, 1, axis=0)
    viz_90_b = np.roll(mag, -1, axis=0)
    viz_135_a = np.roll(np.roll(mag, 1, axis=0), 1, axis=1)
    viz_135_b = np.roll(np.roll(mag, -1, axis=0), -1, axis=1)

    bin0 = (deg < 22.5) | (deg >= 157.5)
    bin45 = (deg >= 22.5) & (deg < 67.5)
    bin90 = (deg >= 67.5) & (deg < 112.5)
    bin135 = (deg >= 112.5) & (deg < 157.5)

    maximo_local = np.zeros_like(mag, dtype=bool)
    maximo_local |= bin0 & (mag >= viz_0_a) & (mag >= viz_0_b)
    maximo_local |= bin45 & (mag >= viz_45_a) & (mag >= viz_45_b)
    maximo_local |= bin90 & (mag >= viz_90_a) & (mag >= viz_90_b)
    maximo_local |= bin135 & (mag >= viz_135_a) & (mag >= viz_135_b)
    return np.where(maximo_local, mag, 0.0)


def otsu_threshold(mag):
    hist, bin_edges = np.histogram(mag, bins=256)
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    centros = (bin_edges[:-1] + bin_edges[1:]) / 2
    mu = np.cumsum(prob * centros)
    mu_t = mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    sigma_b2 = np.nan_to_num(sigma_b2)
    return bin_edges[np.argmax(sigma_b2)]


def hysteresis_threshold(mag, low_ratio=0.5, high_ratio=1.2):
    alto_base = otsu_threshold(mag)
    high = alto_base * high_ratio
    low = alto_base * low_ratio
    forte = mag >= high
    fraca = (mag >= low) & (mag < high)
    candidatos = forte | fraca
    rotulado, _ = label(candidatos, structure=np.ones((3, 3)))
    rotulos_com_forte = set(rotulado[forte])
    rotulos_com_forte.discard(0)
    return np.isin(rotulado, list(rotulos_com_forte))


def remover_fragmentos_pequenos(mask, area_minima=6):
    rotulado, n = label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    tamanhos = np.bincount(rotulado.ravel())
    mask_valida = tamanhos >= area_minima
    mask_valida[0] = False
    return mask_valida[rotulado]


def ang_para_char_vec(rad):
    deg = np.degrees((rad + np.pi / 2) % np.pi)
    out = np.full(deg.shape, '-', dtype='<U1')
    out[(deg >= 22.5) & (deg < 67.5)] = '/'
    out[(deg >= 67.5) & (deg < 112.5)] = '|'
    out[(deg >= 112.5) & (deg < 157.5)] = '\\'
    return out


def criar_estruturas(bloco):
    tam = bloco + 1
    estrutura_h = np.zeros((tam, tam), dtype=bool)
    estrutura_h[tam // 2, :] = True
    estrutura_v = estrutura_h.T.copy()
    estrutura_diag = np.eye(tam, dtype=bool)
    estrutura_antidiag = np.fliplr(estrutura_diag)
    return {'-': estrutura_h, '|': estrutura_v, '\\': estrutura_diag, '/': estrutura_antidiag}


def block_sum(arr2d, bloco):
    H, W = arr2d.shape
    NL, NC = H // bloco, W // bloco
    arr2d = arr2d[:NL * bloco, :NC * bloco]
    return arr2d.reshape(NL, bloco, NC, bloco).sum(axis=(1, 3))


def agregar_por_bloco(mask, direcoes, mag, contrastada, bloco, ocupacao_minima=0.1):
    contagem = block_sum(mask.astype(np.float32), bloco)
    grid_mask = contagem >= ocupacao_minima * bloco * bloco

    chars_dir = ['-', '|', '/', '\\']
    somas = np.stack([
        block_sum(mag * (mask & (direcoes == ch)), bloco) for ch in chars_dir
    ])
    idx_dominante = np.argmax(somas, axis=0)
    grid_dir = np.array(chars_dir)[idx_dominante]
    grid_dir = np.where(grid_mask, grid_dir, ' ')

    soma_bri = block_sum(contrastada, bloco)
    grid_bri = soma_bri / (bloco * bloco)
    return grid_mask, grid_dir, grid_bri


def bri_para_char(bri, paleta):
    t = bri / 255.0
    idx = int(t * (len(paleta) - 1))
    return paleta[max(0, min(idx, len(paleta) - 1))]


def cor_para_densidade(densidade):
    dmin = min(DENSIDADES.values())
    dmax = max(DENSIDADES.values())
    t = (densidade - dmin) / (dmax - dmin)
    t = max(0.0, min(1.0, t))
    idx = int(round(t * (len(CORES) - 1)))
    return CORES[idx]


def draw_char(arr, ch, bx, by, bloco, color, bitmaps):
    bmp = bitmaps.get(ch)
    if bmp is None:
        return
    r, g, b = color
    for row in range(bloco):
        byte = bmp[row]
        py = by * bloco + row
        for col in range(bloco):
            if byte & (0x80 >> col):
                px = bx * bloco + col
                arr[py, px, 0] = r
                arr[py, px, 1] = g
                arr[py, px, 2] = b


# ---------------------------------------------------------------------
# Helpers de salvamento de imagens de debug
# ---------------------------------------------------------------------

def salvar_cinza(arr, path):
    """Normaliza qualquer array float (DoG, magnitude, etc.) para 0-255
    antes de salvar, já que essas etapas podem ter valores negativos
    ou fora da faixa de exibição direta."""
    a = arr.astype(np.float64)
    a = a - a.min()
    maxv = a.max()
    if maxv > 1e-8:
        a = a / maxv * 255.0
    Image.fromarray(a.astype(np.uint8), 'L').save(path)


def salvar_mask(mask, path):
    img = np.where(mask, 255, 0).astype(np.uint8)
    Image.fromarray(img, 'L').save(path)


def salvar_mask_colorida_por_direcao(mask, direcoes, path):
    """Visualiza a mascara de bordas colorindo cada direcao diferente,
    pra facilitar ver o efeito do fechamento morfologico direcional."""
    cores_dir = {
        '-': (255, 80, 80),
        '|': (80, 255, 80),
        '/': (80, 80, 255),
        '\\': (255, 255, 80),
        ' ': (0, 0, 0),
    }
    H, W = mask.shape
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for ch, cor in cores_dir.items():
        if ch == ' ':
            continue
        sel = mask & (direcoes == ch)
        arr[sel] = cor
    Image.fromarray(arr, 'RGB').save(path)


# ---------------------------------------------------------------------
# Processamento principal (com salvamento de cada etapa)
# ---------------------------------------------------------------------

def processar_com_debug(entrada, pasta_saida, bloco=8, paleta_tamanho=20):
    os.makedirs(pasta_saida, exist_ok=True)
    nome_base = os.path.splitext(os.path.basename(entrada))[0]

    def caminho(sufixo):
        return os.path.join(pasta_saida, f"{nome_base}_{sufixo}.png")

    # ---- 00: carregamento em escala de cinza ----
    img = Image.open(entrada).convert('L')
    img_f = np.array(img, dtype=np.float32)
    H, W = img_f.shape
    NL, NC = H // bloco, W // bloco
    if NL == 0 or NC == 0:
        print(f"Imagem pequena demais pro bloco={bloco}")
        return
    img_f = img_f[:NL * bloco, :NC * bloco]
    salvar_cinza(img_f, caminho("00_original"))

    # ---- 01: suavizacao (mediana, pre-DoG) ----
    suv = suavizar_pre_dog(img_f)
    salvar_cinza(suv, caminho("01_suavizada"))

    # ---- 02: contraste (equalizacao + autocontraste) ----
    contrastada = aumentar_contraste_forte(img_f)
    salvar_cinza(contrastada, caminho("02_contraste"))

    # ---- 03: DoG ----
    filtrada = dog(suv)
    salvar_cinza(filtrada, caminho("03_dog"))

    # ---- 04: Sobel (magnitude) sobre a DoG ----
    mag, ang = sobel(filtrada)
    salvar_cinza(mag, caminho("04_sobel_magnitude"))

    # ---- 05: supressao nao-maxima ----
    mag_nms = non_max_suppression(mag, ang)
    salvar_cinza(mag_nms, caminho("05_nms"))

    # ---- 06: limiarizacao por histerese (Otsu) ----
    mask_bruta = hysteresis_threshold(mag_nms)
    salvar_mask(mask_bruta, caminho("06_histerese"))

    # ---- 07: remocao de fragmentos pequenos ----
    mask_limpa = remover_fragmentos_pequenos(mask_bruta)
    salvar_mask(mask_limpa, caminho("07_fragmentos_removidos"))

    # ---- 08: fechamento morfologico direcional ----
    direcao = ang_para_char_vec(ang)
    estruturas = criar_estruturas(bloco)
    final_mask = np.zeros_like(mask_limpa)
    final_dir = np.full(mask_limpa.shape, ' ', dtype='<U1')
    for ch, estrutura in estruturas.items():
        dir_mask = mask_limpa & (direcao == ch)
        fechado = binary_closing(dir_mask, structure=estrutura)
        livre = fechado & (final_dir == ' ')
        final_dir[livre] = ch
        final_mask |= fechado
    salvar_mask(final_mask, caminho("08_fechamento_direcional"))
    salvar_mask_colorida_por_direcao(final_mask, final_dir, caminho("08b_fechamento_por_direcao_colorido"))

    # ---- agregacao por bloco 8x8 ----
    grid_mask, grid_dir, grid_bri = agregar_por_bloco(final_mask, final_dir, mag, contrastada, bloco)

    paleta = criar_paleta(paleta_tamanho)

    # ---------------------------------------------------------------
    # Render 1: APENAS DENSIDADE (ignora completamente a camada de borda)
    # ---------------------------------------------------------------
    arr_dens = np.zeros((NL * bloco, NC * bloco, 3), dtype=np.uint8)
    arr_dens[:] = BG
    for by in range(NL):
        for bx in range(NC):
            bri = float(grid_bri[by, bx])
            ch_f = bri_para_char(bri, paleta)
            if ch_f != ' ':
                cor_fill = cor_para_densidade(DENSIDADES[ch_f])
                draw_char(arr_dens, ch_f, bx, by, bloco, cor_fill, FILL_BITMAPS)
    Image.fromarray(arr_dens, 'RGB').save(caminho("apenas_densidade"))

# ---------------------------------------------------------------
    # Render 2: APENAS BORDAS (so desenha onde grid_mask == True)
    # ---------------------------------------------------------------
    arr_bordas = np.zeros((NL * bloco, NC * bloco, 3), dtype=np.uint8)
    arr_bordas[:] = BG
    for by in range(NL):
        for bx in range(NC):
            if grid_mask[by, bx]:
                cor_edge = cor_para_brilho(float(grid_bri[by, bx]))
                draw_char(arr_bordas, grid_dir[by, bx], bx, by, bloco, cor_edge, EDGE_BITMAPS)
    Image.fromarray(arr_bordas, 'RGB').save(caminho("apenas_bordas"))

    # ---------------------------------------------------------------
    # Render 3: FINAL COMPLETO (borda onde ha borda, densidade no resto)
    # ---------------------------------------------------------------
    arr_final = np.zeros((NL * bloco, NC * bloco, 3), dtype=np.uint8)
    arr_final[:] = BG
    for by in range(NL):
        for bx in range(NC):
            if grid_mask[by, bx]:
                cor_edge = cor_para_brilho(float(grid_bri[by, bx]))
                draw_char(arr_final, grid_dir[by, bx], bx, by, bloco, cor_edge, EDGE_BITMAPS)
            else:
                bri = float(grid_bri[by, bx])
                ch_f = bri_para_char(bri, paleta)
                if ch_f != ' ':
                    cor_fill = cor_para_densidade(DENSIDADES[ch_f])
                    draw_char(arr_final, ch_f, bx, by, bloco, cor_fill, FILL_BITMAPS)
    Image.fromarray(arr_final, 'RGB').save(caminho("final_completo"))
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python pipeline1_debug.py caminho/imagem.png [pasta_saida] [paleta]")
        sys.exit(1)

    entrada = sys.argv[1]
    pasta_saida = sys.argv[2] if len(sys.argv) > 2 else "debug_pipeline1"
    paleta = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    processar_com_debug(entrada, pasta_saida, bloco=8, paleta_tamanho=paleta)
