"""
infer_ascii_cnn.py
Usa o modelo CNN treinado (best_model.pt) pra classificar cada bloco 8x8
da imagem e gerar o ASCII art renderizado -- mesmo pipeline do a.py
(grid, carregamento, renderização), só que a escolha do glifo agora vem
da rede em vez de correlação (ZNCC).

Importante: a DoG de cada bloco é calculada ISOLADA (borda replicada
dentro do próprio bloco 8x8), igual ao dog_8x8() do gen_dataset_v3.cpp
usado pra treinar -- e não a partir da DoG da imagem inteira, que usa
vizinhos reais de blocos adjacentes e teria uma distribuição diferente
do que a rede viu no treino.

Requer: pip install torch numpy pillow
"""

from pathlib import Path
import numpy as np
import torch
from hufman_acc import save_acc
from PIL import Image

from a import (
    GLYPHS_RAW,
    bits_to_block,
    compute_grid_from_original,
    compute_dog,
    load_and_prepare,
    render_charmap_to_image,
    EXTENSOES_VALIDAS,
)
from cnn import AsciiCNN, load_label_names
FLAT_THRESHOLD = 0.025

def find_flat_class_indices(names):
    """Índices das classes 'sem estrutura' (delicacy=0 no C++): espaço e bloco cheio."""
    flat_idx = {}
    for idx, ch in names.items():
        block = bits_to_block(GLYPHS_RAW[ch])
        if block.std() < 1e-4:
            flat_idx[idx] = ch
    return flat_idx


def match_glyph_cnn(intensity_block, model, names, device, flat_idx):
    std = intensity_block.std()

    # Bloco genuinamente liso -> decide por brilho direto, sem CNN
    if std < FLAT_THRESHOLD:
        mean = intensity_block.mean()
        # entre as classes flat, escolhe a de brilho médio mais próximo
        candidates = {idx: bits_to_block(GLYPHS_RAW[ch]).mean()
                      for idx, ch in flat_idx.items()}
        best_idx = min(candidates, key=lambda i: abs(candidates[i] - mean))
        return names[best_idx]

    # Bloco tem textura real -> deixa a CNN escolher, mas SEM as classes flat
    dog_block = compute_dog(intensity_block)
    x = np.stack([
        block_to_model_input(intensity_block, "intensity"),
        block_to_model_input(dog_block, "dog"),
    ], axis=0)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x).squeeze(0)          # (num_classes,)
        out = out.clone()
        for idx in flat_idx:
            out[idx] = -float("inf")       # remove espaço/bloco da disputa
        pred_idx = out.argmax().item()

    return names[pred_idx]

# ---------------------------------------------------------------------
# 1. Carregamento do modelo
# ---------------------------------------------------------------------

def load_model(model_path="best_model.pt", labels_path="labels.txt"):
    names = load_label_names(labels_path)          # {idx: char}
    num_classes = len(names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AsciiCNN(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, names, device


# ---------------------------------------------------------------------
# 2. Pré-processamento de bloco -- consistente com gen_dataset_v3.cpp
# ---------------------------------------------------------------------

def block_to_model_input(block, kind="intensity"):
    """Replica exatamente a escala usada na geração do dataset.bin:
    - intensidade: só clip em [0,1] (o C++ nunca faz min-max, só clamp)
    - DoG: remapeada como d + 0.5 e clampada, igual ao dog_8x8() do C++
    """
    if kind == "intensity":
        return np.clip(block, 0.0, 1.0).astype(np.float32)
    else:  # dog
        return np.clip(block + 0.5, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------
# 3. Charmap via CNN
# ---------------------------------------------------------------------
def image_to_charmap_cnn(path, model, names, device):
    with Image.open(path) as im:
        orig_w, orig_h = im.size

    cols, rows_out = compute_grid_from_original(orig_w, orig_h)
    intensity_img, _ = load_and_prepare(path, cols, rows_out)
    flat_idx = find_flat_class_indices(names)

    charmap = []
    for r in range(rows_out):
        row_chars = []
        for c in range(cols):
            i_block = intensity_img[r*8:(r+1)*8, c*8:(c+1)*8]
            ch = match_glyph_cnn(i_block, model, names, device, flat_idx)
            row_chars.append(ch)
        charmap.append(row_chars)

    return charmap, cols, rows_out

def build_render_glyphs(names):
    """Monta só o necessário pra renderização (bitmap de cada char usado
    pelo modelo), sem depender da ordem/quantidade fixa do ORDER do a.py."""
    chars = set(names.values())
    return {ch: {"block": bits_to_block(GLYPHS_RAW[ch])} for ch in chars}


# ---------------------------------------------------------------------
# 4. Processamento em lote de um diretório
# ---------------------------------------------------------------------

def process_directory_cnn(input_dir=".", output_dir="s_cnn",
                           model_path="best_model.pt", labels_path="labels.txt"):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model, names, device = load_model(model_path, labels_path)
    glyphs = build_render_glyphs(names)

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
            charmap, cols, rows_out = image_to_charmap_cnn(img_path, model, names, device)
            out_img = render_charmap_to_image(charmap, glyphs)

            out_name = f"{img_path.stem}_cnn.png"
            out_img.save(out_path / out_name)
            print(f"  -> {out_img.width}x{out_img.height}px salvo em {out_path / out_name}")
            acc_name = f"{img_path.stem}_cnn.acc"
            size = save_acc(out_path / acc_name, charmap)
            print(f"  -> {out_img.width}x{out_img.height}px salvo em {out_path / out_name} "
                  f"| .acc: {size} bytes")
        except Exception as e:
            print(f"  Erro ao processar {img_path.name}: {e}")

    print("\nConcluído.")


if __name__ == "__main__":
    process_directory_cnn(input_dir=".", output_dir="s_cnn")
