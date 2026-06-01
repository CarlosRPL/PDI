"""
acc_viewer.py
Visualizador de arquivos .acc com janela gráfica (tkinter).
Detecta automaticamente o formato:
  - mono (a.py / rnc.py): símbolos de 1 caractere -- agora renderizado
    com cor por densidade (mesma escala de cores do all2.py), não mais P&B
  - colorido (all2.py): símbolos "E-"/"F." (camada+char), cor reconstruída

Uso:
    python acc_viewer.py              # abre a janela, lista os .acc do diretório atual
    python acc_viewer.py pasta/       # lista os .acc de uma pasta específica
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

from hufman_acc import load_acc
from a import GLYPHS_RAW, bits_to_block
from all2 import (
    FILL_BITMAPS,
    EDGE_BITMAPS,
    DENSIDADES,
    EDGE_COLOR,
    BG,
    CORES,
    cor_para_densidade,
    draw_char,
)


# ---------------------------------------------------------------------
# Densidade + cor para os glifos do a.py/rnc.py (GLYPHS_RAW), na mesma
# escala 0-len(CORES)-1 que o all2.py usa pro preenchimento
# ---------------------------------------------------------------------

def _densidade_glyphs_raw(ch):
    """Conta bits ligados no bitmap 8x8 do a.py -- equivalente ao
    DENSIDADES do all2.py, mas pro alfabeto GLYPHS_RAW (28 símbolos,
    que não é exatamente o mesmo conjunto do FILL_BITMAPS)."""
    bmp = GLYPHS_RAW[ch]
    return sum(bin(byte).count('1') for byte in bmp)


def _cor_para_glyph_raw(ch, dmin, dmax):
    dens = _densidade_glyphs_raw(ch)
    t = (dens - dmin) / (dmax - dmin) if dmax > dmin else 0.0
    t = max(0.0, min(1.0, t))
    idx = int(round(t * (len(CORES) - 1)))
    return CORES[idx]


# ---------------------------------------------------------------------
# Detecção de formato
# ---------------------------------------------------------------------

def eh_formato_colorido(charmap):
    for row in charmap:
        for symbol in row:
            if len(symbol) != 2 or symbol[0] not in ('E', 'F'):
                return False
    return True


# ---------------------------------------------------------------------
# Renderização mono -> agora colorida por densidade
# ---------------------------------------------------------------------

def render_mono_colorido(charmap, bloco=8):
    rows_out = len(charmap)
    cols = len(charmap[0]) if rows_out else 0

    chars_usados = {ch for row in charmap for ch in row}
    densidades = {ch: _densidade_glyphs_raw(ch) for ch in chars_usados}
    dmin, dmax = min(densidades.values()), max(densidades.values())

    arr = np.zeros((rows_out * bloco, cols * bloco, 3), dtype=np.uint8)
    arr[:] = BG

    for by, row in enumerate(charmap):
        for bx, ch in enumerate(row):
            if ch == ' ':
                continue
            block = bits_to_block(GLYPHS_RAW[ch])
            cor = _cor_para_glyph_raw(ch, dmin, dmax)
            r, g, b = cor
            for py in range(8):
                for px in range(8):
                    if block[py, px] > 0.5:
                        y = by * bloco + py
                        x = bx * bloco + px
                        arr[y, x, 0] = r
                        arr[y, x, 1] = g
                        arr[y, x, 2] = b

    return Image.fromarray(arr, 'RGB')


# ---------------------------------------------------------------------
# Renderização colorida (all2.py) -- igual antes
# ---------------------------------------------------------------------

def render_colorido(charmap, bloco=8):
    rows_out = len(charmap)
    cols = len(charmap[0]) if rows_out else 0

    arr = np.zeros((rows_out * bloco, cols * bloco, 3), dtype=np.uint8)
    arr[:] = BG

    for by, row in enumerate(charmap):
        for bx, symbol in enumerate(row):
            camada, ch = symbol[0], symbol[1:]
            if camada == 'E':
                draw_char(arr, ch, bx, by, bloco, EDGE_COLOR, EDGE_BITMAPS)
            else:
                if ch != ' ':
                    dens = DENSIDADES[ch]
                    cor_fill = cor_para_densidade(dens)
                    draw_char(arr, ch, bx, by, bloco, cor_fill, FILL_BITMAPS)

    return Image.fromarray(arr, 'RGB')


def render_acc_to_image(acc_path):
    """Decodifica o .acc e devolve um objeto PIL.Image em memória
    (sem salvar nada em disco) -- sempre colorido."""
    charmap, cols, rows_out = load_acc(acc_path)
    if eh_formato_colorido(charmap):
        return render_colorido(charmap), "colorido (all2.py)"
    return render_mono_colorido(charmap), "mono colorizado (a.py/rnc.py)"


# ---------------------------------------------------------------------
# Janela gráfica (sem mudanças na estrutura)
# ---------------------------------------------------------------------

class AccViewerApp:
    def __init__(self, root, diretorio="."):
        self.root = root
        self.root.title("Visualizador .acc")
        self.root.geometry("1000x650")

        self.diretorio = Path(diretorio)
        self.arquivos = sorted(self.diretorio.glob("*.acc"))
        self.imagem_tk = None  # precisa manter referência viva

        self._montar_layout()
        self._popular_lista()

    def _montar_layout(self):
        painel_esq = ttk.Frame(self.root, width=260)
        painel_esq.pack(side="left", fill="y", padx=8, pady=8)

        ttk.Label(painel_esq, text="Arquivos .acc:").pack(anchor="w")

        self.lista = tk.Listbox(painel_esq, width=36)
        self.lista.pack(fill="y", expand=True)
        self.lista.bind("<<ListboxSelect>>", self._ao_selecionar)

        nav = ttk.Frame(painel_esq)
        nav.pack(fill="x", pady=6)
        ttk.Button(nav, text="◀ Anterior", command=self._anterior).pack(side="left", expand=True, fill="x")
        ttk.Button(nav, text="Próxima ▶", command=self._proxima).pack(side="left", expand=True, fill="x")

        painel_dir = ttk.Frame(self.root)
        painel_dir.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        self.label_info = ttk.Label(painel_dir, text="Selecione um arquivo à esquerda")
        self.label_info.pack(anchor="w")

        self.canvas = tk.Canvas(painel_dir, bg="#222222")
        self.canvas.pack(fill="both", expand=True)

    def _popular_lista(self):
        if not self.arquivos:
            self.label_info.config(text=f"Nenhum arquivo .acc encontrado em '{self.diretorio}'")
            return
        for f in self.arquivos:
            self.lista.insert(tk.END, f.name)
        self.lista.selection_set(0)
        self._mostrar_indice(0)

    def _ao_selecionar(self, event):
        sel = self.lista.curselection()
        if sel:
            self._mostrar_indice(sel[0])

    def _anterior(self):
        sel = self.lista.curselection()
        idx = sel[0] - 1 if sel else 0
        if idx >= 0:
            self.lista.selection_clear(0, tk.END)
            self.lista.selection_set(idx)
            self.lista.see(idx)
            self._mostrar_indice(idx)

    def _proxima(self):
        sel = self.lista.curselection()
        idx = sel[0] + 1 if sel else 0
        if idx < len(self.arquivos):
            self.lista.selection_clear(0, tk.END)
            self.lista.selection_set(idx)
            self.lista.see(idx)
            self._mostrar_indice(idx)

    def _mostrar_indice(self, idx):
        acc_path = self.arquivos[idx]
        try:
            img, formato = render_acc_to_image(acc_path)
        except Exception as e:
            self.label_info.config(text=f"Erro ao renderizar {acc_path.name}: {e}")
            return

        self.label_info.config(
            text=f"{acc_path.name}  |  {img.width}x{img.height}px  |  formato: {formato}"
        )

        self.canvas.update_idletasks()
        canvas_w = max(self.canvas.winfo_width(), 400)
        canvas_h = max(self.canvas.winfo_height(), 300)

        escala = min(canvas_w / img.width, canvas_h / img.height, 1.0) or 1.0
        if escala < 1.0 or max(img.width, img.height) < 300:
            fator_zoom = max(1, min(canvas_w // img.width, canvas_h // img.height, 8))
            img_exibida = img.resize(
                (img.width * fator_zoom, img.height * fator_zoom),
                resample=Image.NEAREST,
            )
        else:
            img_exibida = img

        self.imagem_tk = ImageTk.PhotoImage(img_exibida)
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2, canvas_h // 2, anchor="center", image=self.imagem_tk
        )


if __name__ == "__main__":
    diretorio = sys.argv[1] if len(sys.argv) > 1 else "."
    root = tk.Tk()
    app = AccViewerApp(root, diretorio)
    root.mainloop()
