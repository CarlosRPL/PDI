"""
huffman_acc.py
Formato de arquivo .acc: guarda um charmap ASCII (matriz de caracteres)
comprimido com Huffman canônico.

Layout do arquivo (tudo big-endian):
  4 bytes  magic       b'ACC1'
  2 bytes  cols        uint16
  2 bytes  rows        uint16
  1 byte   n_symbols   uint8   (quantidade de símbolos distintos, até 256)
  para cada símbolo (n_symbols vezes):
      1 byte  utf8_len     tamanho em bytes do caractere em utf-8
      N bytes utf8_bytes   o caractere em si (suporta '·', '█', etc)
      1 byte  code_len     comprimento do código Huffman canônico (bits)
  1 byte   padding_bits (bits de padding no fim do bitstream)
  N bytes  bitstream    o charmap inteiro (row-major) codificado em Huffman

Huffman canônico: só precisamos guardar o COMPRIMENTO do código de cada
símbolo (não a árvore inteira) -- os códigos são reconstruídos
deterministicamente a partir dos comprimentos + ordem (comprimento, símbolo).
Isso deixa o cabeçalho minúsculo (poucos bytes por símbolo distinto).
"""

import heapq
from collections import Counter

MAGIC = b'ACC1'


# ---------------------------------------------------------------------
# 1. Construção do código Huffman canônico
# ---------------------------------------------------------------------

def _build_code_lengths(freq):
    """Árvore de Huffman clássica -> {simbolo: comprimento_em_bits}."""
    if len(freq) == 1:
        only = next(iter(freq))
        return {only: 1}  # caso degenerado: só 1 símbolo distinto

    heap = [[w, i, [sym]] for i, (sym, w) in enumerate(freq.items())]
    heapq.heapify(heap)
    lengths = {sym: 0 for sym in freq}
    counter = len(heap)

    while len(heap) > 1:
        w1, _, syms1 = heapq.heappop(heap)
        w2, _, syms2 = heapq.heappop(heap)
        for s in syms1 + syms2:
            lengths[s] += 1
        heapq.heappush(heap, [w1 + w2, counter, syms1 + syms2])
        counter += 1

    return lengths


def _canonical_codes(lengths):
    """A partir de {simbolo: comprimento}, gera os códigos canônicos:
    ordena por (comprimento, símbolo) e atribui códigos binários
    sequenciais -- mesma ideia do Huffman canônico usado em DEFLATE."""
    symbols_sorted = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))

    codes = {}
    code = 0
    prev_len = 0
    for sym, length in symbols_sorted:
        code <<= (length - prev_len)
        codes[sym] = format(code, f'0{length}b')
        code += 1
        prev_len = length

    return codes


# ---------------------------------------------------------------------
# 2. Empacotamento de bits
# ---------------------------------------------------------------------

class BitWriter:
    def __init__(self):
        self.bits = []

    def write(self, bitstring):
        self.bits.append(bitstring)

    def to_bytes(self):
        joined = ''.join(self.bits)
        padding = (-len(joined)) % 8
        joined += '0' * padding
        out = bytearray()
        for i in range(0, len(joined), 8):
            out.append(int(joined[i:i+8], 2))
        return bytes(out), padding


class BitReader:
    def __init__(self, data, padding_bits):
        bits = ''.join(f'{byte:08b}' for byte in data)
        if padding_bits:
            bits = bits[:-padding_bits]
        self.bits = bits
        self.pos = 0

    def read_bit(self):
        b = self.bits[self.pos]
        self.pos += 1
        return b


# ---------------------------------------------------------------------
# 3. Encode / Decode de um charmap completo
# ---------------------------------------------------------------------

def encode_acc(charmap):
    """charmap: lista de listas de caracteres (row-major).
    Retorna bytes prontos pra escrever em .acc."""
    rows = len(charmap)
    cols = len(charmap[0]) if rows else 0

    flat = [ch for row in charmap for ch in row]
    freq = Counter(flat)

    if len(freq) > 256:
        raise ValueError("Mais de 256 símbolos distintos não é suportado pelo formato .acc")

    lengths = _build_code_lengths(freq)
    codes = _canonical_codes(lengths)

    writer = BitWriter()
    for ch in flat:
        writer.write(codes[ch])
    bitstream, padding = writer.to_bytes()

    out = bytearray()
    out += MAGIC
    out += cols.to_bytes(2, 'big')
    out += rows.to_bytes(2, 'big')

    symbols_sorted = sorted(lengths.items(), key=lambda kv: (kv[1], kv[0]))
    out += len(symbols_sorted).to_bytes(1, 'big')
    for sym, length in symbols_sorted:
        sym_bytes = sym.encode('utf-8')
        out += len(sym_bytes).to_bytes(1, 'big')
        out += sym_bytes
        out += length.to_bytes(1, 'big')

    out += padding.to_bytes(1, 'big')
    out += bitstream

    return bytes(out)


def decode_acc(data):
    """Recebe os bytes crus de um .acc e devolve (charmap, cols, rows)."""
    if data[:4] != MAGIC:
        raise ValueError("Arquivo .acc inválido (magic incorreto)")

    pos = 4
    cols = int.from_bytes(data[pos:pos+2], 'big'); pos += 2
    rows = int.from_bytes(data[pos:pos+2], 'big'); pos += 2
    n_symbols = data[pos]; pos += 1

    lengths = {}
    for _ in range(n_symbols):
        utf8_len = data[pos]; pos += 1
        sym = data[pos:pos+utf8_len].decode('utf-8'); pos += utf8_len
        code_len = data[pos]; pos += 1
        lengths[sym] = code_len

    codes = _canonical_codes(lengths)
    decode_table = {code: sym for sym, code in codes.items()}

    padding_bits = data[pos]; pos += 1
    bitstream = data[pos:]

    reader = BitReader(bitstream, padding_bits)

    total = cols * rows
    flat = []
    buf = ''
    while len(flat) < total:
        buf += reader.read_bit()
        if buf in decode_table:
            flat.append(decode_table[buf])
            buf = ''

    charmap = [flat[r*cols:(r+1)*cols] for r in range(rows)]
    return charmap, cols, rows


# ---------------------------------------------------------------------
# 4. Helpers de arquivo
# ---------------------------------------------------------------------

def save_acc(path, charmap):
    data = encode_acc(charmap)
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


def load_acc(path):
    with open(path, 'rb') as f:
        data = f.read()
    return decode_acc(data)
