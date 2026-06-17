// gen_dataset_v3.cpp
// Compile: g++ -O2 -std=c++17 gen_dataset_v3.cpp -o gen_dataset_v3
// Uso:     ./gen_dataset_v3 [amostras_por_classe] [seed] [use_dog: 0=Sobel, 1=DoG]
// Saída:   dataset.bin (label + 64 bytes intensidade + 64 bytes borda, por amostra)
//          labels.txt

#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>

using Core = std::array<float, 64>; // 8x8 em linha, 0.0-1.0

// ---------------------------------------------------------------------
// 1. Glifos base
// ---------------------------------------------------------------------

struct GlyphDef {
    std::string symbol;
    uint8_t rows[8];
    float delicacy;
};

static const std::vector<GlyphDef> GLYPHS = {
    {" ",              {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, 0.0f},
    {".",              {0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x18}, 1.0f},
    {"\xC2\xB7",       {0x00,0x00,0x00,0x00,0x00,0x00,0x18,0x00}, 1.0f},
    {"`",              {0x10,0x08,0x00,0x00,0x00,0x00,0x00,0x00}, 1.0f},
    {",",              {0x00,0x00,0x00,0x00,0x00,0x18,0x08,0x10}, 0.9f},
    {":",              {0x00,0x00,0x18,0x18,0x00,0x18,0x18,0x00}, 0.9f},
    {";",              {0x00,0x00,0x18,0x18,0x00,0x18,0x08,0x10}, 0.9f},
    {"!",              {0x18,0x18,0x18,0x18,0x18,0x00,0x18,0x00}, 0.6f},
    {"'",              {0x08,0x10,0x00,0x00,0x00,0x00,0x00,0x00}, 1.0f},
    {"-",              {0x00,0x00,0x00,0x00,0x7E,0x00,0x00,0x00}, 0.5f},
    {"~",              {0x00,0x00,0x00,0x32,0x4C,0x00,0x00,0x00}, 0.6f},
    {"+",              {0x00,0x00,0x18,0x18,0x7E,0x18,0x18,0x00}, 0.4f},
    {"*",              {0x00,0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00}, 0.5f},
    {"=",              {0x00,0x00,0x7E,0x00,0x7E,0x00,0x00,0x00}, 0.4f},
    {"?",              {0x3C,0x42,0x02,0x0C,0x18,0x00,0x18,0x00}, 0.5f},
    {"c",              {0x00,0x00,0x3C,0x42,0x40,0x42,0x3C,0x00}, 0.8f},
    {"o",              {0x00,0x00,0x3C,0x42,0x42,0x42,0x3C,0x00}, 0.5f},
    {"x",              {0x00,0x42,0x24,0x18,0x18,0x24,0x42,0x00}, 0.3f},
    {"n",              {0x00,0x00,0x5C,0x62,0x42,0x42,0x42,0x00}, 0.3f},
    {"u",              {0x00,0x00,0x42,0x42,0x42,0x46,0x3A,0x00}, 0.3f},
    {"P",              {0x78,0x44,0x44,0x78,0x40,0x40,0x40,0x00}, 0.2f},
    {"R",              {0x78,0x44,0x44,0x78,0x28,0x24,0x42,0x00}, 0.2f},
    {"M",              {0x3C,0x66,0x6E,0x76,0x66,0x66,0x3C,0x00}, 0.2f},
    {"O",              {0x3C,0x42,0x42,0x42,0x42,0x42,0x3C,0x00}, 0.4f},
    {"B",              {0x78,0x44,0x44,0x78,0x44,0x44,0x78,0x00}, 0.2f},
    {"#",              {0x24,0x24,0x7E,0x24,0x24,0x7E,0x24,0x24}, 0.7f},
    {"@",              {0x3C,0x42,0x5A,0x5E,0x5E,0x40,0x3C,0x00}, 0.7f},
    {"\xE2\x96\x88",   {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF}, 0.0f},
};

Core bits_to_core(const uint8_t rows[8]) {
    Core g{};
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            g[y*8+x] = ((rows[y] >> x) & 1) ? 1.0f : 0.0f;
    return g;
}

// ---------------------------------------------------------------------
// 2. Augmentation de intensidade
// ---------------------------------------------------------------------

Core blur_core(const Core &in, float sigma) {
    if (sigma <= 0.01f) return in;
    int radius = std::max(1, (int)std::ceil(sigma*2.5f));
    std::vector<float> k(2*radius+1);
    float sum = 0;
    for (int i=-radius;i<=radius;i++){ float v=std::exp(-(i*i)/(2*sigma*sigma)); k[i+radius]=v; sum+=v; }
    for (auto &v : k) v /= sum;
    auto cl = [](int v,int lo,int hi){ return std::max(lo,std::min(hi,v)); };
    Core tmp{}, out{};
    for (int y=0;y<8;y++) for (int x=0;x<8;x++){
        float acc=0; for (int t=-radius;t<=radius;t++){ int xx=cl(x+t,0,7); acc+=in[y*8+xx]*k[t+radius]; }
        tmp[y*8+x]=acc;
    }
    for (int y=0;y<8;y++) for (int x=0;x<8;x++){
        float acc=0; for (int t=-radius;t<=radius;t++){ int yy=cl(y+t,0,7); acc+=tmp[yy*8+x]*k[t+radius]; }
        out[y*8+x]=acc;
    }
    return out;
}

Core contrast_brightness_gamma(const Core &in, float c, float b, float g) {
    Core out{};
    for (int i=0;i<64;i++){
        float v = (in[i]-0.5f)*c + 0.5f + b;
        v = std::clamp(v, 0.0f, 1.0f);
        v = std::pow(v, g);
        out[i] = std::clamp(v, 0.0f, 1.0f);
    }
    return out;
}

Core add_noise(const Core &in, float sigma, std::mt19937 &rng) {
    std::normal_distribution<float> d(0.0f, sigma);
    Core out{};
    for (int i=0;i<64;i++) out[i]=std::clamp(in[i]+d(rng),0.0f,1.0f);
    return out;
}

Core augment_intensity(const Core &base, float delicacy, std::mt19937 &rng) {
    std::uniform_real_distribution<float> u01(0.0f,1.0f);
    float max_blur = 0.9f * (1.0f - 0.6f*delicacy);
    Core g = base;
    g = blur_core(g, u01(rng)*max_blur);
    float contrast = 1.0f + (u01(rng)*2-1)*0.4f;
    float brightness = (u01(rng)*2-1)*0.15f;
    float gamma = 1.0f + (u01(rng)*2-1)*0.3f;
    g = contrast_brightness_gamma(g, contrast, brightness, gamma);
    g = add_noise(g, u01(rng)*0.05f, rng);
    return g;
}

// ---------------------------------------------------------------------
// 3. Sobel e DoG direto no bloco 8x8 (borda replicada)
// ---------------------------------------------------------------------

Core sobel_8x8(const Core &in) {
    static const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    static const int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    auto cl = [](int v){ return std::max(0, std::min(7, v)); };
    Core out{};
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            float sx = 0, sy = 0;
            for (int dy=-1; dy<=1; dy++)
                for (int dx=-1; dx<=1; dx++) {
                    float v = in[cl(y+dy)*8 + cl(x+dx)];
                    sx += v * Gx[dy+1][dx+1];
                    sy += v * Gy[dy+1][dx+1];
                }
            float mag = std::sqrt(sx*sx + sy*sy) / 4.0f;
            out[y*8+x] = std::clamp(mag, 0.0f, 1.0f);
        }
    }
    return out;
}

Core dog_8x8(const Core &in) {
    Core fine = blur_core(in, 0.6f);
    Core coarse = blur_core(in, 1.6f);
    Core out{};
    for (int i = 0; i < 64; i++) {
        float d = fine[i] - coarse[i]; // aprox [-1,1]
        out[i] = std::clamp(d + 0.5f, 0.0f, 1.0f); // remapeia pra faixa visível
    }
    return out;
}

// ---------------------------------------------------------------------
// 4. Geração do dataset
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
    int samples_per_class = 20000;
    unsigned seed = 42;
    bool use_dog = true;

    if (argc >= 2) samples_per_class = std::atoi(argv[1]);
    if (argc >= 3) seed = (unsigned)std::atoi(argv[2]);
    if (argc >= 4) use_dog = std::atoi(argv[3]) != 0;

    std::mt19937 rng(seed);
    std::ofstream bin("dataset.bin", std::ios::binary);
    std::ofstream labels("labels.txt");

    int num_classes = (int)GLYPHS.size();
    labels << num_classes << "\n";
    for (int i = 0; i < num_classes; i++) labels << i << " " << GLYPHS[i].symbol << "\n";

    uint32_t total = 0;

    for (int cls = 0; cls < num_classes; cls++) {
        Core base = bits_to_core(GLYPHS[cls].rows);
        float delicacy = GLYPHS[cls].delicacy;

        for (int s = 0; s < samples_per_class; s++) {
            Core intensity = augment_intensity(base, delicacy, rng);
            Core edge = use_dog ? dog_8x8(intensity) : sobel_8x8(intensity);

            uint8_t label = (uint8_t)cls;
            bin.write((char*)&label, 1);
            for (int i = 0; i < 64; i++) {
                uint8_t px = (uint8_t)std::round(intensity[i]*255.0f);
                bin.write((char*)&px, 1);
            }
            for (int i = 0; i < 64; i++) {
                uint8_t px = (uint8_t)std::round(edge[i]*255.0f);
                bin.write((char*)&px, 1);
            }
            total++;
        }
        std::cout << "Classe " << cls << " (" << GLYPHS[cls].symbol << "): "
                  << samples_per_class << " amostras\n";
    }

    std::cout << "\nTotal: " << total << " amostras, " << num_classes
              << " classes, 2 canais (intensidade + " << (use_dog ? "DoG" : "Sobel")
              << ") por amostra.\n";
    std::cout << "Formato binario: 1 byte label + 64 bytes intensidade + 64 bytes borda\n";
    return 0;
}
