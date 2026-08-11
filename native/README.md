# Yerel yüksek performans çekirdeği

Bu klasör, Python bağımlılığı olmayan C++17 çekirdeğini içerir. Çekirdek:

- AVX2/FMA çalışma-zamanı dağıtımı ile vektör çarpım/toplamı,
- OpenMP ile çok çekirdekli bağlam biriktirme ve matris-vektör çarpımı,
- güvenli skaler geri dönüş yolunu sağlar.

Windows (TDM-GCC dahil) için:

```powershell
cmake -S native -B native/build -G "MinGW Makefiles"
cmake --build native/build --config Release
```

`ridm_kernels.dll` dosyasını `native/` içine kopyalayın veya `native/build/`
içinde bırakın; `ComputeBackend` otomatik bulur. OpenMP bulunmazsa çekirdek
aynı ABI ile tek iş parçacığında derlenir. AVX-512 özelliği raporlanır; mevcut
kernels AVX2/FMA yolunu kullanır, böylece AVX2 cihazlarda da aynı ikili güvenle
çalışır.

Not: Bu çalışma alanındaki TDM-GCC kurulumunda OpenMP çalışma zamanı
(`libgomp.spec`) eksikse, derleme tek iş parçacıklı ikili üretir. Çok çekirdekli
çalışma için eksiksiz MinGW/MSVC OpenMP kurulumu veya CMake'in bulduğu bir
OpenMP aracı zinciri gereklidir.
