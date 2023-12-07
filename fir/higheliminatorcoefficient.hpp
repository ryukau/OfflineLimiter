// Copyright (C) 2022  Takamitsu Endo
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#pragma once

#include <array>

/**
HighEliminationFir removes high frequency components near nyquist frequency to reduce
clipping on true peak mode.

```python
import numpy as np
firLength = 64  # Should be even. `delay` equals to `len(firLength) // 2  - 1`.
normalizedFreq = np.linspace(0, 1, firLength // 2 + 1, endpoint=True)
invMaxTruepeak = np.cos(np.pi * normalizedFreq / 2)
fir = np.fft.irfft(invMaxTruepeak.astype(np.complex128))
fir = np.roll(fir, len(fir) // 2 - 1)
```
*/
template<typename Sample> struct HighEliminatorCoefficient {
  static constexpr size_t delay = 31;

  constexpr static std::array<Sample, 64> fir{
    Sample(0.0003844985110431631),  Sample(-0.0003872953485491333),
    Sample(0.0003920173750087356),  Sample(-0.000398758196713085),
    Sample(0.00040765438143190247), Sample(-0.0004188914625306169),
    Sample(0.00043271241349051746), Sample(-0.0004494292782792489),
    Sample(0.00046943895240961154), Sample(-0.0004932445541488396),
    Sample(0.0005214844800564499),  Sample(-0.0005549722241208874),
    Sample(0.0005947515530382776),  Sample(-0.000642174004709342),
    Sample(0.0006990094878452091),  Sample(-0.000767607027647943),
    Sample(0.0008511332897935347),  Sample(-0.0009539349429931198),
    Sample(0.0010821040895120232),  Sample(-0.001244388003641562),
    Sample(0.001453705485421736),   Sample(-0.001729780642571678),
    Sample(0.0021039457780140224),  Sample(-0.002628427076632147),
    Sample(0.0033956303639039154),  Sample(-0.004581972476481666),
    Sample(0.0065598948680232905),  Sample(-0.010233904536753106),
    Sample(0.01831752967623514),    Sample(-0.0425694018646525),
    Sample(0.21233448929477244),    Sample(0.6364919355013017),
    Sample(0.21233448929477244),    Sample(-0.042569401864652495),
    Sample(0.018317529676235136),   Sample(-0.010233904536753106),
    Sample(0.006559894868023299),   Sample(-0.004581972476481665),
    Sample(0.0033956303639039094),  Sample(-0.002628427076632147),
    Sample(0.0021039457780140177),  Sample(-0.001729780642571677),
    Sample(0.0014537054854217393),  Sample(-0.001244388003641562),
    Sample(0.0010821040895120215),  Sample(-0.0009539349429931185),
    Sample(0.0008511332897935264),  Sample(-0.000767607027647943),
    Sample(0.0006990094878452147),  Sample(-0.0006421740047093442),
    Sample(0.000594751553038269),   Sample(-0.0005549722241208874),
    Sample(0.0005214844800564505),  Sample(-0.0004932445541488405),
    Sample(0.000469438952409612),   Sample(-0.0004494292782792489),
    Sample(0.0004327124134905166),  Sample(-0.0004188914625306182),
    Sample(0.000407654381431909),   Sample(-0.000398758196713085),
    Sample(0.0003920173750087269),  Sample(-0.0003872953485491368),
    Sample(0.0003844985110431631),  Sample(-0.0003835722204519887),
  };
};
