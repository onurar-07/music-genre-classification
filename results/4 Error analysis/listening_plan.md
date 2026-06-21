# Listening Plan for Error Analysis

Source predictions: `results/3 Hybrid Modal/predictions.csv`

Human judgement 建议填写：`Correct label` / `Model reasonable` / `Ambiguous` / `Model wrong`.

| # | Group | track_id | mp3_path | true | pred | conf. | Focus |
|---:|---|---:|---|---|---|---:|---|
| 1 | A. High-confidence errors | 113305 | `data/fma_small/113/113305.mp3` | Rock | Electronic | 0.9626 | 先判断这是模型明显错，还是标签/genre 边界本身模糊。重点听主导音色、节奏、人声、语言和整体风格。 |
| 2 | A. High-confidence errors | 99135 | `data/fma_small/099/099135.mp3` | Electronic | International | 0.8771 | 先判断这是模型明显错，还是标签/genre 边界本身模糊。重点听主导音色、节奏、人声、语言和整体风格。 |
| 3 | A. High-confidence errors | 49477 | `data/fma_small/049/049477.mp3` | Pop | Hip-Hop | 0.8462 | 先判断这是模型明显错，还是标签/genre 边界本身模糊。重点听主导音色、节奏、人声、语言和整体风格。 |
| 4 | A. High-confidence errors | 111386 | `data/fma_small/111/111386.mp3` | Pop | Electronic | 0.8119 | 先判断这是模型明显错，还是标签/genre 边界本身模糊。重点听主导音色、节奏、人声、语言和整体风格。 |
| 5 | A. High-confidence errors | 126183 | `data/fma_small/126/126183.mp3` | Instrumental | Rock | 0.8102 | 先判断这是模型明显错，还是标签/genre 边界本身模糊。重点听主导音色、节奏、人声、语言和整体风格。 |
| 6 | A. High-confidence errors | 104068 | `data/fma_small/104/104068.mp3` | Pop | Rock | 0.7944 | 先判断这是模型明显错，还是标签/genre 边界本身模糊。重点听主导音色、节奏、人声、语言和整体风格。 |
| 7 | A. High-confidence errors | 114037 | `data/fma_small/114/114037.mp3` | Hip-Hop | Instrumental | 0.7887 | 先判断这是模型明显错，还是标签/genre 边界本身模糊。重点听主导音色、节奏、人声、语言和整体风格。 |
| 8 | A. High-confidence errors | 132779 | `data/fma_small/132/132779.mp3` | Pop | International | 0.7806 | 先判断这是模型明显错，还是标签/genre 边界本身模糊。重点听主导音色、节奏、人声、语言和整体风格。 |
| 9 | B. Pop -> Rock | 131436 | `data/fma_small/131/131436.mp3` | Pop | Rock | 0.7631 | 听是否有吉他/鼓组/摇滚能量导致模型偏 Rock。 |
| 10 | B. Pop -> Rock | 57435 | `data/fma_small/057/057435.mp3` | Pop | Rock | 0.7470 | 听是否有吉他/鼓组/摇滚能量导致模型偏 Rock。 |
| 11 | B. Pop -> Rock | 138065 | `data/fma_small/138/138065.mp3` | Pop | Rock | 0.6952 | 听是否有吉他/鼓组/摇滚能量导致模型偏 Rock。 |
| 12 | B. Pop -> Rock | 131452 | `data/fma_small/131/131452.mp3` | Pop | Rock | 0.6768 | 听是否有吉他/鼓组/摇滚能量导致模型偏 Rock。 |
| 13 | B. Pop -> Electronic | 58474 | `data/fma_small/058/058474.mp3` | Pop | Electronic | 0.7133 | 听是否合成器、电子鼓或制作感压过了 Pop 旋律。 |
| 14 | B. Pop -> Electronic | 92124 | `data/fma_small/092/092124.mp3` | Pop | Electronic | 0.4584 | 听是否合成器、电子鼓或制作感压过了 Pop 旋律。 |
| 15 | B. Pop -> Folk | 87187 | `data/fma_small/087/087187.mp3` | Pop | Folk | 0.7748 | 听是否 acoustic/民谣编曲让模型忽略了 Pop song structure。 |
| 16 | B. Pop -> Folk | 127289 | `data/fma_small/127/127289.mp3` | Pop | Folk | 0.7190 | 听是否 acoustic/民谣编曲让模型忽略了 Pop song structure。 |
| 17 | B. Pop -> International | 39658 | `data/fma_small/039/039658.mp3` | Pop | International | 0.6405 | 听是否语言/地域音色让模型偏 International。 |
| 18 | B. Pop -> Instrumental | 69182 | `data/fma_small/069/069182.mp3` | Pop | Instrumental | 0.7088 | 听是否人声弱或缺失导致模型偏 Instrumental。 |
| 19 | C. Experimental -> Instrumental | 110274 | `data/fma_small/110/110274.mp3` | Experimental | Instrumental | 0.7650 | 听是否无明确人声/结构，导致模型把实验性理解成 Instrumental。 |
| 20 | C. Experimental -> Instrumental | 126512 | `data/fma_small/126/126512.mp3` | Experimental | Instrumental | 0.6395 | 听是否无明确人声/结构，导致模型把实验性理解成 Instrumental。 |
| 21 | C. Experimental -> Instrumental | 139638 | `data/fma_small/139/139638.mp3` | Experimental | Instrumental | 0.6226 | 听是否无明确人声/结构，导致模型把实验性理解成 Instrumental。 |
| 22 | C. Experimental -> Electronic | 30090 | `data/fma_small/030/030090.mp3` | Experimental | Electronic | 0.6195 | 听是否电子音色是主要线索。 |
| 23 | C. Experimental -> Electronic | 38912 | `data/fma_small/038/038912.mp3` | Experimental | Electronic | 0.5777 | 听是否电子音色是主要线索。 |
| 24 | C. Experimental -> Rock | 44791 | `data/fma_small/044/044791.mp3` | Experimental | Rock | 0.6793 | 听是否吉他/鼓组/噪声墙让模型偏 Rock。 |
| 25 | C. Experimental -> Rock | 109106 | `data/fma_small/109/109106.mp3` | Experimental | Rock | 0.5403 | 听是否吉他/鼓组/噪声墙让模型偏 Rock。 |
| 26 | C. Experimental -> Folk | 74377 | `data/fma_small/074/074377.mp3` | Experimental | Folk | 0.6956 | 听是否 acoustic texture 或弱节奏让模型偏 Folk。 |
| 27 | D. Hip-Hop -> Electronic | 70773 | `data/fma_small/070/070773.mp3` | Hip-Hop | Electronic | 0.7007 | 听 beat、rap/vocal 是否弱，电子制作是否更突出。 |
| 28 | D. Hip-Hop -> Electronic | 75386 | `data/fma_small/075/075386.mp3` | Hip-Hop | Electronic | 0.6403 | 听 beat、rap/vocal 是否弱，电子制作是否更突出。 |
| 29 | D. Electronic -> Hip-Hop | 56034 | `data/fma_small/056/056034.mp3` | Electronic | Hip-Hop | 0.7677 | 听是否强鼓点/节奏型让模型误判为 Hip-Hop。 |
| 30 | D. Electronic -> Hip-Hop | 115321 | `data/fma_small/115/115321.mp3` | Electronic | Hip-Hop | 0.6733 | 听是否强鼓点/节奏型让模型误判为 Hip-Hop。 |
| 31 | D. International -> Folk | 13539 | `data/fma_small/013/013539.mp3` | International | Folk | 0.7286 | 听 acoustic/传统乐器与地域标签的边界。 |
| 32 | D. International -> Folk | 38782 | `data/fma_small/038/038782.mp3` | International | Folk | 0.6945 | 听 acoustic/传统乐器与地域标签的边界。 |
| 33 | D. Rock -> Pop | 98077 | `data/fma_small/098/098077.mp3` | Rock | Pop | 0.4256 | 听 chorus/旋律性是否让 Rock 被拉向 Pop。 |
| 34 | D. Rock -> Pop | 87363 | `data/fma_small/087/087363.mp3` | Rock | Pop | 0.3616 | 听 chorus/旋律性是否让 Rock 被拉向 Pop。 |
| 35 | E. High-confidence correct controls | 84009 | `data/fma_small/084/084009.mp3` | Folk | Folk | 0.9347 | 作为对照，听模型在 Folk 上学到的清晰 cue。 |
| 36 | E. High-confidence correct controls | 71255 | `data/fma_small/071/071255.mp3` | Hip-Hop | Hip-Hop | 0.9640 | 作为对照，听模型在 Hip-Hop 上学到的清晰 cue。 |
| 37 | E. High-confidence correct controls | 59686 | `data/fma_small/059/059686.mp3` | International | International | 0.9332 | 作为对照，听模型在 International 上学到的清晰 cue。 |
| 38 | E. High-confidence correct controls | 107799 | `data/fma_small/107/107799.mp3` | Rock | Rock | 0.9172 | 作为对照，听模型在 Rock 上学到的清晰 cue。 |
| 39 | E. High-confidence correct controls | 114289 | `data/fma_small/114/114289.mp3` | Electronic | Electronic | 0.9174 | 作为对照，听模型在 Electronic 上学到的清晰 cue。 |
| 40 | E. High-confidence correct controls | 86485 | `data/fma_small/086/086485.mp3` | Instrumental | Instrumental | 0.8514 | 作为对照，听模型在 Instrumental 上学到的清晰 cue。 |

## Notes Template

| track_id | human_judgement | audible cues | notes |
|---:|---|---|---|
| 113305 |  |  |  |
| 99135 |  |  |  |
| 49477 |  |  |  |
| 111386 |  |  |  |
| 126183 |  |  |  |
| 104068 |  |  |  |
| 114037 |  |  |  |
| 132779 |  |  |  |
| 131436 |  |  |  |
| 57435 |  |  |  |
| 138065 |  |  |  |
| 131452 |  |  |  |
| 58474 |  |  |  |
| 92124 |  |  |  |
| 87187 |  |  |  |
| 127289 |  |  |  |
| 39658 |  |  |  |
| 69182 |  |  |  |
| 110274 |  |  |  |
| 126512 |  |  |  |
| 139638 |  |  |  |
| 30090 |  |  |  |
| 38912 |  |  |  |
| 44791 |  |  |  |
| 109106 |  |  |  |
| 74377 |  |  |  |
| 70773 |  |  |  |
| 75386 |  |  |  |
| 56034 |  |  |  |
| 115321 |  |  |  |
| 13539 |  |  |  |
| 38782 |  |  |  |
| 98077 |  |  |  |
| 87363 |  |  |  |
| 84009 |  |  |  |
| 71255 |  |  |  |
| 59686 |  |  |  |
| 107799 |  |  |  |
| 114289 |  |  |  |
| 86485 |  |  |  |