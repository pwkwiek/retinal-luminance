# retinal-vessel-segmentation
*Hue Knew Blood Vessels Look Better in Green?*

Unsupervised retinal blood vessel segmentation with classical image processing.  
Color channels, contrast enhancement, thresholding — and a bit of uncertainty.

Explores how **color representation**, **preprocessing**, and **thresholding algorithms**
affect retinal vessel segmentation, and proposes an optimized luminance channel
based on weighted RGB.


## What It Does

- Evaluates **19 single-channel representations**  (GRAY, RGB, HSV, HSL, YUV, YCrCb, Lab)
- Tests **5 thresholding methods**  Otsu, Yen, IsoData, Moments (Tsai), Global
- Compares preprocessing:
  - inversion only
  - CLAHE
  - CLAHE + white top-hat
- Finds the **best luminance channel** via **two-stage RGB weight optimization**
- Generates **pixel-wise uncertainty maps** (image-based + geometric)
- Analyzes **accuracy vs coverage** trade-off using uncertainty-aware filtering


## Key Findings

- **Luminance beats chrominance** — consistently.
- **Moments thresholding** is the most robust across channels.
- **CLAHE + top-hat** significantly improves Dice scores.
- An **optimized weighted RGB luminance** outperforms all standard channels.
- High uncertainty correlates with segmentation errors and thin vessels.


## Pipeline

1. Channel extraction → normalize to `[0,1]`
2. Invert → optional CLAHE → optional white top-hat
3. Global thresholding
4. Small-object removal
5. Evaluation (Dice, IoU, precision, recall, specificity)
6. RGB weight search (coarse → refine)
7. Uncertainty estimation + coverage analysis
8. Statistical testing (paired, non-parametric)


## Repo Structure

- main.ipynb
- rvseg/
    - config.py
    - channels.py
    - preproc.py
    - thresholding.py
    - metrics.py
    - run_part1.py # channel × preproc × threshold
    - run_part2.py # weighted RGB search
    - run_part3.py # uncertainty maps
    - run_part4.py # figures + trade-offs
    - run_part5.py # statistics

