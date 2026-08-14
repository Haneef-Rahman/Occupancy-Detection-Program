# Foundations for a mathematical human-shape filter

Reading list and derivation notes for building the thermal person filter from
first principles rather than by tuning. Organised as four layers: what a shape
*is* mathematically, what a *human* shape is specifically, what the body does
*thermally*, and how prior work has combined the two.

Every reference below was checked to exist; where a free copy is available the
link points at it.

---

## Layer 1 — What a shape is, mathematically

The core question: reduce a binary region to numbers that identify it,
independent of where it sits, how big it is, and which way it faces.

**Hu, M.-K. (1962). "Visual Pattern Recognition by Moment Invariants."**
*IRE Transactions on Information Theory*, 8(2), 179–187.
The founding paper. Treats the image as a mass distribution, takes central
moments, and derives **seven combinations invariant to translation, rotation
and scale**. This is the "absolute basics" of shape description — everything
later is a refinement or an alternative.
[PDF](https://github.com/CDahmsTemp/Computer_Vision_Papers/blob/master/Visual%20Pattern%20Recognition%20by%20Moment%20Invariants%20-%20Hu%201962.pdf)

**Zhang, D. & Lu, G. (2004). "Review of shape representation and description
techniques."** *Pattern Recognition*, 37(1), 1–19.
The survey to read next. Divides descriptors into **region-based** (moments,
area, solidity) and **boundary-based** (chain codes, Fourier descriptors,
circularity, eccentricity, convexity). Its central conclusion is directly
relevant to us: *simple descriptors are not adequate alone — a combination is
required to describe a shape accurately.*
[Free PDF](https://cis.temple.edu/~latecki/Courses/CIS601-04/ProjectPapers/shapeRepPR04.pdf)

### The descriptors worth deriving by hand

| Descriptor | Definition | Why a body differs from an object |
|---|---|---|
| Compactness / circularity | `4πA / P²` | Limbs add perimeter faster than area → bodies score low, boxes and discs high |
| Solidity | `A / A_convexhull` | Armpits, leg gap, neck notch cut solidity; manufactured objects ≈ 1.0 |
| Eccentricity | from 2nd-order moments | Orientation and elongation without a bounding box |
| Extent | `A / A_bbox` | Crude but cheap (already implemented) |
| Rect fill | `A / A_minAreaRect` | Rotation-invariant boxiness (already implemented) |
| Hu moments φ₁…φ₇ | Hu (1962) | Full invariant signature; φ₁, φ₂ carry most of the discriminating power |
| Convexity defects | hull minus contour | **The limb detector** — count and depth of gaps |
| Medial axis / skeleton | thinning | Limb *topology*: endpoints and branch count |
| Projection profiles | row/column sums | The 1-D "head → shoulders → torso → legs" signature |
| Fourier descriptors | FFT of contour signal | Low harmonics = gross shape, high = detail; naturally invariant |

---

## Layer 2 — What a *human* shape is specifically

**The Ω (omega) head-and-shoulders model.** The outline of a head above
shoulders traces an omega. It is the classical hand-crafted human feature
because the head–shoulder geometry is the most pose-stable part of the body —
limbs move, that relationship does not.

- Li, Zhang et al. "Rapid and robust human detection and tracking based on
  omega-shape features." *ICIP 2009*.
  [ACM DL](https://dl.acm.org/doi/10.5555/1819298.1819438)
- Later deep reformulation, useful for seeing what the feature encodes:
  "Rapid Pedestrian Detection Based on Deep Omega-Shape Features with Partial
  Occlusion Handling," *Neural Processing Letters* (2018).
  [Springer](https://link.springer.com/article/10.1007/s11063-018-9837-1)

**Dalal, N. & Triggs, B. (2005). "Histograms of Oriented Gradients for Human
Detection."** *CVPR 2005*.
The best pre-deep-learning person detector. Worth reading not to implement HOG,
but to see the lesson: the performance came from a carefully **hand-designed
feature**; the learning layer (a linear SVM) was almost trivial. Understanding
shape came first.

---

## Layer 3 — What the body does thermally

This layer decides the *temperature* half of the filter, and is where the
"recalculate from ambient" work belongs.

**Skin emissivity ≈ 0.98.** Steketee, J. (1973), "Spectral emissivity of skin
and pericardium," *Physics in Medicine and Biology* — the most-cited source for
the 0.98 figure, agreed by ~96 % of experts in a later Delphi study. Earlier
foundational work: Hardy & Muschenheim (1934). Practical consequence: skin is
very close to a blackbody, so indicated temperature is nearly true temperature
— *for bare skin*. Clothing, hair and any window in front change this, which is
exactly why measured values sit below textbook skin temperature.

- Emissivity effect on measurement:
  [Physica Medica (2012)](https://www.physicamedica.com/article/S1120-1797(12)00186-X/pdf)
- Pigmentation does **not** meaningfully change emissivity — important for a
  fairness claim in the report:
  [PMC7688144](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7688144/)

**Fanger's thermal comfort model / ASHRAE Standard 55.** Fanger, P.O. (1970),
*Thermal Comfort*. The heat-balance framework behind the surface-temperature
model already implemented in `thermal_detect.py`: body core, clothing
insulation in **clo** units (1 clo = 0.155 m²·K/W), convective and radiative
loss to the room. This is the rigorous version of
`T_surface = ambient + coupling × (T_core − ambient)` — the coupling constants
are effectively a lumped clo value, and can be derived properly from here.
[Fanger model summary (NRC IRC-RR-162)](https://nascoinc.com/content/resources/PO-Fanger-Thermal-Comfort.pdf)

---

## Layer 4 — Prior art: shape + thermal together

**"Pedestrians' detection in thermal bands – Critical survey."** *Journal of
Electrical Systems and Information Technology* (2015). Survey of the whole
problem space; good for locating which techniques survive on low-resolution
thermal.
[ScienceDirect (open)](https://www.sciencedirect.com/science/article/pii/S2314717215000343)

**"Robust Pedestrian Detection in Thermal Infrared Imagery Using a Shape
Distribution Histogram Feature."** Directly relevant: builds a *shape
distribution* descriptor for thermal silhouettes rather than using appearance.
[ResearchGate](https://www.researchgate.net/publication/272405556)

**Overhead geometry.** "Robust people detection using depth information from an
overhead Time-of-Flight camera," *Expert Systems with Applications* (2016).
The overhead head-and-shoulder morphology problem, solved on depth — which is
the same geometry your SPAD will see, so it doubles as preparation for the
fusion stage.
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417416306480)

---

## Suggested build order

1. **Implement and log** compactness, solidity, eccentricity, Hu φ₁–φ₂ and
   convexity-defect count on the blobs already being extracted. Do not filter
   on them yet — just record them.
2. **Measure** the distributions for people vs. the MacBook, radiator, lamp and
   empty room. This produces the actual separating values instead of guessed
   thresholds.
3. **Derive the thermal band from Fanger** rather than from assumed coupling
   constants: pick a clo value for indoor clothing, solve the heat balance for
   surface temperature at the measured ambient.
4. **Add the projection-profile / Ω test** for the horizontal mode, where the
   head–shoulder signature is strongest.
5. **Only then** decide which descriptors earn a place in the filter, using the
   measured separations as evidence.

Step 2 is the one that turns this from tuning into engineering: every threshold
in the final filter should be traceable to a measured distribution or a
physical constant, not to a value that seemed to work.
