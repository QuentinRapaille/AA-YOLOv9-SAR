# AA-YOLOv9-SAR

Base: YOLOv9 (WongKinYiu)  
Extension: adaptation AA-YOLO pour SAR (branche objectness a contrario + ré-pondération des scores de classe).

## Ce qui a été modifié

- Base du repo migrée vers **YOLOv9** (structure `models/detect/*.yaml`, `train.py`, `val.py`, TAL/DFL).
- Ajout d'une tête **AA**: `DDetect_AA`.
- Ajout des modules AA:
  - `filtering2D`
  - `anomaly_testing` (test a contrario avec stabilisation numérique via gamma incomplète)
- Intégration loss AA dans `utils/loss_tal.py`:
  - cible objectness binaire (`1` sur positives TAL, `0` ailleurs)
  - MSE objectness pondérée par `obj_aa`.
- Modèle YAML AA prêt à l'emploi:
  - `models/detect/AA-gelan-t.yaml`

## Arborescence utile

- Modèle AA: `models/detect/AA-gelan-t.yaml`
- Hypers AA: `data/hyp.scratch.AA_yolo.yaml`
- Dataset HRSID: `data/hrsid.yaml`
- Scripts dataset:
  - `scripts/prepare_sar_optionA.py`
  - `scripts/prepare_ssdd.py`
  - `scripts/prepare_ls_ssdd_v1.py`

## Installation (venv)

```bash
cd AA-YOLOv9-SAR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Le `requirements.txt` est contraint pour éviter les incompatibilités connues:
- `numpy<2`
- `opencv-python<4.12`
- `setuptools<81`

## Entraînement

### HRSID (single GPU)

```bash
python train.py \
  --data data/hrsid.yaml \
  --cfg models/detect/AA-gelan-t.yaml \
  --hyp data/hyp.scratch.AA_yolo.yaml \
  --weights '' \
  --epochs 100 \
  --batch-size 16 \
  --img 640 \
  --device 0 \
  --workers 4 \
  --project runs/train \
  --name aa_gelan_t_hrsid
```

### Smoke test (1 époque)

```bash
python train.py \
  --data data/hrsid.yaml \
  --cfg models/detect/AA-gelan-t.yaml \
  --hyp data/hyp.scratch.AA_yolo.yaml \
  --weights '' \
  --epochs 1 \
  --batch-size 4 \
  --img 640 \
  --device 0 \
  --workers 2 \
  --name smoke_aa_gelan_t
```

## Validation

```bash
python val.py \
  --data data/hrsid.yaml \
  --weights runs/train/aa_gelan_t_hrsid/weights/best.pt \
  --img 640 \
  --batch 16 \
  --conf 0.001 \
  --iou 0.7 \
  --device 0
```

## Inférence

```bash
python detect.py \
  --weights runs/train/aa_gelan_t_hrsid/weights/best.pt \
  --source /chemin/vers/images \
  --img 640 \
  --conf 0.25 \
  --iou 0.45 \
  --device 0
```

## Erreurs fréquentes (Ruche)

### 1) `FloatTensor` vs `HalfTensor` en validation

Cause: mismatch de dtype AMP dans la branche AA objectness.  
Correctif appliqué: conversion explicite `xobj` au dtype de la branche régression dans `DDetect_AA`.

### 2) Warning `torch.cuda.amp.autocast` déprécié

Migration appliquée vers:

```python
torch.amp.autocast('cuda', enabled=amp)
```

### 3) `pkg_resources` / NumPy 2.x incompatibles

Corrigé via les pins de dépendances (`setuptools<81`, `numpy<2`).

## Notes

- Le repo original pré-migration est conservé localement dans `AA-YOLOv9-SAR_v7base_backup`.
- Le fichier dataset `data/hrsid.yaml` pointe vers:
  - `../AA_YOLO_SAR/data/datasets/HRSID_YOLO/HRSID_train.txt`
  - `../AA_YOLO_SAR/data/datasets/HRSID_YOLO/HRSID_val.txt`
  - `../AA_YOLO_SAR/data/datasets/HRSID_YOLO/HRSID_test.txt`

Adaptez ces chemins si vous déplacez les datasets.
