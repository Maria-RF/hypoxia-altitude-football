# hypoxia-altitude-football

## 📁 Estructura del repo

```
hypoxia-altitude-football/
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ .gitignore
├─ LICENSE
├─ data/
│  ├─ DataBase_Futbol_2025v.xlsx           # (put your Excel here)
│  ├─ export_15-09-25.csv                   # (put your GPS CSV here)
│  └─ positions.csv                         # mapping name → position (provided sample)
├─ config/
│  └─ config.yaml
├─ src/
│  ├─ utils_io.py
│  ├─ preprocess_hypoxia.py
│  ├─ preprocess_gps.py
│  ├─ merge_and_positions.py
│  ├─ models_and_plots.py
│  └─ run_pipeline.py
├─ notebooks/
│  └─ 00_explore_and_checks.ipynb
└─ scripts/
   └─ run_all.sh
```
# Hypoxia × GPS in Elite Football (Chile NT)

Pipeline to merge **normobaric hypoxia training** data with **match GPS** at altitude, compute SpO₂ adaptations, and relate them to GPS performance with and without **position adjustment**.

## Quick start
```bash
git clone <your-repo-url>
cd hypoxia-altitude-football
python -m pip install -r requirements.txt
# Put your files in data/
#  - DataBase_Futbol_2025v.xlsx
#  - export_15-09-25.csv
#  - positions.csv (edit/complete the mapping)
python src/run_pipeline.py
