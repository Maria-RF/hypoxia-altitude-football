# hypoxia-altitude-football

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
