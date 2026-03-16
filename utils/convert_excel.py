# Einmalig ausführen: convert_excel.py
import pandas as pd, json

# file="Schneckenstein_I_Schicht_ID_join.xlsx"
file="Schneckenstein_II_Schicht_ID_join.xlsx"
# file="Schneckenstein_III_Schicht_ID_join.xlsx"

df = pd.read_excel(f'assets/excel/{file}', engine="openpyxl")
mapping = (
    df[["ID_Schicht", "PETVERB1"]]
    .dropna()
    .drop_duplicates("ID_Schicht")
    .assign(ID_Schicht=lambda x: x["ID_Schicht"].astype(int),
            PETVERB1=lambda x: x["PETVERB1"].astype(str).str.strip())
    .set_index("ID_Schicht")["PETVERB1"]
    .to_dict()
)
with open(f'assets/json/class_map_{file}.json', "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in mapping.items()}, f, ensure_ascii=False, indent=2)