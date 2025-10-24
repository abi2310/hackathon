# ===============================
# 📦 1. Libraries importieren
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 🧭 2. Daten einlesen
# ===============================
df = pd.read_csv("data/processed/auftrag_gesamt_features.csv")

features = [
    "Arbeitsschritt",
    "is_transport_ruesten",
    "Maschinenkapazität",
    "Maschinenkapazität_relativ",
    "AFO_Letzter_Schritt",
    "auftrags_eingang_tag_num",
    "auftrags_eingang_monat",
    "auftrags_eingang_jahr",
    "auftrags_ende_tag_num",
    "auftrags_ende_monat",
    "auftrags_ende_jahr",
    "auftrags_geplante_dauer_tage",
    "Lieferabweichung_Stunden",
    "anzahl_steuerventilmodul_in_auftrag",
    "anzahl_schwenkzylinder_in_auftrag",
    "anzahl_daempfungseinheit_in_auftrag",
    "anzahl_maschine_a_in_auftrag",
    "anzahl_maschine_b_in_auftrag",
    "anzahl_maschine_c_in_auftrag",
    "anzahl_maschine_d_in_auftrag",
    "anzahl_maschine_e_in_auftrag",
    "avg_verspaetung_pro_afo",
    "Anzahl_Bauteile_im_Auftrag",
    "Beteiligte_Maschinen_Anzahl",
    "Auftrag_SOLL_Stunden",
    "Auftrag_IST_Stunden"
]


target = "Lieferabweichung_Stunden"

df = df[features + ["Lieferabweichung_Stunden"]].dropna().copy()




# ===============================
# 🎯 5. Zielvariable anpassen (absolute Abweichung)
# ===============================
y = np.abs(df[target])  # absolute Abweichung = Entfernung vom Idealwert (0)
X = df.drop(columns=[target])



# ===============================
# 🧩 6. Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


for col in X_train.columns:
    if X_train[col].astype(str).str.contains(r"\d{4}-\d{2}-\d{2}").any():
        print(f"⚠️ Datumsähnlicher String in: {col}")
        print(X_train[col].head())
# ===============================
# 🌲 7. Random Forest trainieren
# ===============================
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ===============================
# 🔍 8. Feature Importances
# ===============================
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(9,6))
sns.barplot(x=importances, y=importances.index, palette="viridis")
plt.title("Wichtigste Einflussfaktoren auf starke Lieferabweichungen")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\n🎯 Wichtigste Einflussfaktoren (|Lieferabweichung_Stunden|):\n", importances)

# ===============================
# 📊 9. Modellbewertung (optional)
# ===============================
from sklearn.metrics import r2_score, mean_absolute_error

y_pred = rf.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f} Stunden")
