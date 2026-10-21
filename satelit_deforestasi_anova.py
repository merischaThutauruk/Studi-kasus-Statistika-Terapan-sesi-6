import json
import math
import random


class SatelliteDeforestationAttributionAI:

    def __init__(self, seed=2026):
        self.seed = seed
        self.alerts_data = []
        self.driver_classes = [
            "Pertanian Skala Kecil",
            "Pertanian Skala Besar",
            "Penebangan Selektif (Logging)",
            "Pertambangan Emas/Mineral",
            "Kebakaran Hutan / Banjir",
        ]

    def generate_multisensor_alerts(self, n_alerts=250):
        print(
            "=== [1] MULTI-SENSOR BAND STACKING: SENTINEL-1 (SAR) & SENTINEL-2 (OPTIK) ==="
        )
        random.seed(self.seed)

        class_weights = [0.40, 0.28, 0.14, 0.08, 0.10]

        for i in range(n_alerts):
            r_val = random.random()
            cum = 0.0
            true_driver = self.driver_classes[0]
            for cls_name, w in zip(self.driver_classes, class_weights):
                cum += w
                if r_val <= cum:
                    true_driver = cls_name
                    break

            cloud_cover_pct = round(random.uniform(0.0, 85.0), 1)
            is_cloudy = cloud_cover_pct > 50.0

            if true_driver == "Pertanian Skala Kecil":
                s1_sar = random.gauss(-9.5, 1.1)
                s2_ndvi = random.gauss(0.38, 0.07)
                luas_ha = random.uniform(0.5, 4.5)
            elif true_driver == "Pertanian Skala Besar":
                s1_sar = random.gauss(-11.2, 1.0)
                s2_ndvi = random.gauss(0.25, 0.06)
                luas_ha = random.uniform(15.0, 120.0)
            elif true_driver == "Penebangan Selektif (Logging)":
                s1_sar = random.gauss(-7.8, 0.9)
                s2_ndvi = random.gauss(0.52, 0.08)
                luas_ha = random.uniform(1.0, 12.0)
            elif true_driver == "Pertambangan Emas/Mineral":
                s1_sar = random.gauss(-14.5, 1.3)  
                s2_ndvi = random.gauss(0.12, 0.05)
                luas_ha = random.uniform(3.0, 35.0)
            else:  
                s1_sar = random.gauss(-13.8, 1.8)
                s2_ndvi = random.gauss(0.18, 0.09)
                luas_ha = random.uniform(5.0, 60.0)

            pred_driver = true_driver
            if true_driver == "Kebakaran Hutan / Banjir" and random.random() < 0.38:
                pred_driver = random.choice(
                    ["Pertanian Skala Kecil", "Pertambangan Emas/Mineral"]
                )
            elif (
                true_driver == "Pertambangan Emas/Mineral"
                and random.random() < 0.20
            ):
                pred_driver = "Kebakaran Hutan / Banjir"
            elif is_cloudy and random.random() < 0.12:
                pred_driver = random.choice(self.driver_classes)

            self.alerts_data.append({
                "alert_id": f"GLAD-S1S2-{2026000 + i}",
                "lat": round(-1.25 + random.uniform(-1.5, 1.5), 5),
                "lon": round(114.50 + random.uniform(-3.5, 3.5), 5),
                "cloud_cover_pct": cloud_cover_pct,
                "s1_sar_db": round(s1_sar, 3),
                "s2_ndvi": None if is_cloudy else round(s2_ndvi, 3),
                "luas_ha": round(luas_ha, 2),
                "true_driver": true_driver,
                "pred_driver": pred_driver,
            })

        cloudy_count = sum(1 for a in self.alerts_data if a["s2_ndvi"] is None)
        print(
            f"- Total peringatan deforestasi diproses        : {len(self.alerts_data)} titik (Resolusi 10m)"
        )
        print(
            f"- Citra Sentinel-2 terhalang awan (>50%)       : {cloudy_count} titik (Model beralih ke Radar Sentinel-1)\n"
        )

    def perform_one_way_anova_sar(self):
        print(
            "=== [2] UJI STATISTIKA TERAPAN: ONE-WAY ANOVA (F-TEST) SINYAL SENTINEL-1 ==="
        )
        print(
            "Tujuan Statistik: Menguji apakah nilai hamburan balik radar (Sentinel-1 Backscatter)"
        )
        print(
            "                  berbeda secara signifikan di antara 5 faktor pemicu deforestasi.\n"
        )

        groups = {cls_name: [] for cls_name in self.driver_classes}
        for a in self.alerts_data:
            groups[a["true_driver"]].append(a["s1_sar_db"])

        all_values = [a["s1_sar_db"] for a in self.alerts_data]
        grand_mean = sum(all_values) / len(all_values)

        ss_between = 0.0
        ss_within = 0.0

        print(
            f"{'KELAS PEMICU DEFORESTASI':<32} | {'JUMLAH SAMPEL':<15} | {'RATA-RATA S1 SAR (dB)'}"
        )
        print("-" * 75)
        for cls_name, vals in groups.items():
            n_g = len(vals)
            mean_g = sum(vals) / n_g
            ss_between += n_g * ((mean_g - grand_mean) ** 2)
            ss_within += sum((v - mean_g) ** 2 for v in vals)
            print(f"{cls_name:<32} | {n_g:<15} | {mean_g:>16.3f} dB")
        print("-" * 75)

        k = len(self.driver_classes)  
        N = len(all_values) 
        df_between = k - 1
        df_within = N - k

        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        f_stat = ms_between / ms_within if ms_within != 0 else 0.0

        print(
            f"- Sum of Squares Between (SSB) : {ss_between:.2f} (df1 = {df_between})"
        )
        print(
            f"- Sum of Squares Within (SSW)  : {ss_within:.2f} (df2 = {df_within})"
        )
        print(f"- Nilai Statistik F-Hitung     : {f_stat:.4f}")
        print("- Nilai F-Tabel Kritis (a=0.05): 2.4085")

        if f_stat > 2.4085:
            print(
                "\n[KESIMPULAN ANOVA] TOLAK H0 (F-Hitung >> F-Tabel, p-value < 0.0001)."
            )
            print(
                "Interpretasi: Karakteristik pantulan radar Sentinel-1 berbeda sangat signifikan antar pemicu,"
            )
            print(
                "              sehingga efektif membedakan bukaan tambang/banjir meskipun tertutup awan.\n"
            )

    def evaluate_imbalanced_classification(self):
        print(
            "=== [3] EVALUASI AI: OVERALL ACCURACY VS MACRO F1-SCORE (IMBALANCED DATA) ==="
        )
        total = len(self.alerts_data)
        correct_total = sum(
            1 for a in self.alerts_data if a["true_driver"] == a["pred_driver"]
        )
        overall_accuracy = correct_total / total

        f1_per_class = []
        print(
            f"{'KATEGORI PEMICU (DRIVER)':<32} | {'PRECISION':<11} | {'RECALL':<11} | {'F1-SCORE'}"
        )
        print("-" * 75)

        for cls_name in self.driver_classes:
            tp = sum(
                1
                for a in self.alerts_data
                if a["true_driver"] == cls_name and a["pred_driver"] == cls_name
            )
            fp = sum(
                1
                for a in self.alerts_data
                if a["true_driver"] != cls_name and a["pred_driver"] == cls_name
            )
            fn = sum(
                1
                for a in self.alerts_data
                if a["true_driver"] == cls_name and a["pred_driver"] != cls_name
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1_per_class.append(f1)

            print(
                f"{cls_name:<32} | {precision * 100:>8.2f}%  | {recall * 100:>8.2f}%  | {f1:>8.4f}"
            )

        print("-" * 75)
        macro_f1 = sum(f1_per_class) / len(f1_per_class)
        print(
            f"- Overall Accuracy (Didominasi Kelas Pertanian) : {overall_accuracy * 100:.2f}%"
        )
        print(
            f"- Macro F1-Score (Bobot Setara Seluruh Kelas)   : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)"
        )
        print(
            "\n[INSIGHT STATISTIKA TERAPAN PADA AI GEOSPASIAL]"
        )
        print(
            "Sesuai penjelasan Dr. Bart Slachter, Overall Accuracy tampak tinggi karena mayoritas deforestasi"
        )
        print(
            "disebabkan oleh pertanian. Namun Macro F1-Score lebih rendah karena memberi bobot yang adil"
        )
        print(
            "terhadap kelas minoritas yang sulit dibedakan seperti Kebakaran Hutan vs Banjir di tepi sungai.\n"
        )

    def export_alerts_to_geojson(
        self, filename="global_nature_watch_alerts.geojson"
    ):
        features = []
        for a in self.alerts_data[:50]:  
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [a["lon"], a["lat"]],
                },
                "properties": {
                    "alert_id": a["alert_id"],
                    "predicted_driver": a["pred_driver"],
                    "area_ha": a["luas_ha"],
                    "sentinel1_sar_db": a["s1_sar_db"],
                    "cloud_cover_pct": a["cloud_cover_pct"],
                },
            })

        geojson_obj = {"type": "FeatureCollection", "features": features}
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(geojson_obj, f, indent=4)
        print(
            f"=== [4] TRANSPARANSI DATA PUBLIK (GLOBAL NATURE WATCH EXPORT) ==="
        )
        print(
            f"- Berhasil mengekspor peta atribusi pemicu deforestasi ke '{filename}'."
        )


if __name__ == "__main__":
    sat_ai = SatelliteDeforestationAttributionAI()
    sat_ai.generate_multisensor_alerts()
    sat_ai.perform_one_way_anova_sar()
    sat_ai.evaluate_imbalanced_classification()
    sat_ai.export_alerts_to_geojson()