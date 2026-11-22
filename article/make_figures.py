"""Vẽ 4 biểu đồ cho bài viết ngắn (tiếng Việt và tiếng Anh).
Chạy từ thư mục phase3/article/:  python make_figures.py
Đọc dữ liệu ở <gốc repo>/phase1/data/clean và <gốc repo>/phase2/data (tự dò, hoặc đặt HANOI_AQ_DIR)."""
import os
import textwrap
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("HANOI_AQ_DIR", ".." if os.path.isdir(os.path.join("..", "phase1")) else os.path.join("..", ".."))
EP_A, EP_B = "2018-04-16", "2018-04-20"  # đợt minh họa (chọn từ day_clusters: S2 mở đầu, S3 kết thúc)
OUT = "figures"; os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SHAPE_COL = {"S4": "#2a78d6", "S1": "#eb6834", "S2": "#1baf7a", "S3": "#6250d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})

T = {
 "vi": dict(
   f1_title="PM2.5 trung bình ngày theo tháng (trung vị, 2016–2025)",
   f1_y="µg/m³", f1_x="Tháng", who="WHO 2021: 15", qcvn="QCVN 05:2023: 50",
   f1_note="Trạm Đại sứ quán Mỹ, Hà Nội. Thanh: trung vị; vạch mảnh: 25%–75% số ngày.",
   f2_title="Bốn kiểu hình dạng của một ngày",
   f2_y="lần so với mức TB của ngày", f2_x="Giờ",
   shapes={"S4": "Phẳng, nhích lên ban ngày", "S1": "Cao ban đêm, thấp buổi chiều",
           "S2": "Tăng dần về tối", "S3": "Giảm dần trong ngày"},
   f3_title="Trong cùng một mùa, ngày gió yếu thường ô nhiễm hơn",
   f3_y="PM2.5 trung vị (µg/m³)", terc=["Gió yếu", "Gió TB", "Gió mạnh"],
   seasons={"Đông": "Đông (12–2)", "Xuân": "Xuân (3–4)", "Hè": "Hè (5–9)", "Thu": "Thu (10–11)"},
   f3_note="Gió yếu/mạnh = 1/3 số ngày có gió yếu/mạnh nhất trong cùng tháng.",
   f4_title="Một đợt ô nhiễm điển hình: 16–20/4/2018",
   f4_y="PM2.5 theo giờ (µg/m³)", f4_wind="gió (m/s)"),
 "en": dict(
   f1_title="Daily PM2.5 by month (median, 2016–2025)",
   f1_y="µg/m³", f1_x="Month", who="WHO 2021: 15", qcvn="Vietnam QCVN 05:2023: 50",
   f1_note="U.S. Embassy monitor, Hanoi. Bar: median; whisker: 25th–75th percentile of days.",
   f2_title="Four shapes of a day",
   f2_y="× the day's average level", f2_x="Hour",
   shapes={"S4": "Flat, slight daytime bump", "S1": "High at night, low in afternoon",
           "S2": "Rising towards evening", "S3": "Falling through the day"},
   f3_title="Within the same season, calm days tend to be dirtier",
   f3_y="Median PM2.5 (µg/m³)", terc=["Weak wind", "Medium", "Strong wind"],
   seasons={"Đông": "Winter (Dec–Feb)", "Xuân": "Spring (Mar–Apr)", "Hè": "Summer (May–Sep)", "Thu": "Autumn (Oct–Nov)"},
   f3_note="Weak/strong = calmest/windiest third of days within the same month.",
   f4_title="A typical episode: 16–20 April 2018",
   f4_y="Hourly PM2.5 (µg/m³)", f4_wind="wind (m/s)"),
}

daily = pd.read_parquet(os.path.join(ROOT, "phase1/data/clean/hanoi_daily.parquet"))
hourly = pd.read_parquet(os.path.join(ROOT, "phase1/data/clean/hanoi_hourly_enriched.parquet"))
clus = pd.read_csv(os.path.join(ROOT, "phase2/data/day_clusters.csv"), parse_dates=["date"])
cent = pd.read_csv(os.path.join(ROOT, "phase2/data/shape_centroids.csv"), index_col=0)
cent.index = ["S1", "S2", "S3", "S4"]
share = clus.shape_code.str[:2].value_counts(normalize=True)

def save(fig, name, lang):
    fig.savefig(os.path.join(OUT, f"{name}_{lang}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

def fig_month(t, lang):
    d = daily[daily.day_ok]
    g = d.groupby("month").pm25_mean.quantile([.25, .5, .75]).unstack()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(g.index, g[0.5], width=0.6, color="#2a78d6")
    ax.vlines(g.index, g[0.25], g[0.75], color=INK2, lw=1.2)
    for y, lab in [(15, t["who"]), (50, t["qcvn"])]:
        ax.axhline(y, color=INK, lw=1, ls="--"); ax.text(12.45, y + 1.5, lab, ha="right", color=INK, fontsize=9)
    ax.set_xticks(range(1, 13)); ax.set_xlabel(t["f1_x"]); ax.set_ylabel(t["f1_y"])
    ax.set_title(t["f1_title"], loc="left", color=INK, fontsize=13); ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.03, t["f1_note"], color=INK2, fontsize=9)
    save(fig, "01_by_month", lang)

def fig_shapes(t, lang):
    fig, axs = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, sharey=True)
    for ax, s in zip(axs.flat, ["S4", "S1", "S2", "S3"]):
        ax.axhline(1, color=INK2, lw=0.8)
        ax.plot(range(24), cent.loc[s].values, color=SHAPE_COL[s], lw=2.5)
        ax.set_title(f"{t['shapes'][s]}  ({share[s]*100:.0f}%)", loc="left", fontsize=11, color=INK)
        ax.set_xticks([0, 6, 12, 18, 23])
    for ax in axs[:, 0]: ax.set_ylabel(t["f2_y"])
    for ax in axs[1]: ax.set_xlabel(t["f2_x"])
    fig.suptitle(t["f2_title"], x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(); save(fig, "02_day_shapes", lang)

def fig_wind(t, lang):
    c = clus.dropna(subset=["ws_mean", "pm25_mean"]).copy()
    c["ym"] = c.date.dt.month
    c["terc"] = c.groupby("ym").ws_mean.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    g = c.groupby(["season", "terc"]).pm25_mean.median().unstack()
    order = ["Đông", "Xuân", "Thu", "Hè"]; cols = ["#184f95", "#2a78d6", "#9cc3f0"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2)); w = 0.26
    for k in range(3):
        x = np.arange(4) + (k - 1) * w
        ax.bar(x, g.loc[order, k], width=w - 0.02, color=cols[k], label=t["terc"][k])
        for xi, v in zip(x, g.loc[order, k]): ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels([t["seasons"][s] for s in order])
    ax.set_ylabel(t["f3_y"]); ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title(t["f3_title"], loc="left", fontsize=13, color=INK); ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.03, t["f3_note"], color=INK2, fontsize=9)
    save(fig, "03_wind_within_season", lang)

def fig_episode(t, lang):
    a, b = pd.Timestamp(EP_A), pd.Timestamp(EP_B) + pd.Timedelta(hours=23)
    h = hourly.reset_index()
    if h["time"].dt.tz is not None: h["time"] = h["time"].dt.tz_localize(None)
    h = h[(h.time >= a) & (h.time <= b)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(h.time, h.pm25_filled, color=INK, lw=1.8)  # khoảng thiếu ≤ 3 giờ đã nội suy
    cl = clus.set_index("date")
    for day in pd.date_range(a, EP_B):
        if day in cl.index:
            s = cl.loc[day, "shape_code"][:2]
            ax.axvspan(day, day + pd.Timedelta(hours=24), color=SHAPE_COL[s], alpha=0.13, lw=0)
            ax.text(day + pd.Timedelta(hours=12), 0.97, textwrap.fill(t["shapes"][s], 16), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.5, color=INK)
    ax.set_ylabel(t["f4_y"]); ax.set_ylim(0, h.pm25_filled.max() * 1.35)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m"))
    # một trục duy nhất; gió ghi dưới dạng chữ
    wd = h.groupby(h.time.dt.date).wind_speed_10m.mean()
    for day, v in wd.items():
        ax.text(pd.Timestamp(day) + pd.Timedelta(hours=12), 4, f"{t['f4_wind']}: {v:.1f}", ha="center", fontsize=8, color=INK2)
    ax.set_title(t["f4_title"], loc="left", fontsize=13, color=INK)
    save(fig, "04_episode", lang)

for lang, t in T.items():
    fig_month(t, lang); fig_shapes(t, lang); fig_wind(t, lang); fig_episode(t, lang)
print("done", sorted(os.listdir(OUT)))
