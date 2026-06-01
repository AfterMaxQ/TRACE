"""
行业分类映射表生成

构建两套映射:
  映射A: IO部门(GB/T 4754) → CSRC代码 → 申万行业
  映射B: HS2章 → IO部门

输出: data/industry_mapping.csv
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ============================================================
# 映射A: IO部门 → CSRC代码 (基于GB/T 4754-2017)
# ============================================================

IO_TO_CSRC = {
    1:  ("农林牧渔产品和服务",               ["A01","A03","A04","A05"]),
    2:  ("煤炭采选产品",                     ["B06"]),
    3:  ("石油和天然气开采产品",              ["B07"]),
    4:  ("金属矿采选产品",                   ["B08","B09"]),
    5:  ("非金属矿和其他矿采选产品",          ["B10","B11"]),
    6:  ("食品和烟草",                       ["C13","C14","C15","C16"]),
    7:  ("纺织品",                           ["C17"]),
    8:  ("纺织服装鞋帽皮革羽绒及其制品",      ["C18","C19"]),
    9:  ("木材加工品和家具",                 ["C20","C21"]),
    10: ("造纸印刷和文教体育用品",            ["C22","C23","C24"]),
    11: ("石油、炼焦产品和核燃料加工品",      ["C25"]),
    12: ("化学产品",                         ["C26","C27","C28","C29"]),
    13: ("非金属矿物制品",                   ["C30"]),
    14: ("金属冶炼和压延加工品",              ["C31","C32"]),
    15: ("金属制品",                         ["C33"]),
    16: ("通用设备",                         ["C34"]),
    17: ("专用设备",                         ["C35"]),
    18: ("交通运输设备",                     ["C36","C37"]),
    19: ("电气机械和器材",                   ["C38"]),
    20: ("通信设备、计算机和其他电子设备",    ["C39"]),
    21: ("仪器仪表",                         ["C40"]),
    22: ("其他制造产品和废品废料",            ["C41","C42"]),
    23: ("金属制品、机械和设备修理服务",      ["C43"]),
    24: ("电力、热力的生产和供应",            ["D44"]),
    25: ("燃气生产和供应",                   ["D45"]),
    26: ("水的生产和供应",                   ["D46"]),
    27: ("建筑",                             ["E47","E48","E49","E50"]),
    28: ("批发和零售",                       ["F51","F52"]),
    29: ("交通运输、仓储和邮政",              ["G53","G54","G55","G56","G57","G58","G59","G60"]),
    30: ("住宿和餐饮",                       ["H61","H62"]),
    31: ("信息传输、软件和信息技术服务",      ["I63","I64","I65"]),
    32: ("金融",                             ["J66","J67","J68","J69"]),
    33: ("房地产",                           ["K70"]),
    34: ("租赁和商务服务",                   ["L71","L72"]),
    35: ("研究和试验发展",                   ["M73"]),
    36: ("综合技术服务",                     ["M74","M75"]),
    37: ("水利、环境和公共设施管理",          ["N76","N77","N78"]),
    38: ("居民服务、修理和其他服务",          ["O79","O80","O81"]),
    39: ("教育",                             ["P82"]),
    40: ("卫生和社会工作",                   ["Q83","Q84"]),
    41: ("文化、体育和娱乐",                 ["R85","R86","R87","R88","R89"]),
    42: ("公共管理、社会保障和社会组织",      ["S90","S91","S92","S93","S94","S95","S96"]),
}

# 个别没有 CSRC 分类的申万行业手工指定 IO 部门
MANUAL_SHENWAN_IO = {
    "陶瓷": 13,  # 非金属矿物制品
}

# ============================================================
# 映射B: HS2章 → IO部门 (基于产品大类对应)
# ============================================================

HS2_TO_IO = {
    # 活动物;动物产品 (01-05)
    1:1, 2:1, 3:1, 4:1, 5:1,
    # 植物产品 (06-14)
    6:1, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1, 13:1, 14:1,
    # 动植物油脂 (15)
    15:6,
    # 食品饮料烟酒 (16-24)
    16:6, 17:6, 18:6, 19:6, 20:6, 21:6, 22:6, 23:6, 24:6,
    # 矿产品 (25-27)
    25:5, 26:4, 27:3,
    # 化学产品 (28-38)
    28:12, 29:12, 30:12, 31:12, 32:12, 33:12, 34:12, 35:12, 36:12, 37:12, 38:12,
    # 塑料橡胶 (39-40)
    39:12, 40:12,
    # 皮革毛皮 (41-43)
    41:8, 42:8, 43:8,
    # 木制品 (44-46)
    44:9, 45:9, 46:9,
    # 纸浆纸板 (47-49)
    47:10, 48:10, 49:10,
    # 纺织原料及制品 (50-63)
    50:7, 51:7, 52:7, 53:7, 54:7, 55:7, 56:7, 57:7, 58:7, 59:7, 60:7,
    61:8, 62:8, 63:8,
    # 鞋帽 (64-67)
    64:8, 65:8, 66:8, 67:8,
    # 石料陶瓷玻璃 (68-70)
    68:13, 69:13, 70:13,
    # 珍珠宝石 (71)
    71:22,
    # 贱金属 (72-83)
    72:14, 73:14, 74:14, 75:14, 76:14, 78:14, 79:14, 80:14, 81:14,
    82:15, 83:15,
    # 核反应堆锅炉机器 (84)
    84:16,
    # 电机电气设备 (85)
    85:20,
    # 车辆航空器船舶 (86-89)
    86:18, 87:18, 88:18, 89:18,
    # 光学医疗仪器 (90-92)
    90:21, 91:21, 92:21,
    # 武器 (93)
    93:22,
    # 家具玩具杂项 (94-96)
    94:22, 95:22, 96:22,
    # 艺术品古董 (97)
    97:22,
    # 98-99 特殊交易品 → 跳过
}


def main():
    info = pd.read_csv(DATA_DIR / "company_info.csv", dtype={"ts_code": str})
    info["csrc_code"] = info["industry_csrc"].str.extract(r"^([A-Z]\d{2})")

    # CSRC → 申万 交叉表
    csrc_shenwan = (
        info.dropna(subset=["csrc_code", "industry"])
        .groupby("csrc_code")["industry"]
        .apply(lambda x: sorted(x.unique()))
        .to_dict()
    )

    # 展开 IO → CSRC → 申万
    rows = []
    for io_id, (io_name, csrc_list) in sorted(IO_TO_CSRC.items()):
        for csrc in csrc_list:
            shenwan_list = csrc_shenwan.get(csrc, [])
            if shenwan_list:
                for sw in shenwan_list:
                    rows.append({
                        "io_sector": io_id, "io_sector_name": io_name,
                        "csrc_code": csrc, "shenwan_industry": sw,
                    })
            else:
                rows.append({
                    "io_sector": io_id, "io_sector_name": io_name,
                    "csrc_code": csrc, "shenwan_industry": "",
                })

    df = pd.DataFrame(rows)

    # 手工补充无 CSRC 分类的申万行业
    for sw, io_id in MANUAL_SHENWAN_IO.items():
        if not ((df["io_sector"] == io_id) & (df["shenwan_industry"] == sw)).any():
            io_name = IO_TO_CSRC[io_id][0]
            df = pd.concat([df, pd.DataFrame([{
                "io_sector": io_id, "io_sector_name": io_name,
                "csrc_code": "", "shenwan_industry": sw,
            }])], ignore_index=True)

    # 附加 HS2 信息
    io_hs2 = {}
    for hs2, io_id in sorted(HS2_TO_IO.items()):
        io_hs2.setdefault(io_id, []).append(str(hs2))
    df["hs2_codes"] = df["io_sector"].map(lambda x: "|".join(io_hs2.get(x, [])))

    df = df.sort_values(["io_sector", "shenwan_industry"]).reset_index(drop=True)

    out = DATA_DIR / "industry_mapping.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] industry_mapping.csv: {len(df)} 行")
    print(f"     IO部门: {df['io_sector'].nunique()} 个")
    print(f"     申万行业: {df[df['shenwan_industry']!='']['shenwan_industry'].nunique()} 个")
    print(f"     含HS2映射的IO部门: {len(io_hs2)} 个")

    # 未映射申万行业
    mapped_sw = set(df[df["shenwan_industry"] != ""]["shenwan_industry"])
    all_sw = set(info["industry"].dropna().unique())
    unmapped = all_sw - mapped_sw
    if unmapped:
        print(f"\n[!] 未映射的申万行业 ({len(unmapped)} 个):")
        for sw in sorted(unmapped):
            print(f"    {sw} ({(info['industry']==sw).sum()} 只)")


if __name__ == "__main__":
    main()
