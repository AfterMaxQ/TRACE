"""
F-03 NLP 事件分类模块 — DeepSeek v4 Flash

对新闻标题做 5 维度结构化分类，聚合为 9 个公司×季度情感特征。

输出:
  data/sentiment_raw.csv      逐条分类标签 (断点续跑)
  data/sentiment_features.csv 公司×季度聚合特征

用法:
  python src/sentiment.py --input news_sample.csv
  python src/sentiment.py --limit 100 --workers 4
"""

import os
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    except ImportError:
        pass
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"

BATCH_SIZE = 20
WORKERS = 80

VALID_EVENT_TYPES = {"financial","compliance","operation","market","strategy","macro"}
VALID_SENTIMENTS = {"negative","neutral","positive"}
VALID_SCOPES = {"firm","industry","macro"}
VALID_CHANNELS = {"revenue","cost","financing","governance","uncertain"}

SYSTEM_PROMPT = """你是金融风险分析师。对每条财经新闻标题，输出一个 JSON 对象，包含 5 个分类维度。

event_type (事件类型):
- financial: 财务相关 (业绩预告/年报/利润/亏损/分红/资产减值/债务)
- compliance: 合规相关 (立案调查/处罚/诉讼/ST/退市/监管函/违规)
- operation: 经营相关 (订单/中标/产能/研发/扩产/停产/裁员)
- market: 市场相关 (股价涨跌/大宗交易/增减持/回购/解禁)
- strategy: 战略相关 (并购重组/定增/分拆上市/合作/战略协议)
- macro: 宏观行业 (行业政策/关税/汇率/原材料价格/行业景气)

sentiment (对公司信用的影响):
- negative: 利空 (增加违约风险)
- neutral: 中性或无法判断
- positive: 利好 (降低违约风险)

scope (影响范围):
- firm: 仅影响该公司
- industry: 影响整个行业
- macro: 影响宏观经济/全市场

impact_channel (影响传导路径):
- revenue: 影响营业收入
- cost: 影响成本端
- financing: 影响融资/资产/负债/现金流
- governance: 影响公司治理/合规
- uncertain: 无法判断

supply_chain:
- true: 涉及供应商/客户/上下游关系
- false: 不涉及

对每条标题输出一行 JSON，不要解释:
{"event_type":"financial","sentiment":"negative","scope":"firm","impact_channel":"revenue","supply_chain":false}"""


def _build_user_prompt(titles: list[str]) -> str:
    lines = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))
    return f"有 {len(titles)} 条标题，输出 {len(titles)} 行 JSON:\n\n{lines}"


def _parse_response(text: str) -> list[dict | None]:
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            obj.setdefault("event_type", "other")
            obj.setdefault("sentiment", "neutral")
            obj.setdefault("scope", "firm")
            obj.setdefault("impact_channel", "uncertain")
            obj.setdefault("supply_chain", False)
            if obj["event_type"] not in VALID_EVENT_TYPES:
                obj["event_type"] = "other"
            if obj["sentiment"] not in VALID_SENTIMENTS:
                obj["sentiment"] = "neutral"
            if obj["scope"] not in VALID_SCOPES:
                obj["scope"] = "firm"
            if obj["impact_channel"] not in VALID_CHANNELS:
                obj["impact_channel"] = "uncertain"
            if not isinstance(obj["supply_chain"], bool):
                obj["supply_chain"] = False
            results.append(obj)
        except json.JSONDecodeError:
            results.append(None)
    return results


def _call_api(client: OpenAI, titles: list[str], worker_id: int) -> list[dict | None]:
    user_prompt = _build_user_prompt(titles)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
            timeout=120,
            extra_body={"user_id": f"worker-{worker_id}"},
        )
        content = resp.choices[0].message.content
        parsed = _parse_response(content)
        while len(parsed) < len(titles):
            parsed.append(None)
        return parsed[:len(titles)]
    except Exception as e:
        tqdm.write(f"      API error: {e}")
        return [None] * len(titles)


def _classify_batch(task: tuple[int, list[str], int], api_key: str, retries: int = 3) -> list[dict | None]:
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=120)
    for attempt in range(1, retries + 1):
        results = _call_api(client, task[1], task[2])
        if any(r is not None for r in results):
            return results
        if attempt < retries:
            time.sleep(3 ** attempt)
    return [None] * len(task[1])


def _aggregate_to_quarterly(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["quarter"] = (
        df["date"].dt.year.astype(str) + "Q" +
        ((df["date"].dt.month - 1) // 3 + 1).astype(str)
    )
    df["sentiment_score"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1}).fillna(0)

    agg = df.groupby(["code", "quarter"]).agg(
        event_count=("title", "count"),
        sentiment_mean=("sentiment_score", "mean"),
        neg_ratio=("sentiment_score", lambda x: (x == -1).mean()),
        compliance_ratio=("event_type", lambda x: (x == "compliance").mean()),
        financial_ratio=("event_type", lambda x: (x == "financial").mean()),
        industry_scope_ratio=("scope", lambda x: (x == "industry").mean()),
        financing_impact_ratio=("impact_channel", lambda x: (x == "financing").mean()),
        supply_chain_ratio=("supply_chain", lambda x: x.astype(bool).mean()),
    ).reset_index()

    vol = df.groupby(["code", "quarter"])["sentiment_score"].std().reset_index()
    vol.columns = ["code", "quarter", "sentiment_volatility"]
    agg = agg.merge(vol, on=["code", "quarter"], how="left")
    agg["sentiment_volatility"] = agg["sentiment_volatility"].fillna(0)
    return agg


def main():
    parser = argparse.ArgumentParser(description="F-03 NLP 事件分类")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--input", type=str, default="news_clean.csv")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  F-03 NLP 事件分类 (DeepSeek v4 Flash, workers={args.workers})")
    print("=" * 60)

    news_path = DATA_DIR / args.input
    if not news_path.exists():
        print(f"[!!] {args.input} 不存在")
        return

    news = pd.read_csv(news_path)
    if args.limit:
        news = news.head(args.limit)

    total = len(news)
    print(f"  待分类: {total:,}, batch={args.batch_size}, workers={args.workers}")

    # 断点续跑
    raw_path = DATA_DIR / "sentiment_raw.csv"
    done_titles = set()
    if raw_path.exists():
        try:
            existing = pd.read_csv(raw_path)
            done_titles = set(existing["title"])
        except Exception:
            pass

    done_indices = set(news[news["title"].isin(done_titles)].index)
    pending = [(i, row) for i, row in news.iterrows() if i not in done_indices]
    print(f"  已完成: {len(done_indices):,}, 待处理: {len(pending):,}")

    if not pending:
        print("[OK] 全部已完成, 直接聚合...")
    else:
        batches = []
        for i in range(0, len(pending), args.batch_size):
            chunk = pending[i:i + args.batch_size]
            indices = [c[0] for c in chunk]
            titles = [c[1]["title"] for c in chunk]
            batches.append((indices, titles))

        t0 = time.time()
        file_exists = raw_path.exists()
        api_key = DEEPSEEK_API_KEY

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_classify_batch, (bi, titles, bi % 80), api_key): (bi, indices, titles)
                       for bi, (indices, titles) in enumerate(batches)}

            pbar = tqdm(total=len(batches), unit="batch",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                   "[{elapsed}<{remaining}] rpm={postfix[0]} fail={postfix[1]}",
                        postfix=[0, 0])

            for fut in as_completed(futures):
                bi, indices, titles = futures[fut]
                try:
                    parsed = fut.result()
                    ok_count = sum(1 for r in parsed if r is not None)
                except Exception:
                    parsed = [None] * len(titles)
                    ok_count = 0

                rows = []
                for idx, obj in zip(indices, parsed):
                    if obj is None:
                        obj = {"event_type": "other", "sentiment": "neutral",
                               "scope": "firm", "impact_channel": "uncertain",
                               "supply_chain": False}
                    row = news.loc[idx]
                    rows.append({
                        "date": row["date"], "code": row["code"],
                        "source": row["source"], "title": row["title"],
                        "event_type": obj["event_type"], "sentiment": obj["sentiment"],
                        "scope": obj["scope"], "impact_channel": obj["impact_channel"],
                        "supply_chain": obj["supply_chain"],
                    })

                pd.DataFrame(rows).to_csv(
                    raw_path, mode="a" if file_exists else "w",
                    header=not file_exists, index=False)
                file_exists = True

                pbar.update(1)
                pbar.postfix[0] = ok_count

            pbar.close()

        elapsed = time.time() - t0
        print(f"\n[OK] 分类耗时: {elapsed/60:.1f} 分钟")

    # 聚合
    raw_all = pd.read_csv(raw_path)
    print(f"  sentiment_raw.csv: {len(raw_all):,} 行")

    for col in ["event_type", "sentiment", "scope", "impact_channel"]:
        dist = raw_all[col].value_counts().to_dict()
        s = "  ".join(f"{k}:{v}" for k, v in sorted(dist.items(), key=lambda x: -x[1]))
        print(f"  {col}: {s}")
    sc = raw_all["supply_chain"].astype(bool).sum()
    print(f"  supply_chain=true: {sc} ({sc/len(raw_all):.1%})")

    feats = _aggregate_to_quarterly(raw_all)
    feat_path = DATA_DIR / "sentiment_features.csv"
    feats.to_csv(feat_path, index=False, encoding="utf-8-sig")
    print(f"\n[OK] sentiment_features.csv: {len(feats):,} × {len(feats.columns)}")
    print(f"     股票: {feats['code'].nunique():,}, 季度: {feats['quarter'].nunique()}")


if __name__ == "__main__":
    main()
