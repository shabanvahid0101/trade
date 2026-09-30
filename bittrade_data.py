"""Public BitTrade spot candles. No credentials or order endpoints.

REST returns a recent window, not paginated history. Overlap is merged by candle
start time; missing older history is reported, never filled from another venue.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://api-cloud.bittrade.co.jp"
PERIODS = {"5m": ("5min", 300), "15m": ("15min", 900), "1h": ("60min", 3600)}
FIELDS = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "count"]


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


class PublicClient:
    def get(self, path, **params):
        url = BASE + path + ("?" + urlencode(params) if params else "")
        for attempt in range(4):
            try:
                # Stay well below the documented 10 public requests/sec/IP.
                time.sleep(0.2)
                request = Request(url, headers={"User-Agent": "BitTradeResearch/1.0"})
                with urlopen(request, timeout=30) as response:
                    result = json.load(response)
                if result.get("status") != "ok":
                    raise ValueError(f"BitTrade API error: {result}")
                return result
            except (OSError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)


def validate_rows(rows, seconds):
    seen = set()
    for row in rows:
        stamp = datetime.fromisoformat(row["timestamp"])
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("Candle timestamps must be UTC")
        t = stamp.timestamp()
        if t % seconds or t in seen:
            raise ValueError("Duplicate or misaligned candle")
        seen.add(t)
        o, h, l, c, v, q, n = [float(row[k]) for k in FIELDS[1:]]
        if not all(math.isfinite(x) for x in (o, h, l, c, v, q, n)):
            raise ValueError("Non-finite candle value")
        if min(o, h, l, c) <= 0 or l > min(o, c) or h < max(o, c) or h < l:
            raise ValueError("Invalid OHLC values")
        if min(v, q, n) < 0 or n != int(n):
            raise ValueError("Invalid volume or count")


def merge_candles(old, candles, seconds, server_ms):
    new = []
    for c in candles:
        t = int(c["id"])
        if t + seconds > server_ms / 1000:
            continue  # The exchange's current candle is unfinished.
        new.append(dict(zip(FIELDS, [utc(t), c["open"], c["high"], c["low"],
                                     c["close"], c["amount"], c["vol"], c["count"]])))
    validate_rows(old, seconds)
    validate_rows(new, seconds)
    if not new:
        raise ValueError("No closed candles returned")
    merged = {row["timestamp"]: row for row in old}
    merged.update({row["timestamp"]: row for row in new})
    rows = sorted(merged.values(), key=lambda row: row["timestamp"])
    stamps = [datetime.fromisoformat(row["timestamp"]).timestamp() for row in rows]
    age = server_ms / 1000 - (stamps[-1] + seconds)
    if age < 0 or age > 2 * seconds:
        raise ValueError(f"Stale or future data: last close age {age}s")
    gaps = [{"after": utc(a), "before": utc(b), "missing_candles": int((b-a)/seconds)-1}
            for a, b in zip(stamps, stamps[1:]) if b-a != seconds]
    return rows, {"rows": len(rows), "first": rows[0]["timestamp"], "last": rows[-1]["timestamp"],
                  "missing_candles": sum(g["missing_candles"] for g in gaps), "gaps": gaps,
                  "zero_volume_candles": sum(float(r["volume"]) == 0 for r in rows),
                  "last_close_age_seconds": age}


def collect(root, symbols, timeframes, client=None):
    root = Path(root)
    client = client or PublicClient()
    markets = client.get("/v1/common/symbols")["data"]
    tickers = client.get("/market/tickers")
    write_json(root / "markets.json", {"fetched_at": utc(time.time()), "data": markets})
    write_json(root / "tickers.json", tickers)
    by_symbol = {m["symbol"]: m for m in markets}
    if symbols == ["all-online"]:
        symbols = sorted(m["symbol"] for m in markets if m["state"] == "online")
    report = {"provider": "bittrade", "market_type": "spot", "updated_at": utc(time.time()),
              "datasets": {}, "errors": {}, "history": "Latest 2000 candles per REST request; accumulated locally"}
    for symbol in symbols:
        for timeframe in timeframes:
            key = f"{symbol}_{timeframe}"
            try:
                market = by_symbol[symbol]
                if market["state"] != "online":
                    raise ValueError("Market is not online")
                period, seconds = PERIODS[timeframe]
                path = root / f"{key}.csv"
                manifest_path = root / f"{key}.source.json"
                identity = {"provider": "bittrade", "market_type": "spot", "symbol": symbol,
                            "timeframe": timeframe, "timezone": "UTC"}
                if path.exists() and not manifest_path.exists():
                    raise ValueError("Existing CSV has no source manifest")
                if manifest_path.exists() and json.loads(manifest_path.read_text())["source"] != identity:
                    raise ValueError("Dataset source mismatch")
                old = []
                if path.exists():
                    with path.open(newline="") as handle:
                        old = list(csv.DictReader(handle))
                response = client.get("/market/history/kline", symbol=symbol, period=period, size=2000)
                if response.get("ch") != f"market.{symbol}.kline.{period}":
                    raise ValueError("Unexpected candle topic")
                rows, quality = merge_candles(old, response["data"], seconds, response["ts"])
                output = io.StringIO(newline="")
                writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                content = output.getvalue()
                atomic_text(path, content)
                write_json(manifest_path, {"source": identity, "endpoint": BASE + "/market/history/kline",
                    "sha256": hashlib.sha256(content.encode()).hexdigest(), "server_timestamp_ms": response["ts"],
                    "api_trading": market.get("api-trading"), "quality": quality})
                report["datasets"][key] = quality
                print(f"{key}: {quality['rows']} closed candles; missing={quality['missing_candles']}", flush=True)
                if quality["missing_candles"]:
                    report["errors"][key] = "History contains gaps; research must use a contiguous suffix"
            except (ValueError, OSError, KeyError, TypeError) as exc:
                report["errors"][key] = str(exc)
                print(f"ERROR {key}: {exc}", flush=True)
    write_json(root / "collection_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dataset/bittrade")
    parser.add_argument("--symbols", nargs="+", default=["btcjpy", "ethjpy", "xrpjpy"])
    parser.add_argument("--timeframes", nargs="+", choices=PERIODS, default=list(PERIODS))
    args = parser.parse_args()
    result = collect(args.output, args.symbols, args.timeframes)
    raise SystemExit(1 if result["errors"] else 0)


if __name__ == "__main__":
    main()
