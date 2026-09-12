#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import time

from kafka import KafkaProducer


def clean_column_name(name):
    result = name.strip().lower()
    for ch in [" ", ".", "-", "/", "(", ")"]:
        result = result.replace(ch, "_")
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def parse_value(value):
    try:
        if value is None or value == "":
            return 0.0
        return float(value) if "." in value or "e" in value.lower() else int(value)
    except (ValueError, TypeError):
        return value


def main():
    parser = argparse.ArgumentParser(description="Replay CSV rows to Kafka")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="ids-network-flow")
    parser.add_argument("--rate", type=float, default=100.0, help="Rows/s; <=0 sends as fast as possible")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=1000)
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks=1,
        retries=3,
        batch_size=65536,
        linger_ms=10,
    )

    delay = 1.0 / args.rate if args.rate > 0 else 0.0
    total = 0
    started = time.time()

    try:
        with open(args.csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = {name: clean_column_name(name) for name in reader.fieldnames}
            for row in reader:
                payload = {
                    fieldnames[name]: parse_value(value)
                    for name, value in row.items()
                }
                payload["_timestamp"] = time.time()
                producer.send(args.topic, key=str(total), value=payload)
                total += 1

                if args.flush_every and total % args.flush_every == 0:
                    producer.flush()
                    elapsed = time.time() - started
                    print(f"[{total:,}] rate={total / elapsed:.1f} rows/s")

                if args.limit and total >= args.limit:
                    break
                if delay > 0:
                    time.sleep(delay)
    finally:
        producer.flush()
        producer.close()

    elapsed = time.time() - started
    print(f"[OK] Sent {total:,} rows in {elapsed:.1f}s ({total / elapsed:.1f} rows/s)")


if __name__ == "__main__":
    main()
