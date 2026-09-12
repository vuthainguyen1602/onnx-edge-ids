#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import NoBrokersAvailable, TopicAlreadyExistsError, TopicAlreadyExistsError


def create_topics(bootstrap, topics, partitions, replication_factor=1, retries=30):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            admin = KafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=10000)
            try:
                admin.create_topics(
                    [
                        NewTopic(
                            name=topic,
                            num_partitions=partitions,
                            replication_factor=replication_factor,
                        )
                        for topic in topics
                    ],
                    validate_only=False,
                )
                print(f"[OK] Created topics: {', '.join(topics)}")
            except TopicAlreadyExistsError:
                print("[INFO] Topics already exist")
            finally:
                admin.close()
            return
        except NoBrokersAvailable as exc:
            last_error = exc
            if attempt < retries:
                print(f"[WAIT] Kafka not ready at {bootstrap} ({attempt}/{retries})")
                time.sleep(2.0)

    raise last_error or NoBrokersAvailable()


def main():
    parser = argparse.ArgumentParser(description="Create ONNX-EdgeIDS Kafka topics")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--input-topic", default="ids-network-flow")
    parser.add_argument("--suspicious-topic", default="ids-suspicious-flow")
    parser.add_argument("--partitions", type=int, default=2)
    args = parser.parse_args()

    create_topics(
        args.bootstrap,
        [args.input_topic, args.suspicious_topic],
        max(1, args.partitions),
    )


if __name__ == "__main__":
    main()
