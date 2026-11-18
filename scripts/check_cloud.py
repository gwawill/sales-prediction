"""Check that every cloud service and the local model are reachable and working.

    .venv\\Scripts\\python.exe scripts\\check_cloud.py            (all checks)
    .venv\\Scripts\\python.exe scripts\\check_cloud.py s3 kafka   (only some)

Checks: s3, s3vectors, kafka, neo4j, embeddings, llm.
Exit code 0 means everything passed. Each check cleans up after itself, except Kafka,
which leaves one small message with the key "healthcheck" in the topic.
"""

import sys

from sales_poc.checks import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
