"""One place to open a Neo4j connection, so the check, the loader and the AI tools all
connect the same way.

Why certifi: AuraDB's certificate chains up to "SSL.com TLS RSA Root CA 2022". Some Windows
machines do not have that root in their trusted list, so Python rejects the connection with
"self-signed certificate in certificate chain". The certifi package ships an up-to-date list
of trusted roots, and we tell the driver to use it.

The driver only accepts a custom trust list on the plain `neo4j://` / `bolt://` schemes, so
for `neo4j+s://` / `bolt+s://` we switch to the plain scheme and turn encryption on ourselves.
The connection is exactly as secure: still encrypted, still fully verified.
"""

from __future__ import annotations

from typing import Any

import certifi
from neo4j import Driver, GraphDatabase, TrustCustomCAs

from sales_poc.config import Neo4jSettings, get_neo4j_settings

_VERIFIED_TLS_SCHEMES = {"neo4j+s": "neo4j", "bolt+s": "bolt"}


def driver_arguments(uri: str) -> tuple[str, dict[str, Any]]:
    """The URI and extra driver options to use for a NEO4J_URI from .env."""
    scheme, sep, rest = uri.partition("://")
    if sep and scheme in _VERIFIED_TLS_SCHEMES:
        return (
            f"{_VERIFIED_TLS_SCHEMES[scheme]}://{rest}",
            {"encrypted": True, "trusted_certificates": TrustCustomCAs(certifi.where())},
        )
    # Anything else (e.g. a local neo4j://localhost) is passed through unchanged.
    return uri, {}


def driver(settings: Neo4jSettings | None = None) -> Driver:
    """A Neo4j driver. Close it when done (use it in a `with` block)."""
    s = settings or get_neo4j_settings()
    uri, options = driver_arguments(s.neo4j_uri)
    return GraphDatabase.driver(
        uri,
        auth=(s.neo4j_user, s.neo4j_password.get_secret_value()),
        **options,
    )
