"""Helper for connecting to Azure Cosmos DB containers."""

import os
from azure.cosmos import CosmosClient

# Read connection information from environment
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
# Support both naming conventions used across this repo.
COSMOS_DATABASE = (
    os.getenv("COSMOS_DATABASE")
    or os.getenv("COSMOS_DB_NAME")
    or "MyDatabase"
)

if not COSMOS_ENDPOINT or not COSMOS_KEY:
    raise ValueError("COSMOS_ENDPOINT and COSMOS_KEY environment variables must be set")

_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)


def container(name: str):
    """Return a container client for the given container name.

    The database name is taken from the COSMOS_DATABASE environment variable.
    """
    db = _client.get_database_client(COSMOS_DATABASE)
    return db.get_container_client(name)
