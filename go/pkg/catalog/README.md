# catalog

Loads and validates `config/sources.yaml`, syncs it into Postgres.

**Status:** implemented — loads and validates `config/sources.yaml` (unique ids, enum values); optional `rate_limit` field with a default when absent.

## Owns
Catalog model and loader.

## Governing specs
- [11-source-catalog](../../../specs/11-source-catalog.md)
