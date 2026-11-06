# bus

NATS publish helpers and Go protobuf types (generated into `pb/`).

**Status:** implemented — subject taxonomy mirroring `crates/fa-bus/src/subjects.rs`, envelope→subject routing, protobuf `Publisher` (with `Nats-Msg-Id` for jobs), heartbeat loop + counters. Generated code in `pb/` (buf, committed; D-002).

## Owns
Subject helpers mirrored from `fa-bus`.

## Governing specs
- [21-bus-and-contracts](../../../specs/21-bus-and-contracts.md)
