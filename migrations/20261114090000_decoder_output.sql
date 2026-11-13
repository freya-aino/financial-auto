-- Decoder outputs as control-api receives them on `decoder.output.>` (spec 51 §6.3 region
-- detail). Written by control-api (`bins/control-api/src/decoder_store.rs`) through a bounded,
-- non-blocking writer (outputs are dropped, not queued without bound, when Postgres lags);
-- rows older than 30 days (`as_of`) are deleted hourly by the same module.
-- `GET /api/latent/regions/{id}/reconstructions` joins the region's episodes to these rows on
-- snapshot + timescale + instrument + `as_of` within the episode window. Idempotent.

create schema if not exists wh;

create table if not exists wh.decoder_output (
    id            bigserial   primary key,
    decoder       text        not null,                     -- DecoderOutput.decoder_id
    genome        text        not null default '',          -- DecoderOutput.genome_id
    family        text        not null,                     -- visual.chart | retrieval | text.news | …
    snapshot      text        not null default '',          -- '' = output not bound to a snapshot
    timescale     text        not null,
    instrument_id text,                                     -- null = cross-asset / market-wide output
    as_of         timestamptz not null,
    output        jsonb       not null default '{}'::jsonb, -- {output_refs, quantiles, analogs}
    recorded_at   timestamptz not null default now()
);

create index if not exists decoder_output_space_inst_as_of
    on wh.decoder_output (snapshot, timescale, instrument_id, as_of desc);
create index if not exists decoder_output_as_of on wh.decoder_output (as_of);
