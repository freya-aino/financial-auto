package refdata

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/fey/financial-auto/go/pkg/catalog"
)

// Overrides returns the runtime rate-limit / config overrides of catalog.source
// (ctl.ingest.configure, spec 11 §4). Like the status, a stored override wins
// over the YAML seed on every ingestor start and reload.
func (s *Store) Overrides(ctx context.Context) (map[string]catalog.Override, error) {
	rows, err := s.Pool.Query(ctx, `select id, rate_limit, config from catalog.source where rate_limit is not null or config is not null`)
	if err != nil {
		return nil, fmt.Errorf("refdata: overrides: %w", err)
	}
	defer rows.Close()
	out := map[string]catalog.Override{}
	for rows.Next() {
		var id string
		var rl, cfg []byte
		if err := rows.Scan(&id, &rl, &cfg); err != nil {
			return nil, err
		}
		var o catalog.Override
		if len(rl) > 0 {
			o.RateLimit = &catalog.RateLimit{}
			if err := json.Unmarshal(rl, o.RateLimit); err != nil {
				return nil, fmt.Errorf("refdata: override %s rate_limit: %w", id, err)
			}
		}
		if len(cfg) > 0 {
			o.Config = &catalog.Params{}
			if err := json.Unmarshal(cfg, o.Config); err != nil {
				return nil, fmt.Errorf("refdata: override %s config: %w", id, err)
			}
		}
		out[id] = o
	}
	return out, rows.Err()
}

// SetOverride stores the non-nil parts of an override; clear=true resets both
// columns to NULL first (the YAML values apply again after the next reload).
func (s *Store) SetOverride(ctx context.Context, id string, o catalog.Override, clear bool) error {
	var rl, cfg any
	if o.RateLimit != nil {
		b, _ := json.Marshal(o.RateLimit)
		rl = b
	}
	if o.Config != nil {
		b, _ := json.Marshal(o.Config)
		cfg = b
	}
	q := `update catalog.source set rate_limit = coalesce($2::jsonb, rate_limit), config = coalesce($3::jsonb, config), updated_at = now() where id = $1`
	if clear {
		q = `update catalog.source set rate_limit = $2::jsonb, config = $3::jsonb, updated_at = now() where id = $1`
	}
	tag, err := s.Pool.Exec(ctx, q, id, rl, cfg)
	if err != nil {
		return fmt.Errorf("refdata: set override: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("refdata: set override: source %q not in catalog.source", id)
	}
	return nil
}

// SetStatusNote persists a status with its reason (auto-disable, spec 11 §6.1/§8).
func (s *Store) SetStatusNote(ctx context.Context, id, status, note string) error {
	tag, err := s.Pool.Exec(ctx, `update catalog.source set status = $2, status_note = $3, status_changed_at = now(), updated_at = now() where id = $1`, id, status, note)
	if err != nil {
		return fmt.Errorf("refdata: set status: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("refdata: set status: source %q not in catalog.source", id)
	}
	return nil
}
