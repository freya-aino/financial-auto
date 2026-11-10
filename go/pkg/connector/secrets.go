package connector

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Secrets is the parsed config/secrets.toml (D-020): a flat map of
// "<section>.<key>" → value. Only the subset of TOML the file uses is
// understood: `[section]` / `[a.b]` headers, `key = "string"` lines, comments.
type Secrets struct {
	m map[string]string
}

// LoadSecrets reads a secrets file. A missing file is not an error (all
// values then come from the environment); malformed lines are.
func LoadSecrets(path string) (*Secrets, error) {
	s := &Secrets{m: map[string]string{}}
	if path == "" {
		return s, nil
	}
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return s, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	section := ""
	sc := bufio.NewScanner(f)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.TrimSpace(line[1 : len(line)-1])
			continue
		}
		k, v, ok := strings.Cut(line, "=")
		if !ok {
			return nil, fmt.Errorf("%s:%d: expected key = value", path, n)
		}
		k, v = strings.TrimSpace(k), strings.TrimSpace(v)
		if strings.HasPrefix(v, "\"") {
			if end := strings.Index(v[1:], "\""); end >= 0 {
				v = v[:end+2] // drop anything after the closing quote (comments)
			}
			uq, err := strconv.Unquote(v)
			if err != nil {
				return nil, fmt.Errorf("%s:%d: bad string: %w", path, n, err)
			}
			v = uq
		} else {
			if i := strings.Index(v, "#"); i >= 0 {
				v = strings.TrimSpace(v[:i])
			}
			v = strings.Trim(v, "'")
		}
		key := k
		if section != "" {
			key = section + "." + k
		}
		s.m[key] = v
	}
	return s, sc.Err()
}

// Get returns `[section] key` from the file, with the environment variable
// FA_<SECTION>_<KEY> taking precedence (D-020). For `[sources.lemon]` the
// env name is FA_LEMON_API_KEY (the `sources.` prefix is dropped).
func (s *Secrets) Get(section, key string) string {
	if v := os.Getenv(EnvName(section, key)); v != "" {
		return v
	}
	if s == nil {
		return ""
	}
	return s.m[section+"."+key]
}

// ForSource returns every secret of `[sources.<id>]` merged with FA_<ID>_*
// environment overrides, ready for SourceConfig.Secrets.
func (s *Secrets) ForSource(id string) map[string]string {
	out := map[string]string{}
	prefix := "sources." + id + "."
	if s != nil {
		for k, v := range s.m {
			if strings.HasPrefix(k, prefix) {
				out[strings.TrimPrefix(k, prefix)] = v
			}
		}
	}
	envPrefix := EnvName("sources."+id, "")
	for _, kv := range os.Environ() {
		k, v, _ := strings.Cut(kv, "=")
		if strings.HasPrefix(k, envPrefix) && v != "" {
			out[strings.ToLower(strings.TrimPrefix(k, envPrefix))] = v
		}
	}
	return out
}

// EnvName maps (section, key) to the FA_<SOURCE>_<KEY> override variable.
func EnvName(section, key string) string {
	section = strings.TrimPrefix(section, "sources.")
	up := func(x string) string {
		return strings.ToUpper(strings.NewReplacer(".", "_", "-", "_").Replace(x))
	}
	if key == "" {
		return "FA_" + up(section) + "_"
	}
	return "FA_" + up(section) + "_" + up(key)
}

// Secret looks a key up in cfg.Secrets, then FA_<ID>_<KEY>.
func (cfg SourceConfig) Secret(key string) string {
	if v, ok := cfg.Secrets[key]; ok && v != "" {
		return v
	}
	return os.Getenv(EnvName(cfg.Source.ID, key))
}
