//! Readable names for generated identities (D-111): `nimble-elephant` rather than a bare hash.
//!
//! Every generated identity in the platform — genomes, allocators, decoders, experts — was a
//! prefix plus a hex digest. Those are unique and content-addressed but unreadable and, worse,
//! unmemorable: two genomes differing in one nibble look identical at a glance in a log line, an
//! audit row or the evolution UI.
//!
//! A name here is **derived from the identity's own hash, never drawn at random**. That keeps
//! every property the hex digest had: the same genome content yields the same name on every host
//! and every run, nothing has to be coordinated or persisted to allocate one, and the name of a
//! thing can always be recomputed from the thing. Reproducibility is the whole point of a
//! content-addressed id, so a random name would have quietly broken it.
//!
//! The digest is **kept** alongside the name ([`tagged`]) rather than replaced by it. 64 × 64 =
//! 4096 name pairs would collide constantly across tens of thousands of genomes; the name is the
//! human handle and the digest is still what makes the id unique.
//!
//! ```
//! # use fa_core::names::{name_from_hash, tagged, title_case};
//! assert_eq!(tagged("gp-", 0x6433_eb2d_deed_df72), "gp-curious-serval-6433eb2ddeeddf72");
//! assert_eq!(title_case(&name_from_hash(0x6433_eb2d_deed_df72)), "Curious Serval");
//! ```

/// Attributes, 64 of them so six bits of the hash select one.
pub const ATTRIBUTES: [&str; 64] = [
    "agile", "amber", "ancient", "arctic", "ashen", "azure", "bold", "brave", "brisk", "bronze", "calm", "clever",
    "cobalt", "coral", "crimson", "curious", "daring", "deft", "dusky", "eager", "elder", "ember", "fabled",
    "fearless", "fleet", "frosty", "gentle", "gilded", "golden", "granite", "hardy", "hidden", "humble", "indigo",
    "ivory", "jade", "keen", "lively", "lucid", "lunar", "mellow", "merry", "mighty", "nimble", "noble", "patient",
    "placid", "polar", "proud", "quick", "quiet", "rapid", "royal", "rustic", "sage", "scarlet", "serene", "sharp",
    "silent", "silver", "sleek", "solar", "steady", "stormy",
];

/// Animals, likewise 64.
pub const ANIMALS: [&str; 64] = [
    "adder", "albatross", "antelope", "badger", "bison", "bittern", "boar", "caribou", "cheetah", "cobra", "condor",
    "coyote", "crane", "dingo", "dolphin", "dormouse", "eagle", "egret", "elephant", "elk", "falcon", "ferret",
    "finch", "fox", "gazelle", "gecko", "gibbon", "heron", "ibex", "ibis", "jackal", "jaguar", "kestrel",
    "kingfisher", "kite", "lemur", "leopard", "lynx", "magpie", "marten", "mantis", "narwhal", "nightjar", "ocelot",
    "osprey", "otter", "owl", "panther", "pelican", "petrel", "puffin", "puma", "raven", "rook", "sable", "salmon",
    "seal", "serval", "shrike", "stoat", "tapir", "tern", "vulture", "walrus",
];

/// FNV-1a, the same digest the genome crates already use for content addressing, so a name can be
/// derived from a seed string without pulling in a hasher dependency.
pub fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Avalanche finaliser (splitmix64's). FNV-1a alone diffuses its input badly across the word for
/// short, near-identical seeds — slicing its bits directly gave only 603 distinct names out of
/// 4096 over 20 000 seeds, and `snap-1/e3` and `snap-1/e4` landed on the same name. Mixing first
/// costs two multiplies and restores a flat distribution. The raw digest is still what
/// [`tagged`] prints, so content addressing is untouched.
fn mix(h: u64) -> u64 {
    let mut z = h;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// `attribute-animal` for a hash that has already been computed — the usual entry point, because
/// the caller is normally minting a content-addressed id and already holds its digest.
pub fn name_from_hash(h: u64) -> String {
    let m = mix(h);
    let a = ATTRIBUTES[(m & 63) as usize];
    let b = ANIMALS[((m >> 6) & 63) as usize];
    format!("{a}-{b}")
}

/// `attribute-animal` for an arbitrary seed string (an expert uid, a space id, …).
pub fn name_for(seed: &str) -> String {
    name_from_hash(fnv1a(seed))
}

/// `<prefix><attribute>-<animal>-<hash>`: the readable handle with the digest kept for uniqueness.
/// `prefix` carries its own separator, as the existing id prefixes do (`gp-`, `nn-`, `alloc-`).
pub fn tagged(prefix: &str, h: u64) -> String {
    format!("{prefix}{}-{h:016x}", name_from_hash(h))
}

/// Same, seeded from a string.
pub fn tagged_for(prefix: &str, seed: &str) -> String {
    tagged(prefix, fnv1a(seed))
}

/// `nimble-elephant` → `Nimble Elephant`, for UI and log lines.
pub fn title_case(name: &str) -> String {
    name.split(['-', ' '])
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The readable part of a tagged id, without prefix or digest: `gp-silent-ibex-6433…` →
/// `silent-ibex`. Returns `None` when `s` is not shaped like a tagged id (e.g. a legacy
/// hash-only id), which is what makes the change backward compatible at read time.
pub fn name_of(s: &str) -> Option<&str> {
    let (rest, digest) = s.rsplit_once('-')?;
    if digest.len() < 8 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let (rest, animal) = rest.rsplit_once('-')?;
    let (_, attribute) = rest.rsplit_once('-').unwrap_or(("", rest));
    if !ANIMALS.contains(&animal) || !ATTRIBUTES.contains(&attribute) {
        return None;
    }
    let start = s.len() - digest.len() - animal.len() - attribute.len() - 2;
    Some(&s[start..s.len() - digest.len() - 1])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn word_lists_are_unique_and_lowercase() {
        for list in [&ATTRIBUTES[..], &ANIMALS[..]] {
            let set: HashSet<_> = list.iter().collect();
            assert_eq!(set.len(), list.len(), "duplicate word in list");
            assert!(list.iter().all(|w| w.chars().all(|c| c.is_ascii_lowercase())), "non-lowercase word");
        }
    }

    /// The whole point: the same content must always get the same name, on every host and run.
    #[test]
    fn names_are_deterministic() {
        assert_eq!(name_for("snap-1/e3"), name_for("snap-1/e3"));
        assert_eq!(name_from_hash(42), name_from_hash(42));
        assert_ne!(name_for("snap-1/e3"), name_for("snap-1/e4"));
    }

    /// The digest survives, so two genomes that differ anywhere still get different ids even when
    /// the 4096-name space collides.
    #[test]
    fn tagged_keeps_the_digest() {
        let id = tagged("gp-", 0x6433_eb2d_deed_df72);
        assert!(id.starts_with("gp-"), "{id}");
        assert!(id.ends_with("6433eb2ddeeddf72"), "{id}");
        assert_ne!(tagged("gp-", 1), tagged("gp-", 2));
    }

    #[test]
    fn names_spread_over_the_whole_list() {
        let names: HashSet<String> = (0..20_000u64).map(|i| name_from_hash(fnv1a(&i.to_string()))).collect();
        assert!(names.len() > 4_000, "only {} distinct names of 4096", names.len());
    }

    #[test]
    fn title_case_reads_as_prose() {
        assert_eq!(title_case("nimble-elephant"), "Nimble Elephant");
        assert_eq!(title_case("quick-fox"), "Quick Fox");
    }

    #[test]
    fn name_of_extracts_and_rejects() {
        let id = tagged("gp-", 0x6433_eb2d_deed_df72);
        assert_eq!(name_of(&id), Some(name_from_hash(0x6433_eb2d_deed_df72).as_str()));
        assert_eq!(name_of("gp-6433eb2ddeeddf72"), None, "legacy hash-only id is not a tagged id");
        assert_eq!(name_of("alloc-not-a-name-6433eb2ddeeddf72"), None);
        assert_eq!(name_of("plain"), None);
    }

    #[test]
    fn name_of_handles_a_bare_name_without_prefix() {
        let h = 0x1234_5678_9abc_def0;
        let bare = format!("{}-{h:016x}", name_from_hash(h));
        assert_eq!(name_of(&bare), Some(name_from_hash(h).as_str()));
    }
}
