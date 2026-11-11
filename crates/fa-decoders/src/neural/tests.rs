//! Conformance tests of the neural families on SYNTHETIC data (spec 34 §9.1: forward, loss,
//! save/load, cost estimate) and of the D-009 feedback API.

use std::sync::Arc;

use candle_core::{Device, Tensor, Var};
use fa_core::rng::Rng;

use super::{feedback, *};
use crate::{LatentPoint, crps_quantiles};

/// SYNTHETIC regime series: the drift sign flips in blocks of 150 steps; the latent carries the
/// regime (noisy), and every 10th step a news item whose wording matches the regime.
fn regime_data(n: usize, seed: u64) -> (Series, DecodeCtx, TextTimeline) {
    let mut rng = Rng::seed(seed);
    let mut lp = 4.6;
    let mut ts = Vec::with_capacity(n);
    let mut log_price = Vec::with_capacity(n);
    let mut ctx = DecodeCtx::default();
    let mut text = TextTimeline::new();
    for i in 0..n {
        let r = if (i / 150) % 2 == 0 { 1.0 } else { -1.0 };
        lp += 0.0005 * r + 0.001 * rng.normal();
        ts.push(i as i64 * 1000);
        log_price.push(lp);
        let lat = vec![(r + 0.1 * rng.normal()) as f32, (0.1 * rng.normal()) as f32, 0.5];
        ctx.push("inst-syn", LatentPoint { as_of_ms: i as i64 * 1000, vec: lat });
        if i % 10 == 0 {
            let words = if r > 0.0 {
                "stocks rally strong gains record surge"
            } else {
                "shares plunge weak losses slump crash"
            };
            text.push(i as i64 * 1000, words, "");
        }
    }
    text.finish();
    (Series { instrument_id: "inst-syn".into(), ts_ms: ts, log_price, step_ms: 1000 }, ctx, text)
}

fn fit(genome: &str, n: usize) -> (NeuralDecoder, Series) {
    let (s, ctx, text) = regime_data(n, 11);
    let g = DecoderGenome::parse(genome).unwrap();
    let d = train(&g, std::slice::from_ref(&s), Arc::new(ctx), Arc::new(text), &TrainOptions::default()).unwrap();
    (d, s)
}

/// Mean CRPS of the decoder vs the naive zero forecast over the evaluation window.
fn crps_skill(d: &NeuralDecoder, s: &Series) -> (f64, usize) {
    let h = d.genome().horizon;
    let (mut c, mut naive, mut n) = (0.0, 0.0, 0);
    for t in 0..s.len() {
        let Some(f) = d.forecast(s, t) else { continue };
        let Some(y) = s.forward_return(t, h) else { continue };
        assert!(s.ts_ms[t] >= d.meta.cutoff_ms, "forecast inside the training window");
        assert!(f.quantiles.windows(2).all(|w| w[0] <= w[1] + 1e-12), "{:?}", f.quantiles);
        c += crps_quantiles(&f.quantiles, &f.levels, y);
        naive += y.abs();
        n += 1;
    }
    (1.0 - c / naive, n)
}

#[test]
fn nn_forecast_learns_the_regime_from_the_latent() {
    let (d, s) = fit("nn_forecast:h=5;lb=16;width=16;depth=2;steps=250;lr=0.01;train=0.6", 2400);
    let r = &d.meta.train;
    assert!(r.train_loss < r.initial_loss, "{r:?}");
    assert_eq!(r.latent_dim, 3);
    assert!(r.latent_coverage > 0.99);
    let (skill, n) = crps_skill(&d, &s);
    assert!(n > 800, "{n}");
    assert!(skill > 0.1, "skill {skill}");
    assert!(d.ops_per_forecast(s.len()) > 100.0);
    // without the latent the regime is much harder to see
    let (s2, ctx, text) = regime_data(2400, 11);
    let g = DecoderGenome::parse("nn_forecast:h=5;lb=16;width=16;depth=2;steps=250;lr=0.01;train=0.6;lat=0").unwrap();
    let blind = train(&g, std::slice::from_ref(&s2), Arc::new(ctx), Arc::new(text), &TrainOptions::default()).unwrap();
    assert_eq!(blind.meta.layout.latent, 0);
    let (blind_skill, _) = crps_skill(&blind, &s2);
    assert!(skill > blind_skill, "{skill} vs blind {blind_skill}");
}

#[test]
fn gru_prior_trains_and_is_point_in_time() {
    let (d, s) = fit("nn_forecast:h=3;lb=12;width=12;depth=1;gru=1;steps=60;lr=0.01", 1200);
    assert!(d.meta.train.train_loss < d.meta.train.initial_loss, "{:?}", d.meta.train);
    let t = s.len() - 50;
    let f = d.forecast(&s, t).unwrap();
    let mut shorter = s.clone();
    shorter.log_price.truncate(t + 1);
    shorter.ts_ms.truncate(t + 1);
    let g = d.forecast(&shorter, t).unwrap();
    for (a, b) in f.quantiles.iter().zip(&g.quantiles) {
        assert!((a - b).abs() < 1e-9, "future data changed the forecast");
    }
}

#[test]
fn nn_retrieval_respects_the_gap_and_has_skill() {
    let (d, s) = fit("nn_retrieval:h=4;lb=12;k=20;gap=6;emb=8;width=16;steps=150;lr=0.01;temp=0.2", 2400);
    assert!(d.meta.train.train_loss < d.meta.train.initial_loss, "{:?}", d.meta.train);
    let mut checked = 0;
    for t in (0..s.len()).step_by(37) {
        let Some(f) = d.forecast(&s, t) else { continue };
        assert!(!f.analogs.is_empty() && f.analogs.len() <= 20);
        for a in &f.analogs {
            let dt_steps = (a.time_distance_days * 86_400.0).round() as usize;
            assert!(dt_steps >= 6 + 4, "analog too recent: {dt_steps} steps");
        }
        checked += 1;
    }
    assert!(checked > 10);
    let (skill, _) = crps_skill(&d, &s);
    assert!(skill > 0.0, "skill {skill}");
}

#[test]
fn visual_families_reconstruct_better_than_the_mean_raster() {
    let (chart, s) = fit("nn_chart:lb=24;res=8;width=32;steps=250;lr=0.01", 2000);
    assert_eq!(chart.meta.layout.seq, 0, "latent-only input");
    let r = chart.recon_eval(std::slice::from_ref(&s), chart.meta.cutoff_ms, 5000).unwrap().unwrap();
    assert!(r.n > 500 && r.skill > 0.0, "{r:?} {:?}", chart.meta.train);
    assert!(chart.forecast(&s, s.len() - 10).is_none(), "chart does not forecast");

    let (future, s) = fit("nn_future:h=5;lb=16;res=8;width=32;steps=250;lr=0.01", 2000);
    let r = future.recon_eval(std::slice::from_ref(&s), future.meta.cutoff_ms, 5000).unwrap().unwrap();
    assert!(r.skill > 0.0, "{r:?} {:?}", future.meta.train);
    let (skill, n) = crps_skill(&future, &s);
    assert!(n > 500 && skill > 0.0, "future-chart quantiles: skill {skill}");
}

#[test]
fn text_families_reconstruct_news_from_the_latent() {
    let (news, s) = fit("nn_news:lb=8;tdim=64;tw=0.5;width=32;steps=250;lr=0.01", 2000);
    let r = news.recon_eval(std::slice::from_ref(&s), news.meta.cutoff_ms, 5000).unwrap().unwrap();
    assert!(r.n > 200 && r.skill > 0.05, "{r:?}");
    let (narr, s) = fit("nn_narrative:lb=8;tdim=64;tw=0.5;width=32;steps=250;lr=0.01", 2000);
    assert_eq!(narr.meta.layout.extra, 64);
    let r = narr.recon_eval(std::slice::from_ref(&s), narr.meta.cutoff_ms, 5000).unwrap().unwrap();
    assert!(r.n > 200 && r.skill > 0.0, "{r:?}");
    // serving: sentiment score follows the regime
    let (_, ctx, text) = regime_data(2000, 11);
    let up_t = 150 * 12 + 100; // inside an up block (blocks of 150 alternate, 12 is even)
    let mut cut = s.clone();
    cut.log_price.truncate(up_t + 1);
    cut.ts_ms.truncate(up_t + 1);
    let lat = ctx.latent_as_of("inst-syn", cut.ts_ms[up_t]).map(|l| l.vec.clone());
    let served = news.serve(&cut, lat.as_deref(), &text).unwrap();
    assert!(served.score > 0.0, "{served:?}");
    assert!((-1.0..=1.0).contains(&served.score));
}

#[test]
fn bundles_round_trip_and_training_is_deterministic() {
    let (d, s) = fit("nn_forecast:h=3;lb=12;width=8;depth=1;steps=40", 900);
    let bytes = d.to_bundle().unwrap();
    let meta = NeuralDecoder::meta_of(&bytes).unwrap();
    // compared after the same JSON round trip (serde_json may move an f64 by one ulp)
    let expect: BundleMeta = serde_json::from_slice(&serde_json::to_vec(&d.meta).unwrap()).unwrap();
    assert_eq!(meta, expect);
    let (_, ctx, text) = regime_data(900, 11);
    let loaded = NeuralDecoder::from_bundle(&bytes, &Device::Cpu)
        .unwrap()
        .with_context(Arc::new(ctx), Arc::new(text))
        .with_eval_from(Some(d.meta.cutoff_ms));
    assert_eq!(loaded.n_params(), d.n_params());
    let t = s.len() - 20;
    assert_eq!(loaded.forecast(&s, t), d.forecast(&s, t));
    let (again, _) = fit("nn_forecast:h=3;lb=12;width=8;depth=1;steps=40", 900);
    assert_eq!(again.forecast(&s, t), d.forecast(&s, t), "same genome, same weights");
    assert!(NeuralDecoder::meta_of(b"not a bundle").is_err());
}

#[test]
fn warm_start_copies_matching_variables() {
    let (parent, _) = fit("nn_forecast:h=3;lb=12;width=8;depth=1;steps=40", 900);
    let (s, ctx, text) = regime_data(900, 11);
    let g = DecoderGenome::parse("nn_forecast:h=3;lb=12;width=8;depth=1;steps=1;seed=99").unwrap();
    let opts = TrainOptions { warm_start: Some(parent.to_bundle().unwrap()), ..Default::default() };
    let child = train(&g, std::slice::from_ref(&s), Arc::new(ctx), Arc::new(text), &opts).unwrap();
    assert_eq!(child.meta.train.warm_started, 4, "l0 + out weights and biases");
    let wide = DecoderGenome::parse("nn_forecast:h=3;lb=12;width=16;depth=1;steps=1").unwrap();
    let (s, ctx, text) = regime_data(900, 11);
    let c2 = train(&wide, std::slice::from_ref(&s), Arc::new(ctx), Arc::new(text), &opts).unwrap();
    assert_eq!(c2.meta.train.warm_started, 1, "only the out bias keeps its shape");
}

#[test]
fn interruption_and_insufficient_data() {
    let (s, ctx, text) = regime_data(900, 1);
    let g = DecoderGenome::parse("nn_forecast:steps=5000;lb=12").unwrap();
    let opts = TrainOptions { should_stop: Some(Box::new(|| true)), ..Default::default() };
    let d = train(&g, std::slice::from_ref(&s), Arc::new(ctx.clone()), Arc::new(text.clone()), &opts).unwrap();
    assert!(d.meta.train.interrupted && d.meta.train.steps == 0);
    let mut tiny = s.clone();
    tiny.log_price.truncate(30);
    tiny.ts_ms.truncate(30);
    assert!(matches!(
        train(&g, &[tiny], Arc::new(ctx), Arc::new(text), &TrainOptions::default()),
        Err(NeuralError::InsufficientData(_))
    ));
    let base = DecoderGenome::parse("forecast").unwrap();
    assert!(matches!(
        train(&base, &[s], Arc::default(), Arc::default(), &TrainOptions::default()),
        Err(NeuralError::NotNeural(_))
    ));
}

#[test]
fn feedback_gradients_reach_the_latents_only() {
    let mut bundles = Vec::new();
    let mut cards = Vec::new();
    for (i, genome) in [
        "nn_forecast:h=3;lb=12;width=8;steps=60;lr=0.01",
        "nn_retrieval:h=3;lb=12;gap=3;emb=4;width=8;steps=30",
        "nn_chart:lb=12;res=6;width=8;steps=60",
        "nn_news:lb=8;tdim=32;tw=0.5;width=8;steps=60",
    ]
    .iter()
    .enumerate()
    {
        let (mut d, _) = fit(genome, 900);
        d.meta.score = 0.1 + i as f64 * 0.01;
        d.meta.objectives.insert("crps_skill".into(), d.meta.score);
        let key = feedback::bundle_key("gen-test", &d.meta, &format!("job{i}"));
        assert!(key.starts_with("decoders/gen-test/") && key.ends_with(".safetensors"), "{key}");
        assert!(feedback::card_key(&key).ends_with(".card.json"));
        cards.push(feedback::EliteCard {
            meta: d.meta.clone(),
            bucket: feedback::BUCKET.into(),
            key,
            created_ms: i as i64,
        });
        bundles.push((d.to_bundle().unwrap(), 1.0));
    }
    // an unconditioned decoder is not a feedback elite
    let (s, ctx, text) = regime_data(900, 11);
    let blind_g = DecoderGenome::parse("nn_forecast:lb=12;width=8;steps=5;lat=0").unwrap();
    let blind = train(
        &blind_g,
        std::slice::from_ref(&s),
        Arc::new(ctx.clone()),
        Arc::new(text.clone()),
        &TrainOptions::default(),
    )
    .unwrap();
    bundles.push((blind.to_bundle().unwrap(), 1.0));

    let top = feedback::select_top_k(&cards, 1, Some(3));
    assert_eq!(top.len(), 4, "one per output family");
    let w = feedback::quota_weights(&top);
    assert!((w.iter().map(|x| x.1).sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(feedback::select_top_k(&cards, 1, Some(7)).is_empty(), "dimension filter");

    let set = feedback::FeedbackSet::load(&bundles, 3, &Device::Cpu).unwrap();
    assert_eq!(set.elites.len(), 4);
    assert_eq!(set.skipped.len(), 1, "{:?}", set.skipped);
    let steps: Vec<(&Series, usize)> = (100..164).map(|t| (&s, t * 5)).collect();
    let batch = set.prepare(&steps, &text).unwrap();
    assert!(batch.rows_per_elite().iter().all(|n| *n > 0), "{:?}", batch.rows_per_elite());
    let lat: Vec<f32> =
        steps.iter().flat_map(|(s, t)| ctx.latent_as_of("inst-syn", s.ts_ms[*t]).unwrap().vec.clone()).collect();
    let latents = Var::from_tensor(&Tensor::from_vec(lat, (steps.len(), 3), &Device::Cpu).unwrap()).unwrap();
    let out = set.aux_loss(latents.as_tensor(), &batch).unwrap();
    assert_eq!(out.parts.len(), 4, "{:?}", out.parts.iter().map(|p| &p.1).collect::<Vec<_>>());
    let total = out.total.to_scalar::<f32>().unwrap();
    assert!(total.is_finite() && total > 0.0);
    let grads = out.total.backward().unwrap();
    let g = grads.get(latents.as_tensor()).expect("gradient on the latents");
    let gnorm = g.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
    assert!(gnorm > 0.0, "zero gradient");
    // weights are constants: nothing in the elites is a Var
    for e in &set.elites {
        assert!(e.decoder.params.iter().all(|(_, t)| !t.is_variable()));
    }
    // a dimension mismatch is an error, not a panic
    let wrong = Tensor::zeros((steps.len(), 5), candle_core::DType::F32, &Device::Cpu).unwrap();
    assert!(set.aux_loss(&wrong, &batch).is_err());
}

#[test]
fn latent_only_chart_falls_back_when_latents_start_after_the_cutoff() {
    let (s, full, text) = regime_data(900, 4);
    // latents only in the last 10 % of the window (e.g. a worker that just started)
    let mut late = DecodeCtx::default();
    for p in full.latents.values().flatten().filter(|p| p.as_of_ms >= 810_000) {
        late.push("inst-syn", p.clone());
    }
    let g = DecoderGenome::parse("nn_chart:lb=12;res=6;width=8;steps=20").unwrap();
    let d = train(&g, std::slice::from_ref(&s), Arc::new(late), Arc::new(text), &TrainOptions::default()).unwrap();
    assert_eq!(d.meta.layout.latent, 0, "unconditioned fallback");
    assert_eq!(d.meta.layout.seq, 12);
    assert_eq!(d.genome().latent_weight, 0.0);
}

/// First-generation bundles (latent normaliser v1, written before `norm_version` existed) are
/// refused for serving / feedback and listed for re-training (D-068 open item).
#[test]
fn stale_normaliser_bundles_are_refused_and_listed() {
    let (mut d, _) = fit("nn_forecast:h=3;lb=12;width=8;depth=1;steps=20", 900);
    assert_eq!(d.meta.norm_version, NORM_VERSION);
    // a card JSON from before the field existed deserialises as v1
    let mut json = serde_json::to_value(&d.meta).unwrap();
    json.as_object_mut().unwrap().remove("norm_version");
    let old: BundleMeta = serde_json::from_value(json).unwrap();
    assert_eq!(old.norm_version, 1);
    assert!(!old.is_current());
    d.meta.score = 0.2;
    let fresh = feedback::EliteCard { meta: d.meta.clone(), bucket: "b".into(), key: "new".into(), created_ms: 2 };
    let stale = feedback::EliteCard {
        meta: BundleMeta { score: 0.3, ..old },
        bucket: "b".into(),
        key: "old".into(),
        created_ms: 1,
    };
    let top = feedback::select_top_k(&[stale.clone(), fresh.clone()], 3, None);
    assert_eq!(top.iter().map(|c| c.key.as_str()).collect::<Vec<_>>(), vec!["new"], "stale cards never serve");
    assert_eq!(
        feedback::stale_cards(&[stale.clone(), fresh]).iter().map(|c| c.key.as_str()).collect::<Vec<_>>(),
        vec!["old"]
    );
    // the bundle itself is refused by from_bundle, still readable for inspection / warm starts
    d.meta.norm_version = 1;
    let bytes = d.to_bundle().unwrap();
    assert!(matches!(NeuralDecoder::from_bundle(&bytes, &Device::Cpu), Err(NeuralError::StaleNormaliser(1))));
    assert!(NeuralDecoder::from_bundle_any(&bytes, &Device::Cpu).is_ok());
    let set = feedback::FeedbackSet::load(&[(bytes, 1.0)], 3, &Device::Cpu).unwrap();
    assert!(set.is_empty() && set.skipped[0].1.contains("stale"), "{:?}", set.skipped);
}

/// An expert-masked elite reads only its experts' columns of the encoder latent.
#[test]
fn masked_feedback_selects_expert_columns() {
    let (s, mut ctx, text) = regime_data(900, 11);
    ctx.set_layout("inst-syn", vec![(0, 2), (1, 1)]);
    let g = DecoderGenome::parse("nn_forecast:h=3;lb=12;width=8;steps=40;mask=1").unwrap();
    let d =
        train(&g, std::slice::from_ref(&s), Arc::new(ctx.clone()), Arc::new(text.clone()), &TrainOptions::default())
            .unwrap();
    assert_eq!((d.meta.layout.latent, d.meta.full_latent_dim()), (2, 3));
    let mut card = feedback::EliteCard { meta: d.meta.clone(), bucket: "b".into(), key: "k".into(), created_ms: 0 };
    card.meta.score = 0.1;
    assert_eq!(feedback::select_top_k(std::slice::from_ref(&card), 1, Some(3)).len(), 1, "full dimension matches");
    assert!(feedback::select_top_k(&[card], 1, Some(2)).is_empty());
    let set = feedback::FeedbackSet::load(&[(d.to_bundle().unwrap(), 1.0)], 3, &Device::Cpu).unwrap();
    assert_eq!(set.elites.len(), 1, "{:?}", set.skipped);
    let steps: Vec<(&Series, usize)> = (100..140).map(|t| (&s, t * 5)).collect();
    let batch = set.prepare(&steps, &text).unwrap();
    let lat: Vec<f32> =
        steps.iter().flat_map(|(s, t)| ctx.latent_as_of("inst-syn", s.ts_ms[*t]).unwrap().vec.clone()).collect();
    let latents = Var::from_tensor(&Tensor::from_vec(lat, (steps.len(), 3), &Device::Cpu).unwrap()).unwrap();
    let out = set.aux_loss(latents.as_tensor(), &batch).unwrap();
    let grads = out.total.backward().unwrap();
    let gm = grads.get(latents.as_tensor()).unwrap().sqr().unwrap().sum_keepdim(0).unwrap().to_vec2::<f32>().unwrap();
    assert!(gm[0][0] > 0.0 || gm[0][1] > 0.0, "{gm:?}");
    assert_eq!(gm[0][2], 0.0, "the unmasked expert gets no gradient: {gm:?}");
    // serving picks the same experts from a latent state
    let e0 = [1.0f32, 0.5];
    let e1 = [9.0f32];
    assert_eq!(d.select_latent([(1, &e1[..]), (0, &e0[..])]), Some(vec![1.0, 0.5]));
    assert_eq!(d.select_latent([(1, &e1[..])]), None);
}

/// Successive-halving rungs train a fraction of the genome's steps.
#[test]
fn steps_override_limits_training() {
    let (s, ctx, text) = regime_data(900, 11);
    let g = DecoderGenome::parse("nn_forecast:h=3;lb=12;width=8;steps=400").unwrap();
    let opts = TrainOptions { steps: Some(25), ..Default::default() };
    let d = train(&g, std::slice::from_ref(&s), Arc::new(ctx), Arc::new(text), &opts).unwrap();
    assert_eq!(d.meta.train.steps, 25);
    assert!(d.meta.genome.contains("steps=400"), "the genome keeps its full-fidelity steps");
}

/// `text.summary` (Q2: templated summaries of the window + its news): the latent carries the
/// regime, so the move / close / news slots beat the training-mean distribution out of sample;
/// the served sentence and score follow the regime.
#[test]
fn summary_family_decodes_the_window_story_from_the_latent() {
    let (d, s) = fit("nn_summary:lb=24;tdim=64;tw=0.5;width=32;steps=300;lr=0.01", 2000);
    assert_eq!(d.meta.layout.seq, 0, "latent-only input");
    assert_eq!(d.meta.layout.out, crate::summary::width());
    let r = d.recon_eval(std::slice::from_ref(&s), d.meta.cutoff_ms, 5000).unwrap().unwrap();
    assert!(r.n > 500 && r.skill > 0.0, "{r:?} {:?}", d.meta.train);
    assert!(d.forecast(&s, s.len() - 10).is_none(), "summary does not forecast");
    let (_, ctx, text) = regime_data(2000, 11);
    let up_t = 150 * 12 + 100;
    let lat = ctx.latent_as_of("inst-syn", s.ts_ms[up_t]).map(|l| l.vec.clone());
    let served = d.serve_at(&s, up_t, lat.as_deref(), &text).unwrap();
    assert!((0.0..=1.0).contains(&served.score), "intensity on the baseline's [0, 1] scale");
    assert!(!served.text.is_empty() && served.components.len() == 2 + crate::summary::SLOTS.len(), "{served:?}");
    // without a news source (empty timeline) there is no target: such a run sits out
    let g = DecoderGenome::parse("nn_summary:lb=24").unwrap();
    assert!(data::target_at(&g, &s, 100, 1e-3, &TextTimeline::new()).is_none());
}

/// `visual.frame`: frame embeddings from the frame timeline (synthetic byte patterns that follow
/// the regime); never read from a news timeline, and news families never read frames.
#[test]
fn frame_family_reconstructs_frames_and_keeps_timelines_apart() {
    let (s, ctx, news) = regime_data(2000, 11);
    let mut frames = TextTimeline::frames();
    for t in (0..s.len()).step_by(5) {
        let up = (t / 150) % 2 == 0;
        let bytes: Vec<u8> = (0..256).map(|j| if up { (j * 3 % 251) as u8 } else { (j * 7 % 13) as u8 }).collect();
        frames.push_features(s.ts_ms[t], data::frame_features(&bytes), &[]);
    }
    frames.finish();
    let g = DecoderGenome::parse("nn_frame:lb=8;tdim=64;tw=0.2;width=32;steps=250;lr=0.01").unwrap();
    let ctx = Arc::new(ctx);
    let d = train(&g, std::slice::from_ref(&s), ctx.clone(), Arc::new(frames), &TrainOptions::default()).unwrap();
    let r = d.recon_eval(std::slice::from_ref(&s), d.meta.cutoff_ms, 5000).unwrap().unwrap();
    assert!(r.n > 200 && r.skill > 0.05, "{r:?}");
    // a news timeline gives a frame decoder no target (and a frame timeline none to nn_news)
    assert!(train(&g, std::slice::from_ref(&s), ctx.clone(), Arc::new(news), &TrainOptions::default()).is_err());
    let gn = DecoderGenome::parse("nn_news:lb=8;tdim=64;tw=0.2").unwrap();
    assert!(data::target_at(&gn, &s, 500, 1e-3, &TextTimeline::frames()).is_none());
    assert!(data::frame_features(&[1, 2]).is_empty());
}

/// The residual start: a reconstruction decoder begins exactly at its training-mean baseline,
/// so with zero SGD steps its out-of-sample skill is 0, never below.
#[test]
fn reconstruction_decoders_start_at_the_baseline() {
    for genome in ["nn_news:lb=8;tdim=64;tw=0.5;steps=1;lr=1e-12", "nn_chart:lb=16;res=6;steps=1;lr=1e-12"] {
        let (d, s) = fit(genome, 1500);
        let r = d.recon_eval(std::slice::from_ref(&s), d.meta.cutoff_ms, 5000).unwrap().unwrap();
        assert!(r.skill.abs() < 1e-4, "{genome}: {r:?}");
    }
}

/// Spec 34 §8: re-evaluated elites decay in the feedback weights by their transfer ratio, a
/// decoder that no longer beats naive drops out, and the §6.4 item 5 check counts them.
#[test]
fn transfer_records_decay_feedback_weights() {
    let (d, _) = fit("nn_forecast:h=3;lb=12;width=8;depth=1;steps=20", 600);
    let card = |key: &str, fam: &str, score: f64, ratio: Option<f64>| {
        let mut meta = BundleMeta { score, output_family: fam.into(), ..d.meta.clone() };
        if let Some(r) = ratio {
            meta.transfer.push(Transfer { snapshot_id: "s2".into(), score: score * r, base_score: score, ratio: r, ..Default::default() });
        }
        feedback::EliteCard { meta, bucket: "b".into(), key: key.into(), created_ms: 0 }
    };
    let cards = vec![
        card("a", "rnn_prior", 0.3, Some(1.05)),
        card("b", "rnn_prior", 0.2, Some(0.5)),
        card("c", "retrieval", 0.3, Some(-0.2)),
        card("d", "retrieval", 0.1, None),
    ];
    let top = feedback::select_top_k(&cards, 4, None);
    assert!(top.iter().all(|c| c.key != "c"), "failed transfer is no elite");
    let w = feedback::quota_weights(&top);
    let get = |k: &str| w.iter().find(|(c, _)| c.key == k).map(|x| x.1).unwrap();
    assert!((w.iter().map(|x| x.1).sum::<f64>() - 1.0).abs() < 1e-12);
    // rank weights inside rnn_prior are 2/3 and 1/3 of the family share; b halves
    assert!((get("a") / get("b") - 2.0 / 0.5).abs() < 1e-9, "{w:?}");
    assert_eq!(BundleMeta { transfer: vec![], ..d.meta.clone() }.transfer_factor(), 1.0);
    let chk = feedback::transfer_check(&cards, 4, "s2", TRANSFER_TOL);
    assert_eq!((chk.elites, chk.evaluated, chk.passed), (4, 3, 1));
    assert!((chk.worst_ratio + 0.2).abs() < 1e-12);
}
