use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use p3::{
    baby_bear::BabyBear,
    field::{extension::BinomialExtensionField, ExtPackedValue, FieldAlgebra, FromUniformBytes},
    poly::multilinear::{eq_expand, evaluate, fix_last_var, MultiPoly},
};
use rand::{rngs::StdRng, SeedableRng};
use util::enumerate;

type F = BabyBear;
type E = BinomialExtensionField<F, 5>;

fn bench_eq(c: &mut Criterion) {
    fn eq(r: &[E]) -> Vec<E> {
        let mut evals = E::zero_vec(1 << r.len());
        evals[0] = E::ONE;
        enumerate(r).for_each(|(i, r_i)| eq_expand(&mut evals, *r_i, i));
        evals
    }

    let mut group = c.benchmark_group("eq");
    (16..20).for_each(|num_vars| {
        let r = E::random_vec(num_vars, StdRng::from_entropy());
        let param = format!("num_vars={num_vars}");
        group.bench_function(BenchmarkId::new("simple", &param), |b| {
            b.iter(|| eq(black_box(&r)))
        });
        group.bench_function(BenchmarkId::new("packing", &param), |b| {
            b.iter(|| MultiPoly::<F, E>::eq(black_box(&r), E::ONE))
        });
    });
}

fn bench_fix_last_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("fix_last_var");
    (16..20).for_each(|num_vars| {
        let param = format!("num_vars={num_vars}");
        let evals = E::random_vec(1 << num_vars, StdRng::from_entropy());
        let packed = MultiPoly::<F, E>::ext_packing(ExtPackedValue::<F, E>::ext_pack_slice(&evals));
        let r_i = E::random(StdRng::from_entropy());
        group.bench_function(BenchmarkId::new("simple", &param), |b| {
            b.iter_batched(
                || evals.clone(),
                |mut evals| fix_last_var(&mut evals, black_box(r_i)),
                BatchSize::LargeInput,
            )
        });
        group.bench_function(BenchmarkId::new("packing", &param), |b| {
            b.iter_batched(
                || packed.clone(),
                |mut packed| packed.fix_last_var(black_box(r_i)),
                BatchSize::LargeInput,
            )
        });
    });
}

fn bench_evaluate(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate");
    (16..20).for_each(|num_vars| {
        let param = format!("num_vars={num_vars}");
        let evals = E::random_vec(1 << num_vars, StdRng::from_entropy());
        let packed = MultiPoly::<F, E>::ext_packing(ExtPackedValue::<F, E>::ext_pack_slice(&evals));
        let r = E::random_vec(num_vars, StdRng::from_entropy());
        group.bench_function(BenchmarkId::new("simple", &param), |b| {
            b.iter(|| evaluate(&evals, black_box(&r)))
        });
        group.bench_function(BenchmarkId::new("packing", &param), |b| {
            b.iter(|| packed.evaluate(black_box(&r)))
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = bench_eq, bench_fix_last_var, bench_evaluate,
);
criterion_main!(benches);
