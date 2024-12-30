use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BatchSize, BenchmarkGroup,
    BenchmarkId, Criterion,
};
use p3::{
    baby_bear::BabyBear,
    challenger::{CanObserve, GenericChallenger},
    field::{
        extension::BinomialExtensionField, BinaryField, ExtensionField, Field, FromUniformBytes,
        TwoAdicField,
    },
    keccak::Keccak256Hash,
    matrix::dense::{DenseMatrix, RowMajorMatrix},
    merkle_tree::{keccak256_merkle_tree, Keccak256MerkleTreeMmcs},
    polyval::Polyval,
};
use poly_commit::{
    basefold::{
        code::{BinaryReedSolomonCode, GenericRandomFoldableCode, ReedSolomonCode},
        Basefold, BasefoldConfig,
    },
    MultiPolyEvalPoint, PolyCommitScheme, PolyEvals,
};
use rand::{distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng};
use util::Itertools;

fn basefold_generic<F, E, const C: usize, const LOG2_K_0: usize>(
    log_n: usize,
    rng: &mut StdRng,
) -> Basefold<F, E, GenericRandomFoldableCode<F>, Keccak256MerkleTreeMmcs<F>>
where
    F: Field + FromUniformBytes,
    E: ExtensionField<F>,
{
    let d = log_n - LOG2_K_0;
    let code = GenericRandomFoldableCode::reed_solomon_g_0_with_random_ts(128, C, LOG2_K_0, d, rng);
    let config = BasefoldConfig {
        code,
        mmcs: keccak256_merkle_tree(),
    };
    PolyCommitScheme::setup(config).unwrap()
}

fn basefold_reed_solomon<F, E, const C: usize>(
    log_n: usize,
    _: &mut StdRng,
) -> Basefold<F, E, ReedSolomonCode<F>, Keccak256MerkleTreeMmcs<F>>
where
    F: TwoAdicField + Ord,
    E: ExtensionField<F>,
{
    let code = ReedSolomonCode::new(128, C, log_n);
    let config = BasefoldConfig {
        code,
        mmcs: keccak256_merkle_tree(),
    };
    PolyCommitScheme::setup(config).unwrap()
}

fn basefold_binary_reed_solomon<F, E, const C: usize>(
    log_n: usize,
    _: &mut StdRng,
) -> Basefold<F, E, BinaryReedSolomonCode<F>, Keccak256MerkleTreeMmcs<F>>
where
    F: BinaryField,
    E: ExtensionField<F>,
{
    let code = BinaryReedSolomonCode::new(128, C, log_n);
    let config = BasefoldConfig {
        code,
        mmcs: keccak256_merkle_tree(),
    };
    PolyCommitScheme::setup(config).unwrap()
}

macro_rules! run_all {
    ($group:expr, $(($name:literal, $pcs:expr),)*) => {
        let mut group= $group;
        $(run(&mut group, $name, $pcs);)*
    };
}

fn bench_commit(c: &mut Criterion) {
    fn run<F, E, P>(
        group: &mut BenchmarkGroup<impl Measurement>,
        name: impl AsRef<str>,
        f: impl Fn(usize, &mut StdRng) -> P,
    ) where
        F: Field,
        E: ExtensionField<F>,
        P: PolyCommitScheme<F, E, Data = RowMajorMatrix<F>>,
        Standard: Distribution<F>,
    {
        let mut rng = StdRng::from_entropy();
        (16..20)
            .cartesian_product(8..9)
            .for_each(|(log_n, log_polys)| {
                let param = format!("log_n={}/log_polys={}", log_n, log_polys);
                let id = BenchmarkId::new(name.as_ref(), param);
                let pcs = f(log_n, &mut rng);
                group.bench_function(id, |b| {
                    b.iter_batched(
                        || DenseMatrix::rand(&mut rng, 1 << log_n, 1 << log_polys),
                        |data| pcs.commit(data).unwrap(),
                        BatchSize::LargeInput,
                    );
                });
            })
    }

    run_all!(
        c.benchmark_group("commit"),
        (
            "basefold_generic_baby_bear",
            basefold_generic::<BabyBear, BinomialExtensionField<BabyBear, 5>, 1, 3>
        ),
        (
            "basefold_reed_solomon_baby_bear",
            basefold_reed_solomon::<BabyBear, BinomialExtensionField<BabyBear, 5>, 1>
        ),
        (
            "basefold_binary_reed_solomon_polyval",
            basefold_binary_reed_solomon::<Polyval, Polyval, 1>
        ),
    );
}

fn bench_open(c: &mut Criterion) {
    fn run<F, E, P>(
        group: &mut BenchmarkGroup<impl Measurement>,
        name: impl AsRef<str>,
        f: impl Fn(usize, &mut StdRng) -> P,
    ) where
        F: Field + FromUniformBytes,
        E: ExtensionField<F> + FromUniformBytes,
        P: PolyCommitScheme<F, E, Data = RowMajorMatrix<F>, Point = MultiPolyEvalPoint<E>>,
        Standard: Distribution<F>,
        GenericChallenger<F, Keccak256Hash>: CanObserve<P::Commitment>,
    {
        let mut rng = StdRng::from_entropy();
        (16..20)
            .cartesian_product(8..9)
            .for_each(|(log_n, log_polys)| {
                let param = format!("log_n={}/log_polys={}", log_n, log_polys);
                let id = BenchmarkId::new(name.as_ref(), param);
                let pcs = f(log_n, &mut rng);
                let data = DenseMatrix::rand(&mut rng, 1 << log_n, 1 << log_polys);
                let comm_data = pcs.commit(data).unwrap();
                let evals = {
                    let point = E::random_vec(log_n, &mut rng);
                    let values = pcs.eval(pcs.data(&comm_data), &point);
                    [PolyEvals { point, values }]
                };
                group.bench_function(id, |b| {
                    b.iter(|| {
                        pcs.open(&comm_data, &evals, GenericChallenger::keccak256())
                            .unwrap()
                    });
                });
            })
    }

    run_all!(
        c.benchmark_group("open"),
        (
            "basefold_generic_baby_bear",
            basefold_generic::<BabyBear, BinomialExtensionField<BabyBear, 5>, 1, 3>
        ),
        (
            "basefold_reed_solomon_baby_bear",
            basefold_reed_solomon::<BabyBear, BinomialExtensionField<BabyBear, 5>, 1>
        ),
        (
            "basefold_binary_reed_solomon_polyval",
            basefold_binary_reed_solomon::<Polyval, Polyval, 1>
        ),
    );
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_commit, bench_open,
);
criterion_main!(benches);
