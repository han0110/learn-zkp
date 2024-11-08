use crate::{basefold::code::RandomFoldableCode, MultiPolyEvalPoint, PolyCommitScheme, PolyEvals};
use core::{fmt::Debug, iter::repeat_with, mem::take};
use p3::{
    challenger::{CanObserve, FieldChallenger, FieldChallengerExt},
    commit::{ExtensionMmcs, Mmcs},
    field::{ExtensionField, Field},
    matrix::{dense::RowMajorMatrix, Dimensions},
    poly::multilinear::{eq_eval, evaluate, interpolate, transpose, MultiPoly},
};
use sumcheck::{
    function::eval::{Eval, EvalProver},
    prove_sumcheck_round, verify_sumcheck_round, SumcheckProof, SumcheckSubclaim,
};
use util::{cloned, izip, rev, Itertools};

mod code;

pub use code::GenericRandomFoldableCode;

#[derive(Clone, Debug)]
pub struct Basefold<F, E, R, M> {
    code: R,
    mmcs: M,
    mmcs_ext: ExtensionMmcs<F, E, M>,
}

#[derive(Clone, Debug)]
pub struct BasefoldConfig<R, M> {
    pub code: R,
    pub mmcs: M,
}

#[derive(Debug, derive_more::AsRef)]
pub struct BasefoldCommitmentData<F, M>
where
    F: Field,
    M: Mmcs<F>,
{
    #[as_ref]
    comm: M::Commitment,
    codewords: M::ProverData<RowMajorMatrix<F>>,
    data: RowMajorMatrix<F>,
}

#[allow(clippy::type_complexity)]
#[derive(Clone, Debug)]
pub struct BasefoldProof<F: Field, E: ExtensionField<F>, M: Mmcs<F>> {
    sum_check: SumcheckProof<E>,
    final_poly: Vec<E>,
    pi_comms: Vec<M::Commitment>,
    openings: Vec<(
        (Vec<Vec<F>>, M::Proof),
        Vec<(Vec<Vec<E>>, <M as Mmcs<F>>::Proof)>,
    )>,
}

#[derive(Clone, Debug)]
pub enum BasefoldError<F: Field, M: Mmcs<F>> {
    Mmcs(M::Error),
}

impl<F, E, R, M, C> PolyCommitScheme<E, C> for Basefold<F, E, R, M>
where
    F: Field,
    E: ExtensionField<F>,
    R: RandomFoldableCode<F>,
    M: Mmcs<F, Commitment: Debug, ProverData<RowMajorMatrix<F>>: Debug, Proof: Debug> + Debug,
    C: FieldChallenger<F> + CanObserve<M::Commitment>,
{
    type Config = BasefoldConfig<R, M>;
    // TODO: Support multi-matrix with different dimension.
    // TODO: Support coefficients or evaluations as input.
    /// Evaluations of multilinear polynomials in row major matrix.
    type Data = RowMajorMatrix<F>;
    type Commitment = M::Commitment;
    type CommitmentData = BasefoldCommitmentData<F, M>;
    type Point = MultiPolyEvalPoint<E>;
    type Proof = BasefoldProof<F, E, M>;
    type Error = BasefoldError<F, M>;

    fn setup(config: Self::Config) -> Result<Self, Self::Error> {
        Ok(Self {
            code: config.code,
            mmcs: config.mmcs.clone(),
            mmcs_ext: ExtensionMmcs::new(config.mmcs),
        })
    }

    fn commit(&self, data: Self::Data) -> Result<Self::CommitmentData, Self::Error> {
        let codewords = self.code.batch_encode(data.as_view());
        let (comm, codewords) = self.mmcs.commit_matrix(codewords);
        Ok(Self::CommitmentData {
            comm,
            codewords,
            data,
        })
    }

    fn data<'a>(&self, comm_data: &'a Self::CommitmentData) -> &'a Self::Data {
        &comm_data.data
    }

    fn eval(&self, data: &Self::Data, point: &Self::Point) -> Vec<E> {
        transpose(data)
            .into_iter()
            .map(|poly| MultiPoly::base(poly).evaluate(point))
            .collect()
    }

    fn open(
        &self,
        comm_data: Self::CommitmentData,
        evals: &[PolyEvals<E, Self::Point>],
        mut challenger: C,
    ) -> Result<Self::Proof, Self::Error> {
        let polys = transpose(&comm_data.data);

        #[cfg(debug_assertions)]
        evals.iter().for_each(|eval| {
            izip!(&polys, &eval.values).for_each(|(poly, value)| {
                assert_eq!(MultiPoly::base(poly).evaluate(&eval.point), *value);
            });
        });

        // TODO: Batch by sumcheck
        assert_eq!(polys.len(), 1);
        assert_eq!(evals.len(), 1);

        let poly = &polys[0];
        let point = &evals[0].point;
        let num_vars = point.len();
        let claim = evals[0].values[0];

        let mut pi = cloned(&self.mmcs.get_matrices(&comm_data.codewords)[0].values)
            .map(E::from)
            .collect_vec();
        let mut sumcheck_prover =
            EvalProver::new(Eval::new(num_vars, point), MultiPoly::base(poly));
        let mut sumcheck_subclaim = SumcheckSubclaim::new(claim);

        let (pi_comms, compressed_round_polys) = rev(0..self.code.d())
            .map(|i| {
                let pi_comm = self.mmcs_ext.commit_matrix(RowMajorMatrix::new(
                    izip!(&pi[..pi.len() / 2], &pi[pi.len() / 2..])
                        .flat_map(|(l, r)| [*l, *r])
                        .collect_vec(),
                    2,
                ));
                challenger.observe(pi_comm.0.clone());

                let compressed_round_poly = prove_sumcheck_round(
                    &mut sumcheck_prover,
                    &mut sumcheck_subclaim,
                    i + self.code.log2_k_0(),
                    &mut challenger,
                );
                let r_i = sumcheck_subclaim.r_rev().last().unwrap();
                pi = self.code.fold(i, take(&mut pi), *r_i);
                (pi_comm, compressed_round_poly)
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let final_poly = sumcheck_prover.into_ext_poly();

        challenger.observe_ext_slice(&final_poly);

        #[cfg(debug_assertions)]
        assert_eq!(self.code.encode0(&interpolate(&final_poly)), pi);

        // TODO: Config num queries
        let num_queries = 10;

        let opening_indices =
            repeat_with(|| challenger.sample_bits(self.code.n_d().ilog2() as usize))
                .take(num_queries)
                .collect_vec();

        let openings = opening_indices
            .iter()
            .map(|idx| {
                let opening = self.mmcs.open_batch(*idx, &comm_data.codewords);
                let opening_ext = izip!(rev(0..self.code.d()), &pi_comms)
                    .map(|(i, (_, pi_i))| {
                        let idx = idx % self.code.n_i(i);
                        self.mmcs_ext.open_batch(idx, pi_i)
                    })
                    .collect_vec();
                (opening, opening_ext)
            })
            .collect_vec();

        Ok(Self::Proof {
            sum_check: SumcheckProof {
                compressed_round_polys,
            },
            final_poly,
            pi_comms: pi_comms.into_iter().map(|(comm, _)| comm).collect(),
            openings,
        })
    }

    fn verify(
        &self,
        comm: Self::Commitment,
        evals: &[PolyEvals<E, Self::Point>],
        proof: &Self::Proof,
        mut challenger: C,
    ) -> Result<(), Self::Error> {
        // TODO: Batch by sumcheck
        assert_eq!(evals.len(), 1);

        let comm = &comm;
        let point = &evals[0].point;
        let num_vars = point.len();
        let claim = evals[0].values[0];

        let sumcheck_func = Eval::new(num_vars, point);
        let mut sumcheck_subclaim = SumcheckSubclaim::new(claim);

        izip!(
            rev(0..self.code.d()),
            &proof.pi_comms,
            &proof.sum_check.compressed_round_polys
        )
        .for_each(|(round, pi_comm, compressed_round_poly)| {
            challenger.observe(pi_comm.clone());

            verify_sumcheck_round(
                &sumcheck_func,
                &mut sumcheck_subclaim,
                round,
                compressed_round_poly,
                &mut challenger,
            )
        });

        let r = sumcheck_subclaim.r().copied().collect_vec();

        challenger.observe_ext_slice(&proof.final_poly);

        let (point_lo, point_hi) = evals[0].point.split_at(self.code.log2_k_0());
        assert_eq!(
            evaluate(&proof.final_poly, point_lo) * eq_eval(point_hi, &r),
            *sumcheck_subclaim
        );

        let pi_0 = self.code.encode0(&interpolate(&proof.final_poly));

        // TODO: Config num queries
        let num_queries = 10;

        let opening_indices =
            repeat_with(|| challenger.sample_bits(self.code.n_d().ilog2() as usize))
                .take(num_queries)
                .collect_vec();

        izip!(&opening_indices, &proof.openings)
            .try_for_each(|(idx, opening)| {
                self.mmcs.verify_batch(
                    comm,
                    &[Dimensions {
                        width: 0,
                        height: self.code.n_d(),
                    }],
                    *idx,
                    &opening.0 .0,
                    &opening.0 .1,
                )?;
                izip!(rev(0..self.code.d()), &proof.pi_comms, &opening.1).try_for_each(
                    |(i, comm, opening)| {
                        let idx = idx % self.code.n_i(i);
                        self.mmcs_ext.verify_batch(
                            comm,
                            &[Dimensions {
                                width: 0,
                                height: self.code.n_i(i),
                            }],
                            idx,
                            &opening.0,
                            &opening.1,
                        )
                    },
                )?;
                assert_eq!(opening.0 .0.len(), 1);
                assert_eq!(opening.0 .0[0].len(), 1);
                let leave = E::from(opening.0 .0[0][0]);
                let value = izip!(rev(0..self.code.d()), rev(&r), &opening.1).fold(
                    leave,
                    |leave, (i, r_i, (values, _))| {
                        let idx = idx % self.code.n_i(i);
                        assert_eq!(values.len(), 1);
                        assert_eq!(values[0].len(), 2);
                        assert!(values[0].contains(&leave));
                        self.code
                            .interpolate(i, idx, values[0][0], values[0][1], *r_i)
                    },
                );
                assert!(pi_0.contains(&value));
                Ok(())
            })
            .map_err(Self::Error::Mmcs)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        basefold::{code::RandomFoldableCode, Basefold, BasefoldConfig, GenericRandomFoldableCode},
        test::run_multi_poly_commit_scheme,
    };
    use p3::{
        field::{
            extension::BinomialExtensionField, BabyBear, ExtensionField, Field, FromUniformBytes,
        },
        matrix::dense::RowMajorMatrix,
        merkle_tree::{keccak256_merkle_tree, Keccak256MerkleTreeMmcs},
    };
    use rand::rngs::StdRng;

    type M<F> = Keccak256MerkleTreeMmcs<F>;

    fn run_basefold<
        F: Field + FromUniformBytes,
        E: ExtensionField<F> + FromUniformBytes,
        R: RandomFoldableCode<F>,
    >(
        f: impl Fn(usize, &mut StdRng) -> R,
    ) {
        run_multi_poly_commit_scheme::<F, E, Basefold<F, E, R, M<F>>>(
            |num_vars, num_polys, rng| {
                let code = f(num_vars, rng);
                let mmcs = keccak256_merkle_tree();
                let config = BasefoldConfig { code, mmcs };
                let polys =
                    RowMajorMatrix::new(F::random_vec(num_polys << num_vars, rng), num_polys);
                (config, polys)
            },
        );
    }

    #[test]
    fn generic_random_foldable_code() {
        fn run<F: Field + FromUniformBytes, E: ExtensionField<F> + FromUniformBytes>(
            log2_c: usize,
            log2_k_0: usize,
        ) {
            run_basefold::<F, E, _>(|num_vars, rng| {
                let d = num_vars.saturating_sub(log2_k_0);
                let log2_k_0 = log2_k_0.min(num_vars);
                GenericRandomFoldableCode::reed_solomon_g_0_with_random_ts(log2_c, log2_k_0, d, rng)
            });
        }

        run::<BabyBear, BabyBear>(1, 1);
        run::<BabyBear, BinomialExtensionField<BabyBear, 5>>(1, 1);
    }
}
