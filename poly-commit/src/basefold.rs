//! Implementation of 2023/1705.

use crate::{basefold::code::RandomFoldableCode, MultiPolyEvalPoint, PolyCommitScheme, PolyEvals};
use core::{fmt::Debug, iter::repeat_with, slice};
use p3::{
    challenger::{CanObserve, FieldChallenger, FieldChallengerExt},
    commit::{ExtensionMmcs, Mmcs},
    field::{dot_product, ExtensionField, Field},
    matrix::{dense::RowMajorMatrix, extension::FlatMatrixView, Dimensions, Matrix},
    poly::multilinear::{eq_eval, evaluate, MultiPoly},
};
use sumcheck::{
    function::{
        eval::{Eval, EvalProver},
        SumcheckFunction, SumcheckFunctionProver,
    },
    prove_sumcheck_round, verify_sumcheck_round, SumcheckProof, SumcheckSubclaim,
};
use util::{cloned, rayon::prelude::*, rev, zip, Itertools};

pub mod code;

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

#[derive(Clone, Debug, derive_more::AsRef)]
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
    sumcheck_proof: SumcheckProof<E>,
    final_poly: Vec<E>,
    pi_comms: Vec<M::Commitment>,
    openings: Vec<(
        (
            Vec<F>,   // leaves
            M::Proof, // proof
        ), // input opening
        Vec<(
            E,        // sibling
            M::Proof, // proof
        )>, // pi openings
    )>,
}

impl<F: Field, E: ExtensionField<F>, M: Mmcs<F>> Default for BasefoldProof<F, E, M> {
    fn default() -> Self {
        Self {
            sumcheck_proof: Default::default(),
            final_poly: Default::default(),
            pi_comms: Default::default(),
            openings: Default::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum BasefoldError<F: Field, M: Mmcs<F>> {
    InvalidFinalPoly,
    InvalidQuery,
    Mmcs(M::Error),
}

impl<F, E, R, M> PolyCommitScheme<F, E> for Basefold<F, E, R, M>
where
    F: Field,
    E: ExtensionField<F>,
    R: RandomFoldableCode<F>,
    M: Mmcs<F, Commitment: Debug, ProverData<RowMajorMatrix<F>>: Debug, Proof: Debug> + Debug,
{
    type Config = BasefoldConfig<R, M>;
    // TODO: Support multi-matrix with different dimension.
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
        let codewords = self.code.encode_batch(data.clone());
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
        data.transpose()
            .row_slices()
            .map(|poly| MultiPoly::base(poly).evaluate(point))
            .collect()
    }

    fn open(
        &self,
        comm_data: &Self::CommitmentData,
        evals: &[PolyEvals<E, Self::Point>],
        mut challenger: impl FieldChallenger<F> + CanObserve<Self::Commitment>,
    ) -> Result<Self::Proof, Self::Error> {
        if evals.is_empty() {
            return Ok(Default::default());
        }

        // TODO: Batch by sumcheck
        debug_assert_eq!(evals.len(), 1);

        #[cfg(debug_assertions)]
        evals.iter().for_each(|eval| {
            zip!(comm_data.data.transpose().row_slices(), &eval.values).for_each(
                |(poly, value)| {
                    debug_assert_eq!(MultiPoly::base(poly).evaluate(&eval.point), *value);
                },
            );
        });

        let num_polys = comm_data.data.width();
        debug_assert!(evals.iter().all(|evals| evals.values.len() == num_polys));

        let point = &evals[0].point;
        let (poly, claim, mut pi) = if num_polys == 1 {
            let poly = MultiPoly::base(comm_data.data.values.as_slice());
            let claim = evals[0].values[0];
            let pi = cloned(&self.mmcs.get_matrices(&comm_data.codewords)[0].values)
                .map(E::from)
                .collect_vec();
            (poly, claim, pi)
        } else {
            let scalars = self.sample_random_scalars_by_eq(num_polys, &mut challenger);
            let poly = MultiPoly::ext(
                comm_data
                    .data
                    .values
                    .par_chunks(num_polys)
                    .map(|row| dot_product(&scalars, row))
                    .collect::<Vec<_>>(),
            );
            let claim = dot_product::<E, E>(&scalars, &evals[0].values);
            let pi = self.mmcs.get_matrices(&comm_data.codewords)[0]
                .values
                .par_chunks(num_polys)
                .map(|row| dot_product(&scalars, row))
                .collect::<Vec<_>>();
            (poly, claim, pi)
        };
        let mut sumcheck_prover = EvalProver::new(Eval::new(point), poly);
        let mut sumcheck_subclaim = SumcheckSubclaim::new(claim);

        let (pi_comms, compressed_round_polys) = self.prove_interleaved_folding_sumcheck(
            &mut pi,
            &mut sumcheck_prover,
            &mut sumcheck_subclaim,
            &mut challenger,
        );

        let final_poly = sumcheck_prover.into_ext_poly();

        challenger.observe_ext_slice(&final_poly);

        debug_assert_eq!(self.code.encode0(&final_poly), pi);

        let opening_indices =
            repeat_with(|| challenger.sample_bits(self.code.n_d().ilog2() as usize))
                .take(self.code.num_queries())
                .collect_vec();

        let openings = opening_indices
            .iter()
            .map(|idx| {
                let opening = {
                    let (mut values, proof) = self.mmcs.open_batch(*idx, &comm_data.codewords);
                    (values.pop().unwrap(), proof)
                };
                let opening_ext = zip!(rev(0..self.code.d()), &pi_comms)
                    .map(|(i, (_, pi_i))| {
                        let sibling = ((idx / self.code.n_i(i)) & 1) ^ 1;
                        let idx = idx % self.code.n_i(i);
                        let (values, proof) = self.mmcs_ext.open_batch(idx, pi_i);
                        (values[0][sibling], proof)
                    })
                    .collect();
                (opening, opening_ext)
            })
            .collect();

        Ok(Self::Proof {
            sumcheck_proof: SumcheckProof {
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
        mut challenger: impl FieldChallenger<F> + CanObserve<Self::Commitment>,
    ) -> Result<(), Self::Error> {
        if evals.is_empty() {
            return Ok(());
        }

        // TODO: Batch by sumcheck
        debug_assert_eq!(evals.len(), 1);

        let num_polys = evals[0].values.len();
        debug_assert!(evals.iter().all(|evals| evals.values.len() == num_polys));

        let comm = &comm;
        let point = &evals[0].point;
        let (scalars, claim) = if num_polys == 1 {
            (vec![E::ONE], evals[0].values[0])
        } else {
            let scalars = self.sample_random_scalars_by_eq(num_polys, &mut challenger);
            let claim = dot_product::<E, E>(&scalars, &evals[0].values);
            (scalars, claim)
        };
        let sumcheck_func = Eval::new(point);
        let mut sumcheck_subclaim = SumcheckSubclaim::new(claim);

        let r = self.verify_interleaved_folding_sumcheck(
            &sumcheck_func,
            &mut sumcheck_subclaim,
            &proof.pi_comms,
            &proof.sumcheck_proof,
            &mut challenger,
        );

        challenger.observe_ext_slice(&proof.final_poly);

        let (point_lo, point_hi) = evals[0].point.split_at(self.code.log2_k_0());
        if evaluate(&proof.final_poly, point_lo) * eq_eval(point_hi, &r) != *sumcheck_subclaim {
            return Err(Self::Error::InvalidFinalPoly);
        }

        let pi_0 = self.code.encode0(&proof.final_poly);

        let opening_indices =
            repeat_with(|| challenger.sample_bits(self.code.n_d().ilog2() as usize))
                .take(self.code.num_queries())
                .collect_vec();

        zip!(&opening_indices, &proof.openings).try_for_each(|(idx, (opening, opening_ext))| {
            self.mmcs
                .verify_batch(
                    comm,
                    &[Dimensions {
                        width: 0,
                        height: self.code.n_d(),
                    }],
                    *idx,
                    slice::from_ref(&opening.0),
                    &opening.1,
                )
                .map_err(Self::Error::Mmcs)?;
            let leave = dot_product(&scalars[..num_polys], &opening.0);
            let folded = zip!(rev(0..self.code.d()), rev(&r), &proof.pi_comms, opening_ext)
                .try_fold(leave, |folded, (i, r_i, comm, opening_ext)| {
                    let sibling = ((idx / self.code.n_i(i)) & 1) ^ 1;
                    let idx = idx % self.code.n_i(i);
                    let mut values = vec![folded; 2];
                    values[sibling] = opening_ext.0;
                    self.mmcs_ext
                        .verify_batch(
                            comm,
                            &[Dimensions {
                                width: 0,
                                height: self.code.n_i(i),
                            }],
                            idx,
                            slice::from_ref(&values),
                            &opening_ext.1,
                        )
                        .map_err(Self::Error::Mmcs)?;
                    Ok(self.code.interpolate(i, idx, values[0], values[1], *r_i))
                })?;
            if pi_0[idx % self.code.n_0()] != folded {
                return Err(Self::Error::InvalidQuery);
            }
            Ok(())
        })
    }
}

impl<F, E, R, M> Basefold<F, E, R, M>
where
    F: Field,
    E: ExtensionField<F>,
    R: RandomFoldableCode<F>,
    M: Mmcs<F, Commitment: Debug, ProverData<RowMajorMatrix<F>>: Debug, Proof: Debug> + Debug,
{
    #[allow(clippy::type_complexity)]
    fn prove_interleaved_folding_sumcheck(
        &self,
        pi: &mut Vec<E>,
        sumcheck_prover: &mut impl SumcheckFunctionProver<F, E>,
        sumcheck_subclaim: &mut SumcheckSubclaim<E>,
        mut challenger: impl FieldChallenger<F> + CanObserve<M::Commitment>,
    ) -> (
        Vec<(
            M::Commitment,
            M::ProverData<FlatMatrixView<F, E, RowMajorMatrix<E>>>,
        )>,
        Vec<Vec<E>>,
    ) {
        rev(0..self.code.d())
            .map(|i| {
                let pi_comm = self.mmcs_ext.commit_matrix(RowMajorMatrix::new(
                    zip!(&pi[..pi.len() / 2], &pi[pi.len() / 2..])
                        .flat_map(|(l, r)| [*l, *r])
                        .collect_vec(),
                    2,
                ));
                challenger.observe(pi_comm.0.clone());

                let compressed_round_poly = prove_sumcheck_round(
                    &mut *sumcheck_prover,
                    sumcheck_subclaim,
                    i + self.code.log2_k_0(),
                    &mut challenger,
                );
                let r_i = sumcheck_subclaim.r_rev().last().unwrap();
                self.code.fold(i, pi, *r_i);
                (pi_comm, compressed_round_poly)
            })
            .unzip::<_, _, Vec<_>, Vec<_>>()
    }

    fn verify_interleaved_folding_sumcheck(
        &self,
        sumcheck_func: &impl SumcheckFunction<F, E>,
        sumcheck_subclaim: &mut SumcheckSubclaim<E>,
        pi_comms: &[M::Commitment],
        sumcheck_proof: &SumcheckProof<E>,
        mut challenger: impl FieldChallenger<F> + CanObserve<M::Commitment>,
    ) -> Vec<E> {
        zip!(
            rev(0..self.code.d()),
            pi_comms,
            &sumcheck_proof.compressed_round_polys
        )
        .for_each(|(round, pi_comm, compressed_round_poly)| {
            challenger.observe(pi_comm.clone());

            verify_sumcheck_round(
                sumcheck_func,
                sumcheck_subclaim,
                round,
                compressed_round_poly,
                &mut challenger,
            )
        });
        sumcheck_subclaim.r().copied().collect()
    }

    fn sample_random_scalars_by_eq(
        &self,
        n: usize,
        mut challenger: impl FieldChallengerExt<F>,
    ) -> Vec<E> {
        let log_n = n.next_power_of_two().ilog2() as usize;
        let r = challenger.sample_ext_vec(log_n);
        let mut eq_r = MultiPoly::eq(&r, E::ONE).into_ext();
        eq_r.truncate(n);
        eq_r
    }
}

#[cfg(test)]
mod test {
    use crate::{
        basefold::{
            code::{
                BinaryReedSolomonCode, GenericRandomFoldableCode, RandomFoldableCode,
                ReedSolomonCode,
            },
            Basefold, BasefoldConfig,
        },
        test::run_multi_poly_commit_scheme,
    };
    use p3::{
        baby_bear::BabyBear,
        field::{extension::BinomialExtensionField, ExtensionField, Field, FromUniformBytes},
        matrix::dense::RowMajorMatrix,
        merkle_tree::{keccak256_merkle_tree, Keccak256MerkleTreeMmcs},
        polyval::Polyval,
    };
    use rand::rngs::StdRng;
    use util::Itertools;

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
        type F = BabyBear;
        type E = BinomialExtensionField<BabyBear, 5>;
        let lambda = 128;
        for (log2_c, log2_k_0) in (0..3).cartesian_product(0..3) {
            run_basefold::<F, E, _>(|num_vars, rng| {
                let log2_k_0 = log2_k_0.min(num_vars);
                let d = num_vars - log2_k_0;
                GenericRandomFoldableCode::reed_solomon_g_0_with_random_ts(
                    lambda, log2_c, log2_k_0, d, rng,
                )
            });
        }
    }

    #[test]
    fn reed_solomon_code() {
        type F = BabyBear;
        type E = BinomialExtensionField<BabyBear, 5>;
        let lambda = 128;
        for log2_c in 0..3 {
            run_basefold::<F, E, _>(|num_vars, _| ReedSolomonCode::new(lambda, log2_c, num_vars));
        }
    }

    #[test]
    fn binary_reed_solomon_code() {
        type F = Polyval;
        type E = Polyval;
        let lambda = 128;
        for log2_c in 0..3 {
            run_basefold::<F, E, _>(|num_vars, _| {
                BinaryReedSolomonCode::new(lambda, log2_c, num_vars)
            });
        }
    }
}
