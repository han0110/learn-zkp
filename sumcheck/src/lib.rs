use core::fmt::Debug;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{ExtensionField, Field};
use util::{izip_eq, poly::univariate::horner};

pub mod function;
pub mod poly;

#[auto_impl::auto_impl(&, &mut)]
pub trait SumcheckFunction<F: Field, E: ExtensionField<F>>: Sized + Send + Sync + Debug {
    fn num_vars(&self) -> usize;

    fn evaluate(&self, evals: &[E]) -> E;

    fn compress_round_poly(&self, _round: usize, round_poly: &[E]) -> Vec<E> {
        round_poly.to_vec()
    }

    fn decompress_round_poly(
        &self,
        _round: usize,
        _claim: E,
        compressed_round_poly: &[E],
    ) -> Vec<E> {
        compressed_round_poly.to_vec()
    }
}

#[auto_impl::auto_impl(&mut)]
pub trait SumcheckFunctionProver<F: Field, E: ExtensionField<F>>: SumcheckFunction<F, E> {
    fn compute_sum(&self, round: usize) -> E;

    fn compute_round_poly(&self, round: usize, claim: E) -> Vec<E>;

    fn fix_var(&mut self, x_i: &E);

    fn evaluations(&self) -> Option<Vec<E>>;
}

#[derive(Clone, Debug, Default)]
pub struct SumcheckProof<E: Field> {
    compressed_round_polys: Vec<Vec<E>>,
}

pub fn prove_sumcheck<F: Field, E: ExtensionField<F>>(
    mut p: impl SumcheckFunctionProver<F, E>,
    claim: E,
    mut challenger: impl FieldChallenger<F> + CanObserve<E>,
) -> (SumcheckProof<E>, Vec<E>) {
    if p.num_vars() == 0 {
        return (SumcheckProof::default(), Vec::default());
    }

    challenger.observe(claim);

    let mut claim = claim;

    let (compressed_round_polys, r) = (0..p.num_vars())
        .map(|round| {
            #[cfg(feature = "sanity-check")]
            assert_eq!(p.compute_sum(round), claim);

            let round_poly = p.compute_round_poly(round, claim);

            let compressed_round_poly = p.compress_round_poly(round, &round_poly);

            challenger.observe_slice(&compressed_round_poly);

            let r_i = challenger.sample_ext_element();

            claim = horner(&round_poly, &r_i);

            p.fix_var(&r_i);

            (compressed_round_poly, r_i)
        })
        .unzip();

    (
        SumcheckProof {
            compressed_round_polys,
        },
        r,
    )
}

pub fn verify_sumcheck<F: Field, E: ExtensionField<F>>(
    f: impl SumcheckFunction<F, E>,
    claim: E,
    proof: &SumcheckProof<E>,
    mut challenger: impl FieldChallenger<F> + CanObserve<E>,
) -> (E, Vec<E>) {
    if f.num_vars() == 0 {
        return (claim, Vec::default());
    }

    challenger.observe(claim);

    let mut claim = claim;

    let r = izip_eq!(0..f.num_vars(), &proof.compressed_round_polys)
        .map(|(round, compressed_round_poly)| {
            challenger.observe_slice(compressed_round_poly);

            let round_poly = f.decompress_round_poly(round, claim, compressed_round_poly);

            let r_i = challenger.sample_ext_element();

            claim = horner(&round_poly, &r_i);

            r_i
        })
        .collect();

    (claim, r)
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{prove_sumcheck, verify_sumcheck, SumcheckFunctionProver};
    use p3_challenger::HashChallenger;
    use p3_field::ExtensionField;
    use p3_keccak::Keccak256Hash;
    use rand::{rngs::StdRng, SeedableRng};
    use util::{challenger::FieldExtChallenger, field::FieldExt};

    pub(crate) fn run_sumcheck<
        F: FieldExt,
        E: ExtensionField<F>,
        P: SumcheckFunctionProver<F, E>,
    >(
        f: impl Fn(usize, &mut StdRng) -> P,
    ) {
        let mut rng = StdRng::from_entropy();
        for num_vars in 0..10 {
            let mut p = f(num_vars, &mut rng);
            let claim = p.compute_sum(0);

            let (proof, _) = prove_sumcheck(
                &mut p,
                claim,
                FieldExtChallenger::new(HashChallenger::new(Vec::new(), Keccak256Hash)),
            );

            let (claim, _) = verify_sumcheck(
                &p,
                claim,
                &proof,
                FieldExtChallenger::new(HashChallenger::new(Vec::new(), Keccak256Hash)),
            );

            assert_eq!(claim, p.evaluate(&p.evaluations().unwrap()));
        }
    }
}
