use crate::function::{SumcheckFunction, SumcheckFunctionProver};
use core::fmt::Debug;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_field::{ExtensionField, Field};
use util::{izip, poly::univariate::horner, Itertools};

pub mod function;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {}

#[derive(Clone, Debug, Default)]
pub struct SumcheckProof<E: Field> {
    compressed_round_polys: Vec<Vec<E>>,
}

pub fn prove_sumcheck<F: Field, E: ExtensionField<F>>(
    mut p: impl SumcheckFunctionProver<F, E>,
    claim: E,
    mut challenger: impl FieldChallenger<F> + CanObserve<E>,
) -> Result<(SumcheckProof<E>, Vec<E>)> {
    if p.num_vars() == 0 {
        return Ok((SumcheckProof::default(), Vec::default()));
    }

    challenger.observe(claim);

    let mut claim = claim;

    let (compressed_round_polys, r_rev) = (0..p.num_vars())
        .map(|round| {
            #[cfg(debug_assertions)]
            assert_eq!(p.compute_sum(round), claim);

            let round_poly = p.compute_round_poly(round, claim);

            let compressed_round_poly = p.compress_round_poly(round, &round_poly);

            challenger.observe_slice(&compressed_round_poly);

            let r_i = challenger.sample_ext_element();

            claim = horner(&round_poly, &r_i);

            p.fix_last_var(r_i);

            (compressed_round_poly, r_i)
        })
        .unzip::<_, _, Vec<_>, Vec<_>>();

    let r = r_rev.into_iter().rev().collect();

    Ok((
        SumcheckProof {
            compressed_round_polys,
        },
        r,
    ))
}

pub fn verify_sumcheck<F: Field, E: ExtensionField<F>>(
    f: impl SumcheckFunction<F, E>,
    claim: E,
    proof: &SumcheckProof<E>,
    mut challenger: impl FieldChallenger<F> + CanObserve<E>,
) -> Result<(E, Vec<E>)> {
    if f.num_vars() == 0 {
        return Ok((claim, Vec::default()));
    }

    challenger.observe(claim);

    let mut claim = claim;

    let r_rev = izip!(0..f.num_vars(), &proof.compressed_round_polys)
        .map(|(round, compressed_round_poly)| {
            challenger.observe_slice(compressed_round_poly);

            let round_poly = f.decompress_round_poly(round, claim, compressed_round_poly);

            let r_i = challenger.sample_ext_element();

            claim = horner(&round_poly, &r_i);

            r_i
        })
        .collect_vec();

    let r = r_rev.into_iter().rev().collect();

    Ok((claim, r))
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{prove_sumcheck, verify_sumcheck, SumcheckFunctionProver};
    use p3_field::{ExtensionField, Field};
    use rand::{rngs::StdRng, SeedableRng};
    use util::{challenger::GenericChallenger, field::FromUniformBytes};

    pub(crate) fn run_sumcheck<
        F: Field + FromUniformBytes,
        E: ExtensionField<F>,
        P: SumcheckFunctionProver<F, E>,
    >(
        f: impl Fn(usize, &mut StdRng) -> P,
    ) {
        let mut rng = StdRng::from_entropy();
        for num_vars in 0..10 {
            let mut p = f(num_vars, &mut rng);
            let claim = p.compute_sum(0);

            let (proof, _) = prove_sumcheck(&mut p, claim, GenericChallenger::keccak256()).unwrap();

            let (claim, _) =
                verify_sumcheck(&p, claim, &proof, GenericChallenger::keccak256()).unwrap();

            assert_eq!(claim, p.evaluate(&p.evaluations().unwrap()));
        }
    }
}
