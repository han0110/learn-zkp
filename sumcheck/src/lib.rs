use crate::function::{SumcheckFunction, SumcheckFunctionProver};
use core::fmt::Debug;
use p3::{
    challenger::{FieldChallenger, FieldChallengerExt},
    field::{ExtensionField, Field},
    poly::univariate::horner,
};
use util::{rev, zip};

pub mod function;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Clone, Debug)]
pub enum Error {}

#[derive(Clone, Debug, Default, derive_more::Deref)]
pub struct SumcheckSubclaim<E> {
    #[deref]
    value: E,
    r_rev: Vec<E>,
}

impl<E> SumcheckSubclaim<E> {
    pub fn new(value: E) -> Self {
        Self {
            value,
            r_rev: Vec::new(),
        }
    }

    pub fn r_rev(&self) -> &[E] {
        &self.r_rev
    }

    pub fn r(&self) -> impl Iterator<Item = &E> {
        rev(&self.r_rev)
    }

    pub fn reduce(&mut self, round_poly: &[E], r_i: E)
    where
        E: Field,
    {
        self.value = horner(round_poly, r_i);
        self.r_rev.push(r_i);
    }
}

#[derive(Clone, Debug, Default)]
pub struct SumcheckProof<E: Field> {
    pub compressed_round_polys: Vec<Vec<E>>,
}

pub fn prove_sumcheck<F: Field, E: ExtensionField<F>>(
    mut p: impl SumcheckFunctionProver<F, E>,
    claim: E,
    mut challenger: impl FieldChallenger<F>,
) -> Result<(SumcheckSubclaim<E>, SumcheckProof<E>)> {
    let mut subclaim = SumcheckSubclaim::new(claim);

    if p.num_vars() == 0 {
        return Ok((subclaim, SumcheckProof::default()));
    }

    challenger.observe_ext_element(*subclaim);

    let compressed_round_polys = rev(0..p.num_vars())
        .map(|round| prove_sumcheck_round(&mut p, &mut subclaim, round, &mut challenger))
        .collect();

    Ok((
        subclaim,
        SumcheckProof {
            compressed_round_polys,
        },
    ))
}

pub fn prove_sumcheck_round<F: Field, E: ExtensionField<F>>(
    mut p: impl SumcheckFunctionProver<F, E>,
    subclaim: &mut SumcheckSubclaim<E>,
    round: usize,
    mut challenger: impl FieldChallenger<F>,
) -> Vec<E> {
    #[cfg(debug_assertions)]
    assert_eq!(p.compute_sum(round), **subclaim);

    let round_poly = p.compute_round_poly(round, subclaim);

    let compressed_round_poly = p.compress_round_poly(round, subclaim, &round_poly);

    challenger.observe_ext_slice(&compressed_round_poly);

    let r_i = challenger.sample_ext_element();

    p.fix_last_var(r_i);

    subclaim.reduce(&round_poly, r_i);

    compressed_round_poly
}

pub fn verify_sumcheck<F: Field, E: ExtensionField<F>>(
    f: impl SumcheckFunction<F, E>,
    claim: E,
    proof: &SumcheckProof<E>,
    mut challenger: impl FieldChallenger<F>,
) -> Result<SumcheckSubclaim<E>> {
    let mut subclaim = SumcheckSubclaim::new(claim);

    if f.num_vars() == 0 {
        return Ok(subclaim);
    }

    challenger.observe_ext_element(*subclaim);

    zip!(rev(0..f.num_vars()), &proof.compressed_round_polys).for_each(
        |(round, compressed_round_poly)| {
            verify_sumcheck_round(
                &f,
                &mut subclaim,
                round,
                compressed_round_poly,
                &mut challenger,
            )
        },
    );

    Ok(subclaim)
}

pub fn verify_sumcheck_round<F: Field, E: ExtensionField<F>>(
    f: impl SumcheckFunction<F, E>,
    subclaim: &mut SumcheckSubclaim<E>,
    round: usize,
    compressed_round_poly: &[E],
    mut challenger: impl FieldChallenger<F>,
) {
    challenger.observe_ext_slice(compressed_round_poly);

    let round_poly = f.decompress_round_poly(round, subclaim, compressed_round_poly);

    let r_i = challenger.sample_ext_element();

    subclaim.reduce(&round_poly, r_i);
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{prove_sumcheck, verify_sumcheck, SumcheckFunctionProver};
    use p3::{
        challenger::GenericChallenger,
        field::FromUniformBytes,
        field::{ExtensionField, Field},
    };
    use rand::{rngs::StdRng, SeedableRng};

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
            let claim = p.compute_sum(num_vars.saturating_sub(1));

            let (_, proof) = prove_sumcheck(&mut p, claim, GenericChallenger::keccak256()).unwrap();

            let subclaim =
                verify_sumcheck(&p, claim, &proof, GenericChallenger::keccak256()).unwrap();

            assert_eq!(
                *subclaim,
                p.evaluate(&p.evaluations().unwrap(), subclaim.r_rev())
            );
        }
    }
}
