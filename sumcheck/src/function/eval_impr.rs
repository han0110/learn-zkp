use crate::{
    function::forward_impl_sumcheck_function, SumcheckFunction, SumcheckFunctionProver,
    SumcheckSubclaim,
};
use core::marker::PhantomData;
use p3::{
    field::{batch_multiplicative_inverse, ExtPackedValue, ExtensionField, Field},
    op_multi_polys,
    poly::multilinear::{eq_eval, evaluate, MultiPoly},
};
use util::{zip, Itertools};

#[derive(Clone, Debug)]
pub struct EvalImpr<F, E> {
    r: Vec<E>,
    r_inv: Vec<E>,
    _marker: PhantomData<F>,
}

impl<F: Field, E: ExtensionField<F>> EvalImpr<F, E> {
    pub fn new(r: &[E]) -> Self {
        Self {
            r: r.to_vec(),
            r_inv: batch_multiplicative_inverse(r),
            _marker: PhantomData,
        }
    }

    pub fn r(&self) -> &[E] {
        &self.r
    }

    pub fn r_i(&self, round: usize) -> E {
        self.r[round]
    }

    pub fn r_i_inv(&self, round: usize) -> E {
        self.r_inv[round]
    }

    pub fn eval_1(&self, round: usize, claim: E, eval_0: E) -> E {
        (claim - (E::ONE - self.r_i(round)) * eval_0) * self.r_i_inv(round)
    }

    pub fn delta_scalar(&self, round: usize, subclaim: &SumcheckSubclaim<E>) -> E {
        if round == self.num_vars().saturating_sub(1) {
            E::ONE
        } else {
            eq_eval(&self.r()[round + 1..], subclaim.r())
        }
    }

    pub fn delta(&self, round: usize, subclaim: &SumcheckSubclaim<E>) -> [E; 2] {
        let scalar = self.delta_scalar(round, subclaim);
        let coeff_0 = E::ONE - self.r_i(round);
        let coeff_1 = self.r_i(round) - coeff_0;
        [scalar * coeff_0, scalar * coeff_1]
    }
}

#[derive(Clone, Debug, derive_more::Deref, derive_more::DerefMut)]
pub struct HalfEq<F: Field, E: ExtensionField<F>> {
    one_minus_r_inv: Vec<E>,
    #[deref]
    #[deref_mut]
    poly: MultiPoly<'static, F, E>,
}

impl<F: Field, E: ExtensionField<F>> HalfEq<F, E> {
    pub fn new(r: &[E]) -> Self {
        let r = &r[..r.len().saturating_sub(1)];
        let one_minus_r_inv =
            batch_multiplicative_inverse(&r.iter().map(|r_i| E::ONE - *r_i).collect_vec());
        let poly = MultiPoly::eq(r, E::ONE);
        Self {
            one_minus_r_inv,
            poly,
        }
    }

    /// Returns `inv((1 - r_{round}) * (1 - r_{round+1}) * ... * (1 - r_{d-2}))`.
    pub fn correcting_factor(&self, round: usize) -> E {
        self.one_minus_r_inv
            .iter()
            .skip(round)
            .copied()
            .product::<E>()
    }

    pub fn halve(&mut self) {
        match &mut self.poly {
            MultiPoly::Base(evals) => {
                let mid = evals.len() / 2;
                evals.to_mut().truncate(mid)
            }
            MultiPoly::Ext(evals) => {
                let mid = evals.len() / 2;
                evals.to_mut().truncate(mid)
            }
            MultiPoly::ExtPacking(evals) => {
                if evals.len() == 1 {
                    let mut evals = E::ExtensionPacking::ext_unpack_slice(evals);
                    evals.truncate(evals.len() / 2);
                    self.poly = MultiPoly::ext(evals);
                } else {
                    let mid = evals.len() / 2;
                    evals.to_mut().truncate(mid)
                }
            }
        }
    }
}

#[derive(Clone, Debug, derive_more::Deref)]
pub struct EvalImprProver<'a, F: Field, E: ExtensionField<F>> {
    #[deref]
    f: EvalImpr<F, E>,
    half_eq: HalfEq<F, E>,
    pub poly: MultiPoly<'a, F, E>,
}

impl<'a, F: Field, E: ExtensionField<F>> EvalImprProver<'a, F, E> {
    pub fn new(f: EvalImpr<F, E>, poly: MultiPoly<'a, F, E>) -> Self {
        let half_eq = HalfEq::new(&f.r);
        Self { f, half_eq, poly }
    }

    pub fn into_ext_poly(self) -> Vec<E> {
        self.poly.into_ext()
    }
}

impl<F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for EvalImpr<F, E> {
    fn num_vars(&self) -> usize {
        self.r.len()
    }

    fn num_polys(&self) -> usize {
        1
    }

    fn evaluate(&self, evals: &[E], _: &[E]) -> E {
        evals[0]
    }

    fn compress_round_poly(&self, _: usize, _: &SumcheckSubclaim<E>, round_poly: &[E]) -> Vec<E> {
        vec![round_poly[0]]
    }

    fn decompress_round_poly(
        &self,
        round: usize,
        subclaim: &SumcheckSubclaim<E>,
        compressed_round_poly: &[E],
    ) -> Vec<E> {
        let &[coeff_0] = compressed_round_poly else {
            unreachable!()
        };
        let coeff_1 = self.eval_1(round, **subclaim, coeff_0) - coeff_0;
        vec![coeff_0, coeff_1]
    }
}

forward_impl_sumcheck_function!(impl<'a, F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for EvalImprProver<'a, F, E>);

impl<F: Field, E: ExtensionField<F>> SumcheckFunctionProver<F, E> for EvalImprProver<'_, F, E> {
    fn compute_sum(&self, round: usize) -> E {
        if self.f.num_vars() == 0 {
            return self.poly.to_ext()[0];
        }

        let half_eq = &self.half_eq;
        let (poly_lo, poly_hi) = &self.poly.split_at(self.poly.len() / 2);
        half_eq.correcting_factor(round)
            * evaluate(
                &[poly_lo, poly_hi].map(|poly| {
                    op_multi_polys!(
                        |half_eq, poly| zip!(half_eq, poly).map(|(f, g)| *f * *g).sum(),
                        |sum| E::from(sum),
                        |sum: E::ExtensionPacking| sum.ext_sum(),
                    )
                }),
                &[self.r_i(round)],
            )
    }

    fn compute_round_poly(&mut self, round: usize, subclaim: &SumcheckSubclaim<E>) -> Vec<E> {
        let half_eq = &self.half_eq;
        let (poly_lo, _) = &self.poly.split_at(self.poly.len() / 2);
        let coeff_0 = half_eq.correcting_factor(round)
            * op_multi_polys!(
                |half_eq, poly_lo| zip!(half_eq, poly_lo).map(|(f, g)| *f * *g).sum(),
                |sum| E::from(sum),
                |sum: E::ExtensionPacking| sum.ext_sum(),
            );
        self.decompress_round_poly(round, subclaim, &[coeff_0])
    }

    fn fix_last_var(&mut self, x_i: E) {
        self.poly.fix_last_var(x_i);
        self.half_eq.halve();
    }

    fn evaluations(&self) -> Option<Vec<E>> {
        (self.poly.num_vars() == 0).then(|| vec![self.poly.to_ext()[0]])
    }
}

#[cfg(test)]
mod test {
    use crate::{
        function::eval_impr::{EvalImpr, EvalImprProver},
        test::run_sumcheck,
    };
    use p3::{
        baby_bear::BabyBear,
        field::{extension::BinomialExtensionField, ExtensionField, Field, FromUniformBytes},
        poly::multilinear::MultiPoly,
    };

    #[test]
    fn eval_imp() {
        fn run<F: Field + FromUniformBytes, E: ExtensionField<F> + FromUniformBytes>() {
            run_sumcheck(|num_vars, rng| {
                let f = EvalImpr::new(&E::random_vec(num_vars, &mut *rng));
                let poly = MultiPoly::base(F::random_vec(1 << num_vars, &mut *rng));
                EvalImprProver::new(f, poly)
            });
        }

        run::<BabyBear, BabyBear>();
        run::<BabyBear, BinomialExtensionField<BabyBear, 5>>();
    }
}
