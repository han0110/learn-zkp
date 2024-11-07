use crate::{SumcheckFunction, SumcheckFunctionProver};
use core::{marker::PhantomData, ops::Deref};
use p3::{
    field::{batch_multiplicative_inverse, ExtPackedValue, ExtensionField, Field},
    op_multi_polys,
    poly::multilinear::{evaluate, MultiPoly},
};
use util::{izip, Itertools};

#[derive(Clone, Debug)]
pub struct Eval<F, E> {
    num_vars: usize,
    r: Vec<E>,
    r_inv: Vec<E>,
    _marker: PhantomData<F>,
}

impl<F: Field, E: ExtensionField<F>> Eval<F, E> {
    pub fn new(num_vars: usize, r: &[E]) -> Self {
        Self {
            num_vars,
            r: r.to_vec(),
            r_inv: batch_multiplicative_inverse(r),
            _marker: PhantomData,
        }
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
pub struct EvalProver<'a, F: Field, E: ExtensionField<F>> {
    #[deref]
    f: Eval<F, E>,
    half_eq: HalfEq<F, E>,
    poly: MultiPoly<'a, F, E>,
}

impl<'a, F: Field, E: ExtensionField<F>> EvalProver<'a, F, E> {
    pub fn new(f: Eval<F, E>, poly: MultiPoly<'a, F, E>) -> Self {
        let half_eq = HalfEq::new(&f.r);
        Self { f, half_eq, poly }
    }

    pub fn into_ext_poly(self) -> Vec<E> {
        self.poly.into_ext()
    }
}

impl<F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for Eval<F, E> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, evals: &[E]) -> E {
        evals[0]
    }

    fn compress_round_poly(&self, _: usize, round_poly: &[E]) -> Vec<E> {
        vec![round_poly[0]]
    }

    fn decompress_round_poly(&self, round: usize, claim: E, compressed_round_poly: &[E]) -> Vec<E> {
        let &[coeff_0] = compressed_round_poly else {
            unreachable!()
        };
        let coeff_1 = self.eval_1(round, claim, coeff_0) - coeff_0;
        vec![coeff_0, coeff_1]
    }
}

impl<'a, F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for EvalProver<'a, F, E> {
    fn num_vars(&self) -> usize {
        self.deref().num_vars()
    }

    fn evaluate(&self, evals: &[E]) -> E {
        self.deref().evaluate(evals)
    }

    fn compress_round_poly(&self, round: usize, round_poly: &[E]) -> Vec<E> {
        self.deref().compress_round_poly(round, round_poly)
    }

    fn decompress_round_poly(&self, round: usize, claim: E, compressed_round_poly: &[E]) -> Vec<E> {
        self.deref()
            .decompress_round_poly(round, claim, compressed_round_poly)
    }
}

impl<'a, F: Field, E: ExtensionField<F>> SumcheckFunctionProver<F, E> for EvalProver<'a, F, E> {
    fn compute_sum(&self, round: usize) -> E {
        if self.f.num_vars == 0 {
            return self.poly.to_ext()[0];
        }

        let half_eq = &self.half_eq;
        let (poly_lo, poly_hi) = &self.poly.split_at(self.poly.len() / 2);
        half_eq.correcting_factor(round)
            * evaluate(
                &[poly_lo, poly_hi].map(|poly| {
                    op_multi_polys!(
                        |half_eq, poly| izip!(half_eq, poly).map(|(f, g)| *f * *g).sum(),
                        |sum| E::from(sum),
                        |sum: E::ExtensionPacking| sum.ext_sum(),
                    )
                }),
                &[self.r_i(round)],
            )
    }

    fn compute_round_poly(&self, round: usize, claim: E) -> Vec<E> {
        let half_eq = &self.half_eq;
        let (poly_lo, _) = &self.poly.split_at(self.poly.len() / 2);
        let coeff_0 = half_eq.correcting_factor(round)
            * op_multi_polys!(
                |half_eq, poly_lo| izip!(half_eq, poly_lo).map(|(f, g)| *f * *g).sum(),
                |sum| E::from(sum),
                |sum: E::ExtensionPacking| sum.ext_sum(),
            );
        self.decompress_round_poly(round, claim, &[coeff_0])
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
        function::eval::{Eval, EvalProver},
        test::run_sumcheck,
    };
    use p3::{
        field::{
            extension::BinomialExtensionField, BabyBear, ExtensionField, Field, FromUniformBytes,
        },
        poly::multilinear::MultiPoly,
    };

    #[test]
    fn eval() {
        fn run<F: Field + FromUniformBytes, E: ExtensionField<F> + FromUniformBytes>() {
            run_sumcheck(|num_vars, rng| {
                let f = Eval::new(num_vars, &E::random_vec(num_vars, &mut *rng));
                EvalProver::new(f, MultiPoly::base(F::random_vec(1 << num_vars, &mut *rng)))
            });
        }

        run::<BabyBear, BabyBear>();
        run::<BabyBear, BinomialExtensionField<BabyBear, 5>>();
    }
}
