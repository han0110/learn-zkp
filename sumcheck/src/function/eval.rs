use crate::{
    function::{
        eval_imp::{EvalImpr, EvalImprProver},
        forward_impl_sumcheck_function,
    },
    SumcheckFunction, SumcheckFunctionProver, SumcheckSubclaim,
};
use p3::{
    field::{ExtensionField, Field},
    poly::multilinear::{eq_eval, MultiPoly},
};
use util::rev;

#[derive(Clone, Debug, derive_more::Deref)]
pub struct Eval<F, E> {
    inner: EvalImpr<F, E>,
}

impl<F: Field, E: ExtensionField<F>> Eval<F, E> {
    pub fn new(num_vars: usize, r: &[E]) -> Self {
        Self {
            inner: EvalImpr::new(num_vars, r),
        }
    }
}

#[derive(Clone, Debug, derive_more::Deref)]
pub struct EvalProver<'a, F: Field, E: ExtensionField<F>> {
    #[deref]
    f: Eval<F, E>,
    inner: EvalImprProver<'a, F, E>,
    inner_subclaim: SumcheckSubclaim<E>,
    inner_round_poly: [E; 2],
}

impl<'a, F: Field, E: ExtensionField<F>> EvalProver<'a, F, E> {
    pub fn new(f: Eval<F, E>, poly: MultiPoly<'a, F, E>) -> Self {
        Self {
            f: f.clone(),
            inner: EvalImprProver::new(f.inner, poly),
            inner_subclaim: SumcheckSubclaim::new(E::ZERO),
            inner_round_poly: [E::ZERO; 2],
        }
    }

    pub fn into_ext_poly(self) -> Vec<E> {
        self.inner.into_ext_poly()
    }
}

impl<F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for Eval<F, E> {
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    fn num_polys(&self) -> usize {
        self.inner.num_polys()
    }

    fn evaluate(&self, evals: &[E], r_rev: &[E]) -> E {
        self.inner.evaluate(evals, r_rev) * eq_eval(self.r(), rev(r_rev))
    }

    fn compress_round_poly(&self, _: usize, _: &SumcheckSubclaim<E>, round_poly: &[E]) -> Vec<E> {
        vec![round_poly[0], round_poly[2]]
    }

    fn decompress_round_poly(
        &self,
        _: usize,
        subclaim: &SumcheckSubclaim<E>,
        compressed_round_poly: &[E],
    ) -> Vec<E> {
        let &[coeff_0, coeff_2] = compressed_round_poly else {
            unreachable!()
        };
        let coeff_1 = **subclaim - coeff_0.double() - coeff_2;
        vec![coeff_0, coeff_1, coeff_2]
    }
}

forward_impl_sumcheck_function!(impl<'a, F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for EvalProver<'a, F, E>);

impl<'a, F: Field, E: ExtensionField<F>> SumcheckFunctionProver<F, E> for EvalProver<'a, F, E> {
    fn compute_sum(&self, round: usize) -> E {
        self.inner.compute_sum(round) * self.delta_scalar(round, &self.inner_subclaim)
    }

    fn compute_round_poly(&mut self, round: usize, subclaim: &SumcheckSubclaim<E>) -> Vec<E> {
        let subclaim = if round == self.num_vars() - 1 {
            subclaim
        } else {
            &self.inner_subclaim
        };
        let &[coeff_0, coeff_1] = &self.inner.compute_round_poly(round, subclaim)[..] else {
            unreachable!()
        };
        self.inner_round_poly = [coeff_0, coeff_1];

        let [delta_0, delta_1] = self.delta(round, subclaim);
        vec![
            coeff_0 * delta_0,
            coeff_0 * delta_1 + coeff_1 * delta_0,
            coeff_1 * delta_1,
        ]
    }

    fn fix_last_var(&mut self, x_i: E) {
        self.inner.fix_last_var(x_i);
        self.inner_subclaim.reduce(&self.inner_round_poly, x_i);
    }

    fn evaluations(&self) -> Option<Vec<E>> {
        self.inner.evaluations()
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
                let poly = MultiPoly::base(F::random_vec(1 << num_vars, &mut *rng));
                EvalProver::new(f, poly)
            });
        }

        run::<BabyBear, BabyBear>();
        run::<BabyBear, BinomialExtensionField<BabyBear, 5>>();
    }
}
