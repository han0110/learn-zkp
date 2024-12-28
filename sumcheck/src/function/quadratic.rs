use crate::{
    function::forward_impl_sumcheck_function, SumcheckFunction, SumcheckFunctionProver,
    SumcheckSubclaim,
};
use core::marker::PhantomData;
use p3::{
    field::{ExtPackedValue, ExtensionField, Field, FieldArray},
    op_multi_polys,
    poly::multilinear::MultiPoly,
};
use util::zip;

#[derive(Clone, Debug)]
pub struct Quadratic<F, E> {
    num_vars: usize,
    pairs: Vec<(E, usize, usize)>,
    _marker: PhantomData<F>,
}

impl<F, E> Quadratic<F, E> {
    pub fn new(num_vars: usize, pairs: Vec<(E, usize, usize)>) -> Self {
        debug_assert!(!pairs.is_empty());
        Self {
            num_vars,
            pairs,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug, derive_more::Deref)]
pub struct QuadraticProver<'a, F: Field, E: ExtensionField<F>> {
    #[deref]
    f: Quadratic<F, E>,
    polys: Vec<MultiPoly<'a, F, E>>,
}

impl<'a, F: Field, E: ExtensionField<F>> QuadraticProver<'a, F, E> {
    pub fn new(f: Quadratic<F, E>, polys: Vec<MultiPoly<'a, F, E>>) -> Self {
        Self { f, polys }
    }

    pub fn into_ext_polys(self) -> Vec<Vec<E>> {
        self.polys.into_iter().map(MultiPoly::into_ext).collect()
    }
}

impl<F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for Quadratic<F, E> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn num_polys(&self) -> usize {
        self.pairs.iter().map(|(_, f, g)| f.max(g)).max().unwrap() + 1
    }

    fn evaluate(&self, evals: &[E], _: &[E]) -> E {
        self.pairs
            .iter()
            .map(|(scalar, f, g)| (*scalar * evals[*f] * evals[*g]))
            .sum()
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

forward_impl_sumcheck_function!(impl<'a, F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for QuadraticProver<'a, F, E>);

impl<F: Field, E: ExtensionField<F>> SumcheckFunctionProver<F, E> for QuadraticProver<'_, F, E> {
    fn compute_sum(&self, _: usize) -> E {
        self.pairs
            .iter()
            .map(|(scalar, f, g)| {
                let (f, g) = (&self.polys[*f], &self.polys[*g]);
                let sum = op_multi_polys!(
                    |f, g| zip!(f, g).map(|(f, g)| *f * *g).sum(),
                    |sum| E::from(sum),
                    |sum: E::ExtensionPacking| sum.ext_sum(),
                );
                sum * *scalar
            })
            .sum()
    }

    fn compute_round_poly(&mut self, round: usize, subclaim: &SumcheckSubclaim<E>) -> Vec<E> {
        let FieldArray([coeff_0, coeff_2]) = self
            .pairs
            .iter()
            .map(|(scalar, f, g)| {
                let (f, g) = (&self.polys[*f], &self.polys[*g]);
                let sum = op_multi_polys!(
                    |f, g| (0..f.len() / 2)
                        .map(|b| {
                            let coeff_0 = f[b] * g[b];
                            let coeff_2 = (f[f.len() / 2 + b] - f[b]) * (g[f.len() / 2 + b] - g[b]);
                            FieldArray([coeff_0, coeff_2])
                        })
                        .sum::<FieldArray<_, 2>>(),
                    |sum| sum.map(E::from),
                    |sum| sum.map(|packed| packed.ext_sum()),
                );
                sum * *scalar
            })
            .sum();
        self.decompress_round_poly(round, subclaim, &[coeff_0, coeff_2])
    }

    fn fix_last_var(&mut self, x_i: E) {
        self.polys
            .iter_mut()
            .for_each(|poly| poly.fix_last_var(x_i));
    }

    fn evaluations(&self) -> Option<Vec<E>> {
        (self.polys[0].num_vars() == 0)
            .then(|| self.polys.iter().map(|poly| poly.to_ext()[0]).collect())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        function::quadratic::{Quadratic, QuadraticProver},
        test::run_sumcheck,
    };
    use core::iter::repeat_with;
    use p3::{
        baby_bear::BabyBear,
        field::{extension::BinomialExtensionField, ExtensionField, Field, FromUniformBytes},
        poly::multilinear::MultiPoly,
    };
    use rand::Rng;

    #[test]
    fn quadratic() {
        fn run<F: Field + FromUniformBytes, E: ExtensionField<F> + FromUniformBytes>() {
            run_sumcheck(|num_vars, rng| {
                let num_polys = rng.gen_range(5..10);
                let num_pairs = rng.gen_range(5..10);
                let f = Quadratic::new(
                    num_vars,
                    (0..num_pairs)
                        .map(|_| {
                            let scalar = E::random(&mut *rng);
                            let f = rng.gen_range(0..num_polys);
                            let g = rng.gen_range(0..num_polys);
                            (scalar, f, g)
                        })
                        .collect(),
                );
                QuadraticProver::new(
                    f,
                    repeat_with(|| MultiPoly::base(F::random_vec(1 << num_vars, &mut *rng)))
                        .take(num_polys)
                        .collect(),
                )
            });
        }

        run::<BabyBear, BabyBear>();
        run::<BabyBear, BinomialExtensionField<BabyBear, 5>>();
    }
}
