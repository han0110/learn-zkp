use crate::{
    poly::{op_poly, op_polys, Poly},
    SumcheckFunction, SumcheckFunctionProver,
};
use core::{marker::PhantomData, ops::Deref};
use p3_field::{ExtensionField, Field};
use util::{collection::FieldArray, izip_eq};

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

#[derive(Clone, Debug)]
pub struct QuadraticProver<'a, F: Field, E: ExtensionField<F>> {
    f: Quadratic<F, E>,
    polys: Vec<Poly<'a, F, E>>,
}

impl<'a, F: Field, E: ExtensionField<F>> QuadraticProver<'a, F, E> {
    pub fn new(f: Quadratic<F, E>, polys: Vec<Poly<'a, F, E>>) -> Self {
        Self { f, polys }
    }
}

impl<'a, F: Field, E: ExtensionField<F>> Deref for QuadraticProver<'a, F, E> {
    type Target = Quadratic<F, E>;

    fn deref(&self) -> &Self::Target {
        &self.f
    }
}

impl<F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for Quadratic<F, E> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, evals: &[E]) -> E {
        self.pairs
            .iter()
            .map(|(scalar, f, g)| (*scalar * evals[*f] * evals[*g]))
            .sum()
    }

    fn compress_round_poly(&self, _: usize, round_poly: &[E]) -> Vec<E> {
        vec![round_poly[0], round_poly[2]]
    }

    fn decompress_round_poly(&self, _: usize, claim: E, compressed_round_poly: &[E]) -> Vec<E> {
        let &[coeff_0, coeff_2] = compressed_round_poly else {
            unreachable!()
        };
        let eval_1 = claim - coeff_0;
        let coeff_1 = eval_1 - coeff_0 - coeff_2;
        vec![coeff_0, coeff_1, coeff_2]
    }
}

impl<'a, F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for QuadraticProver<'a, F, E> {
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

impl<'a, F: Field, E: ExtensionField<F>> SumcheckFunctionProver<F, E>
    for QuadraticProver<'a, F, E>
{
    fn compute_sum(&self, _: usize) -> E {
        self.pairs
            .iter()
            .map(|(scalar, f, g)| {
                let (f, g) = (&self.polys[*f], &self.polys[*g]);
                let sum = op_polys!(|f, g| izip_eq!(f, g).map(|(f, g)| *f * *g).sum(), E::from);
                sum * *scalar
            })
            .sum()
    }

    fn compute_round_poly(&self, round: usize, claim: E) -> Vec<E> {
        let FieldArray([coeff_0, coeff_2]) = self
            .pairs
            .iter()
            .map(|(scalar, f, g)| {
                let (f, g) = (&self.polys[*f], &self.polys[*g]);
                let sum = op_polys!(
                    |f, g| (0..f.len())
                        .step_by(2)
                        .map(|b| {
                            let coeff_0 = f[b] * g[b];
                            let coeff_2 = (f[b + 1] - f[b]) * (g[b + 1] - g[b]);
                            FieldArray([coeff_0, coeff_2])
                        })
                        .sum::<FieldArray<_, 2>>(),
                    |sum| sum.map(E::from)
                );
                sum * *scalar
            })
            .sum();
        self.decompress_round_poly(round, claim, &[coeff_0, coeff_2])
    }

    fn fix_var(&mut self, x_i: &E) {
        self.polys.iter_mut().for_each(|poly| poly.fix_var(x_i));
    }

    fn evaluations(&self) -> Option<Vec<E>> {
        (self.polys[0].num_vars() == 0).then(|| {
            self.polys
                .iter()
                .map(|poly| op_poly!(|poly| poly[0], E::from))
                .collect()
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{
        function::quadratic::{Quadratic, QuadraticProver},
        poly::Poly,
        test::run_sumcheck,
    };
    use core::iter::repeat_with;
    use p3_field::extension::BinomialExtensionField;
    use p3_mersenne_31::Mersenne31;
    use rand::{distributions::Standard, Rng};
    use std::borrow::Cow;
    use util::field::FieldExt;

    #[test]
    fn quadratic() {
        type F = Mersenne31;
        type E = BinomialExtensionField<Mersenne31, 3>;
        run_sumcheck(|num_vars, rng| {
            let num_polys = rng.gen_range(5..10);
            let num_pairs = rng.gen_range(5..10);
            let f = Quadratic::new(
                num_vars,
                (0..num_pairs)
                    .map(|_| {
                        let scalar = rng.sample::<E, _>(Standard);
                        let f = rng.gen_range(0..num_polys);
                        let g = rng.gen_range(0..num_polys);
                        (scalar, f, g)
                    })
                    .collect(),
            );
            QuadraticProver::new(
                f,
                repeat_with(|| Poly::Base(Cow::Owned(F::random_vec(1 << num_vars, &mut *rng))))
                    .take(num_polys)
                    .collect(),
            )
        });
    }
}
