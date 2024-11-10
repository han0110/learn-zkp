use crate::{SumcheckFunction, SumcheckFunctionProver, SumcheckSubclaim};
use core::{iter::zip, marker::PhantomData};
use p3::field::{ExtensionField, Field};
use util::{chain, izip, Itertools};

#[derive(Debug)]
pub struct Batch<F: Field, E: ExtensionField<F>, T> {
    fs: Vec<T>,
    subclaims: Vec<SumcheckSubclaim<E>>,
    round_polys: Vec<Option<Vec<E>>>,
    alpha: E,
    _marker: PhantomData<F>,
}

impl<F: Field, E: ExtensionField<F>, T: SumcheckFunction<F, E>> Batch<F, E, T> {
    pub fn new(fs: Vec<T>, mut claims: Vec<E>, alpha: E) -> Self {
        let max_num_vars = fs.iter().map(|f| f.num_vars()).max().unwrap_or(0);
        izip!(&fs, &mut claims).for_each(|(f, claim)| {
            if f.num_vars() != max_num_vars {
                *claim *= E::TWO.powers().nth(max_num_vars - f.num_vars()).unwrap()
            }
        });
        let subclaims = claims.into_iter().map(SumcheckSubclaim::new).collect();
        let round_polys = vec![None; fs.len()];
        Self {
            fs,
            subclaims,
            round_polys,
            alpha,
            _marker: PhantomData,
        }
    }

    pub fn claim(&self) -> E {
        izip!(&self.subclaims, self.alphas())
            .map(|(subclaim, alpha)| **subclaim * alpha)
            .sum()
    }

    pub fn alphas(&self) -> impl Iterator<Item = E> {
        self.alpha.powers().take(self.fs.len())
    }
}

impl<F: Field, E: ExtensionField<F>, T: SumcheckFunction<F, E>> SumcheckFunction<F, E>
    for Batch<F, E, T>
{
    fn num_vars(&self) -> usize {
        self.fs.iter().map(|f| f.num_vars()).max().unwrap_or(0)
    }

    fn num_polys(&self) -> usize {
        self.fs.iter().map(|f| f.num_polys()).sum()
    }

    fn evaluate(&self, mut evals: &[E], r_rev: &[E]) -> E {
        izip!(&self.fs, self.alphas())
            .map(|(f, alpha)| {
                let eval = f.evaluate(
                    &evals[..f.num_polys()],
                    &r_rev[r_rev.len() - f.num_vars()..],
                );
                evals = &evals[f.num_polys()..];
                alpha * eval
            })
            .sum()
    }

    fn compress_round_poly(&self, _: usize, _: &SumcheckSubclaim<E>, round_poly: &[E]) -> Vec<E> {
        chain![round_poly.first(), round_poly.iter().skip(2)]
            .copied()
            .collect()
    }

    fn decompress_round_poly(
        &self,
        _: usize,
        subclaim: &SumcheckSubclaim<E>,
        compressed_round_poly: &[E],
    ) -> Vec<E> {
        match compressed_round_poly {
            [] => vec![**subclaim],
            &[coeff_0] => vec![coeff_0, **subclaim - coeff_0.double()],
            &[coeff_0, ref rest @ ..] => {
                let coeff_1 = **subclaim - coeff_0.double() - rest.iter().copied().sum::<E>();
                chain![[coeff_0, coeff_1], rest.iter().copied()].collect()
            }
        }
    }
}

impl<F: Field, E: ExtensionField<F>, T: SumcheckFunctionProver<F, E>> SumcheckFunctionProver<F, E>
    for Batch<F, E, T>
{
    fn compute_sum(&self, round: usize) -> E {
        izip!(&self.fs, &self.subclaims, self.alphas())
            .map(|(f, subclaim, alpha)| {
                let sum = if round < f.num_vars() {
                    f.compute_sum(round)
                } else {
                    **subclaim
                };
                alpha * sum
            })
            .sum()
    }

    fn compute_round_poly(&mut self, round: usize, _: &SumcheckSubclaim<E>) -> Vec<E> {
        let round_polys = izip!(&mut self.fs, &mut self.subclaims, &mut self.round_polys)
            .map(|(f, subclaim, round_poly)| {
                if round < f.num_vars() {
                    *round_poly = Some(f.compute_round_poly(round, subclaim));
                    round_poly.clone().unwrap()
                } else {
                    vec![subclaim.halve()]
                }
            })
            .collect_vec();
        izip!(round_polys, self.alphas())
            .map(|(round_poly, alpha)| {
                round_poly
                    .into_iter()
                    .map(|coeff| coeff * alpha)
                    .collect_vec()
            })
            .reduce(|mut a, b| {
                a.resize(a.len().max(b.len()), E::ZERO);
                zip(&mut a, b).for_each(|(a, b)| *a += b);
                a
            })
            .unwrap_or_default()
    }

    fn fix_last_var(&mut self, x_i: E) {
        izip!(&mut self.fs, &mut self.subclaims, &mut self.round_polys).for_each(
            |(f, subclaim, round_poly)| {
                if let Some(round_poly) = round_poly {
                    f.fix_last_var(x_i);
                    subclaim.reduce(round_poly, x_i);
                } else {
                    subclaim.value = subclaim.halve();
                }
            },
        );
    }

    fn evaluations(&self) -> Option<Vec<E>> {
        self.fs
            .iter()
            .map(|f| f.evaluations())
            .collect::<Option<Vec<_>>>()
            .map(|evaluations| evaluations.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        function::{
            batch::Batch,
            eval::{Eval, EvalProver},
            quadratic::{Quadratic, QuadraticProver},
            SumcheckFunction, SumcheckFunctionProver,
        },
        test::run_sumcheck,
    };
    use p3::{
        field::{
            extension::BinomialExtensionField, BabyBear, ExtensionField, Field, FromUniformBytes,
        },
        poly::multilinear::MultiPoly,
    };
    use util::Itertools;

    #[test]
    fn batch_quadratic() {
        fn run<F: Field + FromUniformBytes, E: ExtensionField<F> + FromUniformBytes>() {
            run_sumcheck(|num_vars, rng| {
                let fs = (0..3)
                    .map(|idx| {
                        let num_vars = num_vars.saturating_sub(idx);
                        let f = Quadratic::new(num_vars, vec![(E::random(&mut *rng), 0, 1)]);
                        let polys = (0..2)
                            .map(|_| MultiPoly::base(F::random_vec(1 << num_vars, &mut *rng)))
                            .collect();
                        QuadraticProver::new(f, polys)
                    })
                    .collect_vec();
                let claims = fs
                    .iter()
                    .map(|f| f.compute_sum(f.num_vars().saturating_sub(1)))
                    .collect();
                Batch::new(fs, claims, E::random(rng))
            });
        }

        run::<BabyBear, BabyBear>();
        run::<BabyBear, BinomialExtensionField<BabyBear, 5>>();
    }

    #[test]
    fn batch_eval() {
        fn run<F: Field + FromUniformBytes, E: ExtensionField<F> + FromUniformBytes>() {
            run_sumcheck(|num_vars, rng| {
                let fs = (0..3)
                    .map(|idx| {
                        let num_vars = num_vars.saturating_sub(idx);
                        let f = Eval::new(&E::random_vec(num_vars, &mut *rng));
                        let poly = MultiPoly::base(F::random_vec(1 << num_vars, &mut *rng));
                        EvalProver::new(f, poly)
                    })
                    .collect_vec();
                let claims = fs
                    .iter()
                    .map(|f| f.compute_sum(f.num_vars().saturating_sub(1)))
                    .collect();
                Batch::new(fs, claims, E::random(rng))
            });
        }

        run::<BabyBear, BabyBear>();
        run::<BabyBear, BinomialExtensionField<BabyBear, 5>>();
    }
}
