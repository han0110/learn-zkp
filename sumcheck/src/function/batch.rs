use crate::{SumcheckFunction, SumcheckFunctionProver};
use core::{iter::zip, marker::PhantomData};
use p3::{
    field::{ExtensionField, Field},
    poly::univariate::horner,
};
use util::{chain, izip, Itertools};

#[derive(Debug)]
pub struct Batch<F: Field, E: ExtensionField<F>, T> {
    fs: Vec<T>,
    claims: Vec<E>,
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
        let round_polys = vec![None; fs.len()];
        Self {
            fs,
            claims,
            round_polys,
            alpha,
            _marker: PhantomData,
        }
    }

    pub fn claim(&self) -> E {
        izip!(&self.claims, self.alphas())
            .map(|(claim, alpha)| *claim * alpha)
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

    fn evaluate(&self, mut evals: &[E]) -> E {
        izip!(&self.fs, self.alphas())
            .map(|(f, alpha)| {
                let eval = f.evaluate(&evals[..f.num_polys()]);
                evals = &evals[f.num_polys()..];
                alpha * eval
            })
            .sum()
    }

    fn compress_round_poly(&self, _: usize, round_poly: &[E]) -> Vec<E> {
        chain![round_poly.first(), round_poly.iter().skip(2)]
            .copied()
            .collect()
    }

    fn decompress_round_poly(&self, _: usize, claim: E, compressed_round_poly: &[E]) -> Vec<E> {
        match compressed_round_poly {
            [] => vec![claim],
            &[coeff_0] => vec![coeff_0, claim - coeff_0.double()],
            &[coeff_0, ref rest @ ..] => {
                let coeff_1 = claim - coeff_0.double() - rest.iter().copied().sum::<E>();
                chain![[coeff_0, coeff_1], rest.iter().copied()].collect()
            }
        }
    }
}

impl<F: Field, E: ExtensionField<F>, T: SumcheckFunctionProver<F, E>> SumcheckFunctionProver<F, E>
    for Batch<F, E, T>
{
    fn compute_sum(&self, round: usize) -> E {
        izip!(&self.fs, &self.claims, self.alphas())
            .map(|(f, claim, alpha)| {
                let sum = if round < f.num_vars() {
                    f.compute_sum(round)
                } else {
                    *claim
                };
                alpha * sum
            })
            .sum()
    }

    fn compute_round_poly(&mut self, round: usize, _: E) -> Vec<E> {
        let round_polys = izip!(&mut self.fs, &mut self.claims, &mut self.round_polys)
            .map(|(f, claim, round_poly)| {
                if round < f.num_vars() {
                    *round_poly = Some(f.compute_round_poly(round, *claim));
                    round_poly.clone().unwrap()
                } else {
                    vec![claim.halve()]
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
        izip!(&mut self.fs, &mut self.claims, &mut self.round_polys).for_each(
            |(f, claim, round_poly)| {
                if let Some(round_poly) = round_poly {
                    f.fix_last_var(x_i);
                    *claim = horner(round_poly, x_i);
                } else {
                    *claim = claim.halve();
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
}
