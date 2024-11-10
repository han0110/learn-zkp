use crate::SumcheckSubclaim;
use core::fmt::Debug;
use p3::field::{ExtensionField, Field};

pub mod batch;
pub mod eval;
pub mod eval_impr;
pub mod quadratic;

#[auto_impl::auto_impl(&, &mut, Box)]
pub trait SumcheckFunction<F: Field, E: ExtensionField<F>>: Send + Sync + Debug {
    fn num_vars(&self) -> usize;

    fn num_polys(&self) -> usize;

    fn evaluate(&self, evals: &[E], r_rev: &[E]) -> E;

    fn compress_round_poly(
        &self,
        _round: usize,
        _subclaim: &SumcheckSubclaim<E>,
        round_poly: &[E],
    ) -> Vec<E> {
        round_poly.to_vec()
    }

    fn decompress_round_poly(
        &self,
        _round: usize,
        _subclaim: &SumcheckSubclaim<E>,
        compressed_round_poly: &[E],
    ) -> Vec<E> {
        compressed_round_poly.to_vec()
    }
}

#[auto_impl::auto_impl(&mut, Box)]
pub trait SumcheckFunctionProver<F: Field, E: ExtensionField<F>>: SumcheckFunction<F, E> {
    fn compute_sum(&self, round: usize) -> E;

    fn compute_round_poly(&mut self, round: usize, subclaim: &SumcheckSubclaim<E>) -> Vec<E>;

    fn fix_last_var(&mut self, x_i: E);

    fn evaluations(&self) -> Option<Vec<E>>;
}

macro_rules! forward_impl_sumcheck_function {
    (impl<$($($lf:lifetime),*,)? F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for $type:ty) => {
        impl<$($($lf),*,)? F: Field, E: ExtensionField<F>> SumcheckFunction<F, E> for $type {
            fn num_vars(&self) -> usize {
                core::ops::Deref::deref(self).num_vars()
            }

            fn num_polys(&self) -> usize {
                core::ops::Deref::deref(self).num_polys()
            }

            fn evaluate(&self, evals: &[E], r_rev: &[E]) -> E {
                core::ops::Deref::deref(self).evaluate(evals, r_rev)
            }

            fn compress_round_poly(
                &self,
                round: usize,
                subclaim: &SumcheckSubclaim<E>,
                round_poly: &[E],
            ) -> Vec<E> {
                core::ops::Deref::deref(self).compress_round_poly(round, subclaim, round_poly)
            }

            fn decompress_round_poly(
                &self,
                round: usize,
                subclaim: &SumcheckSubclaim<E>,
                compressed_round_poly: &[E],
            ) -> Vec<E> {
                core::ops::Deref::deref(self).decompress_round_poly(
                    round,
                    subclaim,
                    compressed_round_poly,
                )
            }
        }
    };
}

pub(crate) use forward_impl_sumcheck_function;
