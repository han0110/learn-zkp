use core::fmt::Debug;
use p3::field::{ExtensionField, Field};

pub mod eval;
pub mod quadratic;

#[auto_impl::auto_impl(&, &mut)]
pub trait SumcheckFunction<F: Field, E: ExtensionField<F>>: Sized + Send + Sync + Debug {
    fn num_vars(&self) -> usize;

    fn evaluate(&self, evals: &[E]) -> E;

    fn compress_round_poly(&self, _round: usize, round_poly: &[E]) -> Vec<E> {
        round_poly.to_vec()
    }

    fn decompress_round_poly(
        &self,
        _round: usize,
        _claim: E,
        compressed_round_poly: &[E],
    ) -> Vec<E> {
        compressed_round_poly.to_vec()
    }
}

#[auto_impl::auto_impl(&mut)]
pub trait SumcheckFunctionProver<F: Field, E: ExtensionField<F>>: SumcheckFunction<F, E> {
    fn compute_sum(&self, round: usize) -> E;

    fn compute_round_poly(&self, round: usize, claim: E) -> Vec<E>;

    fn fix_last_var(&mut self, x_i: E);

    fn evaluations(&self) -> Option<Vec<E>>;
}
