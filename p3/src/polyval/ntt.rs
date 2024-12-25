use crate::{
    field::{BinaryField, Field, FieldSlice},
    matrix::{
        dense::{DenseMatrix, RowMajorMatrixViewMut},
        Matrix,
    },
};
use core::{
    iter::{successors, zip},
    marker::PhantomData,
    ops::Deref,
};
use util::{izip, rayon::prelude::*, rev, Itertools};

#[derive(Debug)]
pub struct AddtiveNtt<F, D = Vec<F>> {
    twiddles: Vec<D>,
    _marker: PhantomData<F>,
}

impl<F: BinaryField> AddtiveNtt<F, Vec<F>> {
    pub fn new(log_n: usize) -> Self {
        let twiddles = successors(
            (log_n > 0).then(|| ((1..log_n).map(F::basis).collect_vec(), F::ONE)),
            |(basis, w_beta)| {
                basis.split_first().map(|(basis_0, basis_rest)| {
                    let f = |basis: &F| basis.square() + *basis * *w_beta;
                    (basis_rest.iter().map(f).collect(), f(basis_0))
                })
            },
        )
        .map(|(basis, w_beta)| {
            let w_beta_inv = w_beta.inverse();
            let mut twiddles = vec![F::ZERO; 1 << basis.len()];
            izip!(0..basis.len(), basis).for_each(|(i, basis_i)| {
                let basis_hat_i = basis_i * w_beta_inv;
                let (lo, hi) = twiddles[..2 << i].split_at_mut(1 << i);
                izip!(hi, lo).for_each(|(hi, lo)| *hi = *lo + basis_hat_i);
            });
            twiddles
        })
        .collect_vec();

        Self {
            twiddles,
            _marker: PhantomData,
        }
    }

    pub fn coset(&self, coset_bits: usize, coset_idx: usize) -> AddtiveNtt<F, &[F]> {
        AddtiveNtt {
            twiddles: self
                .twiddles
                .iter()
                .map(|twiddles| {
                    twiddles
                        .chunks(twiddles.len() >> coset_bits)
                        .nth(coset_idx)
                        .unwrap()
                })
                .take(self.twiddles.len() - coset_bits)
                .collect(),
            _marker: PhantomData,
        }
    }
}

impl<F: BinaryField, D: core::fmt::Debug + Deref<Target = [F]>> AddtiveNtt<F, D> {
    pub fn forward(&self, mut data: Vec<F>) -> Vec<F> {
        self.forward_inplace(&mut data);
        data
    }

    pub fn forward_inplace(&self, data: &mut [F]) {
        self.forward_batch_inplace(DenseMatrix::new_col(data));
    }

    pub fn forward_batch_inplace(&self, mut mat: RowMajorMatrixViewMut<F>) {
        debug_assert_eq!(mat.height(), 1 << self.twiddles.len());
        let log_n = mat.height().ilog2() as usize;
        rev(zip(0..log_n, &self.twiddles)).for_each(|(i, twiddles)| {
            mat.par_row_chunks_exact_mut(2 << i)
                .zip(twiddles.deref())
                .for_each(layer::<_, false>)
        })
    }

    pub fn backward(&self, mut data: Vec<F>) -> Vec<F> {
        self.backward_inplace(&mut data);
        data
    }

    pub fn backward_inplace(&self, data: &mut [F]) {
        self.backward_batch_inplace(DenseMatrix::new_col(data));
    }

    pub fn backward_batch_inplace(&self, mut mat: RowMajorMatrixViewMut<F>) {
        debug_assert_eq!(mat.height(), 1 << self.twiddles.len());
        let log_n = mat.height().ilog2() as usize;
        zip(0..log_n, &self.twiddles).for_each(|(i, twiddles)| {
            mat.par_row_chunks_exact_mut(2 << i)
                .zip(twiddles.deref())
                .for_each(layer::<_, true>)
        })
    }
}

fn layer<F: Field, const INV: bool>((mut mat, twiddle): (RowMajorMatrixViewMut<F>, &F)) {
    let (mut lo, mut hi) = mat.split_rows_mut(mat.height() / 2);
    lo.par_rows_mut()
        .zip(hi.par_rows_mut())
        .for_each(|(mut lo, mut hi)| {
            if INV {
                hi.slice_add_assign(lo);
                lo.slice_add_scaled_assign(hi, *twiddle);
            } else {
                lo.slice_add_scaled_assign(hi, *twiddle);
                hi.slice_add_assign(lo);
            }
        });
}

#[cfg(test)]
mod test {
    use crate::{
        field::{dot_product, BinaryField, FromUniformBytes},
        matrix::dense::RowMajorMatrix,
        polyval::{ntt::AddtiveNtt, Polyval},
    };
    use rand::{rngs::StdRng, SeedableRng};
    use util::Itertools;

    fn forward_matrix<F: BinaryField>(log_n: usize) -> RowMajorMatrix<F> {
        fn u<F: BinaryField>(i: usize) -> impl Iterator<Item = F> {
            (0..1 << i).map(F::from_canonical_u64)
        }

        fn w_beta<F: BinaryField>(i: usize) -> F {
            u(i).map(|u| F::basis(i) - u).product()
        }

        fn q<F: BinaryField>(i: usize, x: F) -> F {
            (w_beta::<F>(i).square() / w_beta(i + 1)) * x * (x + F::ONE)
        }

        fn row<F: BinaryField>(log_n: usize, x: F) -> Vec<F> {
            return recurse(0, u(log_n).collect(), x);

            fn recurse<F: BinaryField>(i: usize, s: Vec<F>, x: F) -> Vec<F> {
                if s.len() == 1 {
                    return vec![F::ONE];
                }
                let s_next = s.iter().step_by(2).map(|s| q(i, *s)).collect();
                recurse(i + 1, s_next, q(i, x))
                    .into_iter()
                    .flat_map(|v| [v, v * x])
                    .collect()
            }
        }

        RowMajorMatrix::new(u(log_n).flat_map(|u| row(log_n, u)).collect(), 1 << log_n)
    }

    #[test]
    fn forward() {
        type F = Polyval;
        let mut rng = StdRng::from_entropy();
        for log_n in 0..4 {
            let mat = forward_matrix::<F>(log_n);
            let ntt = AddtiveNtt::new(log_n);
            let data = F::random_vec(1 << log_n, &mut rng);
            assert_eq!(
                mat.row_slices()
                    .map(|row| dot_product(row, &data))
                    .collect_vec(),
                ntt.forward(data)
            );
        }
    }

    #[test]
    fn round_trip() {
        type F = Polyval;
        let mut rng = StdRng::from_entropy();
        for (log_n, width) in (0..10).cartesian_product(1..10) {
            let a = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_n, width);
            let mut b = a.clone();
            let ntt = AddtiveNtt::new(log_n);
            ntt.forward_batch_inplace(b.as_view_mut());
            ntt.backward_batch_inplace(b.as_view_mut());
            assert_eq!(a, b);
        }
    }
}
