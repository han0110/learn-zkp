use crate::{
    field::{Field, FieldSlice},
    matrix::{
        dense::{DenseMatrix, RowMajorMatrixViewMut},
        Matrix,
    },
};
use core::{iter::successors, marker::PhantomData, ops::Deref};
use util::{par_zip, rayon::prelude::*, rev, zip, Itertools};

pub trait BinaryField: Field {
    fn basis(i: usize) -> Self;
}

#[derive(Clone, Debug)]
pub struct AddtiveNtt<F, D = Vec<F>> {
    twiddles: Vec<D>,
    _marker: PhantomData<F>,
}

impl<F: BinaryField> AddtiveNtt<F, Vec<F>> {
    pub fn new(log_n: usize) -> Self {
        let mut betas = (0..log_n).map(F::basis);
        let twiddles_rev = successors(
            // (W_i(\beta_i), {W_i(\beta_{i+1}), \cdots, W_i(\beta_{log_n-1})})
            betas.next().zip(Some(betas.collect_vec())),
            |(w_beta_i, w_betas)| {
                w_betas.split_first().map(|(w_beta_first, w_beta_rest)| {
                    let q = |w_beta: &F| w_beta.square() + *w_beta * *w_beta_i;
                    (q(w_beta_first), w_beta_rest.iter().map(q).collect())
                })
            },
        )
        .map(|(w_beta_i, w_betas)| {
            let w_beta_i_inv = w_beta_i.inverse();
            let mut twiddles = vec![F::ZERO; 1 << w_betas.len()];
            zip!(0..w_betas.len(), w_betas).for_each(|(j, w_beta)| {
                let w_beta = w_beta * w_beta_i_inv;
                let (lo, hi) = twiddles[..2 << j].split_at_mut(1 << j);
                zip!(hi, lo).for_each(|(hi, lo)| *hi = *lo + w_beta);
            });
            twiddles
        })
        .collect_vec();
        let twiddles = rev(twiddles_rev).collect();

        Self {
            twiddles,
            _marker: PhantomData,
        }
    }

    pub fn coset(&self, coset_bits: usize, coset_idx: usize) -> AddtiveNtt<F, &[F]> {
        AddtiveNtt {
            twiddles: self.twiddles[coset_bits..]
                .iter()
                .map(|twiddles| {
                    twiddles
                        .chunks(twiddles.len() >> coset_bits)
                        .nth(coset_idx)
                        .unwrap()
                })
                .collect(),
            _marker: PhantomData,
        }
    }

    pub fn cosets(&self, coset_bits: usize) -> impl Iterator<Item = AddtiveNtt<F, &[F]>> {
        (0..1 << coset_bits).map(move |coset_idx| self.coset(coset_bits, coset_idx))
    }

    pub fn par_cosets(
        &self,
        coset_bits: usize,
    ) -> impl IndexedParallelIterator<Item = AddtiveNtt<F, &[F]>> {
        (0..1 << coset_bits)
            .into_par_iter()
            .map(move |coset_idx| self.coset(coset_bits, coset_idx))
    }
}

impl<F: BinaryField, D: Deref<Target = [F]>> AddtiveNtt<F, D> {
    pub fn twiddles(&self) -> &[D] {
        &self.twiddles
    }

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
        zip!(rev(0..log_n), &self.twiddles).for_each(|(i, twiddles)| {
            par_zip!(mat.par_row_chunks_exact_mut(2 << i), twiddles.deref())
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
        zip!(0..log_n, rev(&self.twiddles)).for_each(|(i, twiddles)| {
            par_zip!(mat.par_row_chunks_exact_mut(2 << i), twiddles.deref())
                .for_each(layer::<_, true>)
        })
    }
}

fn layer<F: Field, const INV: bool>((mut mat, twiddle): (RowMajorMatrixViewMut<F>, &F)) {
    let (mut lo, mut hi) = mat.split_rows_mut(mat.height() / 2);
    par_zip!(lo.par_rows_mut(), hi.par_rows_mut()).for_each(|(mut lo, mut hi)| {
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
        field::{dot_product, AddtiveNtt, BinaryField, FromUniformBytes},
        matrix::dense::RowMajorMatrix,
        polyval::Polyval,
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
                recurse(i + 1, s.iter().map(|s| q(i, *s)).dedup().collect(), q(i, x))
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
        for log_n in 0..10 {
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
