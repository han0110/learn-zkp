use crate::{
    enumerate,
    field::{ExtPackedValue, FieldSlice},
    izip, Itertools,
};
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, Field, PackedValue};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use std::borrow::Cow;

pub fn interpolate<F: Field>(mut evals: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
    debug_assert!(evals.height().is_power_of_two());
    for i in 0..evals.height().ilog2() {
        let chunk_size = 1 << (i + 1);
        evals.par_row_chunks_mut(chunk_size).for_each(|mut evals| {
            let (lo, mut hi) = evals.split_rows_mut(chunk_size >> 1);
            hi.values.slice_sub_assign(lo.values);
        })
    }
    evals
}

pub fn transpose<F: Field>(evals: RowMajorMatrix<F>) -> Vec<Vec<F>> {
    (0..evals.width())
        .map(|offset| {
            evals.values[offset..]
                .iter()
                .step_by(evals.height())
                .copied()
                .collect()
        })
        .collect()
}

#[derive(Clone, Debug)]
pub enum MultiPoly<'a, F: Field, E: ExtensionField<F>> {
    Base(Cow<'a, [F]>),
    Ext(Vec<E>),
    ExtPacking(Vec<E::ExtensionPacking>),
}

impl<'a, F: Field, E: ExtensionField<F>> MultiPoly<'a, F, E> {
    const __: () = assert!(F::Packing::WIDTH.is_power_of_two());

    pub fn base(evals: impl Into<Cow<'a, [F]>>) -> Self {
        Self::Base(evals.into())
    }

    pub fn eq(x: &[E], scalar: E) -> Self {
        if E::D == 1 || F::Packing::WIDTH == 1 || (1 << x.len()) <= F::Packing::WIDTH {
            let mut evals = E::zero_vec(1 << x.len());
            evals[0] = scalar;
            enumerate(x).for_each(|(i, x_i)| eq_expand(&mut evals, *x_i, i));
            return Self::Ext(evals);
        }

        let mut lo = E::zero_vec(F::Packing::WIDTH);
        lo[0] = scalar;
        let (x_lo, x_hi) = x.split_at(F::Packing::WIDTH.ilog2() as usize);
        enumerate(x_lo).for_each(|(i, x_i)| eq_expand(&mut lo, *x_i, i));

        let mut evals = E::ExtensionPacking::zero_vec((1 << x.len()) / F::Packing::WIDTH);
        evals[0] = E::ExtensionPacking::ext_pack(&lo);
        let x_hi = x_hi.iter().copied().map(E::ExtensionPacking::ext_broadcast);
        enumerate(x_hi).for_each(|(i, x_i)| eq_expand(&mut evals, x_i, i));
        Self::ExtPacking(evals)
    }

    pub fn num_vars(&self) -> usize {
        match self {
            Self::Base(evals) => evals.len().ilog2() as usize,
            Self::Ext(evals) => evals.len().ilog2() as usize,
            Self::ExtPacking(evals) => (E::Packing::WIDTH * evals.len()).ilog2() as usize,
        }
    }

    pub fn fix_last_var(&mut self, x_i: E) {
        match self {
            Self::Base(evals) => *self = Self::fix_last_var_from_base(evals, x_i),
            Self::Ext(evals) => fix_last_var(evals, x_i),
            Self::ExtPacking(evals) => {
                if evals.len() == 1 {
                    let mut evals = evals[0].ext_unpack();
                    fix_last_var(&mut evals, x_i);
                    *self = Self::Ext(evals);
                } else {
                    let x_i = E::ExtensionPacking::ext_broadcast(x_i);
                    fix_last_var(evals, x_i);
                }
            }
        }
    }

    fn fix_last_var_from_base(evals: &[F], x_i: E) -> Self {
        if F::D > 1 && F::Packing::WIDTH > 1 && evals.len() > F::Packing::WIDTH {
            let evals = F::Packing::pack_slice(evals);
            let x_i = E::ExtensionPacking::ext_broadcast(x_i);
            Self::ExtPacking(fix_last_var_from_base(evals, x_i))
        } else {
            Self::Ext(fix_last_var_from_base(evals, x_i))
        }
    }

    pub fn evaluate(&self, x: &[E]) -> E {
        fn recurse<F: AbstractField + Copy>(evals: &[F], x: &[F]) -> F {
            debug_assert_eq!(evals.len(), 1 << x.len());
            match x {
                [] => evals[0],
                &[ref x @ .., x_i] => {
                    let (lo, hi) = evals.split_at(evals.len() / 2);
                    let (lo, hi) = (recurse(lo, x), recurse(hi, x));
                    x_i * (hi - lo) + lo
                }
            }
        }
        match self {
            Self::Base(evals) => x
                .split_last()
                .map(|(x_last, x)| Self::fix_last_var_from_base(evals, *x_last).evaluate(x))
                .unwrap_or_else(|| E::from_base(evals[0])),
            Self::Ext(evals) => recurse(evals, x),
            Self::ExtPacking(evals) => {
                let (x_lo, x_hi) = x.split_at(F::Packing::WIDTH.ilog2() as usize);
                let x_hi = x_hi
                    .iter()
                    .copied()
                    .map(E::ExtensionPacking::ext_broadcast)
                    .collect_vec();
                recurse(&recurse(evals, &x_hi).ext_unpack(), x_lo)
            }
        }
    }

    pub fn op<T>(
        a: &Self,
        b: &Self,
        bb: &impl Fn(&[F], &[F]) -> T,
        ee: &impl Fn(&[E], &[E]) -> T,
        pp: &impl Fn(&[E::ExtensionPacking], &[E::ExtensionPacking]) -> T,
        eb: &impl Fn(&[E], &[F]) -> T,
        pb: &impl Fn(&[E::ExtensionPacking], &[F::Packing]) -> T,
    ) -> T {
        use MultiPoly::*;
        match (a, b) {
            (Base(a), Base(b)) => bb(a, b),
            (Ext(a), Ext(b)) => ee(a, b),
            (ExtPacking(a), ExtPacking(b)) => pp(a, b),
            (Base(b), Ext(a)) | (Ext(a), Base(b)) => eb(a, b),
            (Base(b), ExtPacking(a)) | (ExtPacking(a), Base(b)) => pb(a, F::Packing::pack_slice(b)),
            (Ext(_), ExtPacking(_)) | (ExtPacking(_), Ext(_)) => unimplemented!(),
        }
    }

    pub fn to_ext(&self) -> Vec<E> {
        match self {
            Self::Base(evals) => evals.iter().copied().map(E::from_base).collect(),
            Self::Ext(evals) => evals.clone(),
            Self::ExtPacking(evals) => E::ExtensionPacking::ext_unpack_slice(evals),
        }
    }

    pub fn into_ext(self) -> Vec<E> {
        match self {
            Self::Base(evals) => evals.iter().copied().map(E::from_base).collect(),
            Self::Ext(evals) => evals,
            Self::ExtPacking(evals) => E::ExtensionPacking::ext_unpack_slice(&evals),
        }
    }
}

fn fix_last_var<F: AbstractField + Copy>(evals: &mut Vec<F>, x_i: F) {
    let mid = evals.len() / 2;
    let (lo, hi) = evals.split_at_mut(mid);
    izip!(lo, hi).for_each(|(lo, hi)| *lo += x_i * (*hi - *lo));
    evals.truncate(mid);
}

fn fix_last_var_from_base<F: AbstractField + Copy, E: AbstractExtensionField<F> + Copy>(
    evals: &[F],
    x_i: E,
) -> Vec<E> {
    let (lo, hi) = evals.split_at(evals.len() / 2);
    izip!(lo, hi)
        .map(|(lo, hi)| x_i * (*hi - *lo) + *lo)
        .collect()
}

fn eq_expand<F: AbstractField>(evals: &mut [F], x_i: F, i: usize) {
    let (lo, hi) = evals[..1 << (i + 1)].split_at_mut(1 << i);
    izip!(lo, hi).for_each(|(lo, hi)| {
        *hi = lo.clone() * x_i.clone();
        *lo -= hi.clone();
    });
}

pub fn eq_eval<E: Field>(x: &[E], y: &[E]) -> E {
    E::product(izip!(x, y).map(|(&x, &y)| (x * y).double() + E::ONE - x - y))
}

#[macro_export]
macro_rules! op_multi_polys {
    (|$a:ident, $b:ident| $op:expr, |$b_out:ident| $op_b_out:expr, |$p_out:ident| $op_p_out:expr $(,)?) => {
        $crate::poly::multilinear::MultiPoly::op(
            $a,
            $b,
            &|a, b| {
                let $a = a;
                let $b = b;
                let $b_out = $op;
                $op_b_out
            },
            &|a, b| {
                let $a = a;
                let $b = b;
                $op
            },
            &|a, b| {
                let $a = a;
                let $b = b;
                let $p_out = $op;
                $op_p_out
            },
            &|a, b| {
                let $a = a;
                let $b = b;
                $op
            },
            &|a, b| {
                let $a = a;
                let $b = b;
                let $p_out = $op;
                $op_p_out
            },
        )
    };
    (|$a:ident, $b:ident| $op:expr, $op_b_out:expr, $op_p_out:expr $(,)?) => {
        op_multi_polys!(|$a, $b| $op, |out| $op_b_out(out), |out| $op_p_out(out))
    };
    (|$a:ident, $b:ident| $op:expr $(,)?) => {
        op_multi_polys!(|$a, $b| $op, |out| out, |out| out)
    };
}

#[cfg(test)]
mod test {
    use crate::{field::FromUniformBytes, poly::multilinear::MultiPoly};
    use p3_baby_bear::BabyBear;
    use p3_field::{extension::BinomialExtensionField, ExtensionField, Field};
    use rand::{rngs::StdRng, SeedableRng};
    use std::borrow::Cow;

    mod definition {
        use crate::izip;
        use p3_field::{ExtensionField, Field};

        pub fn eq_evals<E: Field>(x: &[E], scalar: E) -> Vec<E> {
            (0..1 << x.len())
                .map(|i| {
                    (0..x.len())
                        .map(|j| if i >> j & 1 == 0 { E::ONE - x[j] } else { x[j] })
                        .product::<E>()
                        * scalar
                })
                .collect()
        }

        pub fn evaluate<F: Field, E: ExtensionField<F>>(evals: &[F], x: &[E]) -> E {
            izip!(eq_evals(x, E::ONE), evals)
                .map(|(eq_eval_i, eval_i)| eq_eval_i * *eval_i)
                .sum()
        }
    }

    #[test]
    fn eq_evals() {
        fn run<F: Field + FromUniformBytes, E: ExtensionField<F> + FromUniformBytes>() {
            let mut rng = StdRng::from_entropy();
            for num_vars in 0..10 {
                let x = E::random_vec(num_vars, &mut rng);
                let scalar = E::random(&mut rng);
                assert_eq!(
                    MultiPoly::eq(&x, scalar).into_ext(),
                    definition::eq_evals(&x, scalar),
                );
            }
        }

        run::<BabyBear, BabyBear>();
        run::<BabyBear, BinomialExtensionField<BabyBear, 5>>();
    }

    #[test]
    fn evaluate() {
        fn run<F: Field + FromUniformBytes, E: ExtensionField<F> + FromUniformBytes>() {
            let mut rng = StdRng::from_entropy();
            for num_vars in 0..10 {
                let x = E::random_vec(num_vars, &mut rng);
                let evals = F::random_vec(1 << num_vars, &mut rng);
                let poly = MultiPoly::Base(Cow::Borrowed(&evals));
                assert_eq!(
                    x.iter()
                        .rfold(poly.clone(), |mut poly, x_i| {
                            poly.fix_last_var(*x_i);
                            poly
                        })
                        .into_ext(),
                    vec![definition::evaluate(&evals, &x)]
                );
                assert_eq!(poly.evaluate(&x), definition::evaluate(&evals, &x));

                let evals = E::random_vec(1 << num_vars, &mut rng);
                let poly = MultiPoly::Ext(evals.clone());
                assert_eq!(poly.evaluate(&x), definition::evaluate(&evals, &x));
            }
        }

        run::<BabyBear, BabyBear>();
        run::<BabyBear, BinomialExtensionField<BabyBear, 5>>();
    }
}
