use crate::{
    field::{
        ExtPackedValue, ExtensionField, Field, FieldAlgebra, FieldExtensionAlgebra, PackedValue,
    },
    op_multi_poly,
};
use std::borrow::Cow;
use util::{
    enumerate, par_zip,
    rayon::{self, prelude::*},
    zip, Itertools,
};

#[derive(Clone, Debug)]
pub enum MultiPoly<'a, F: Field, E: ExtensionField<F>> {
    Base(Cow<'a, [F]>),
    Ext(Cow<'a, [E]>),
    ExtPacking(Cow<'a, [E::ExtensionPacking]>),
}

impl<'a, F: Field, E: ExtensionField<F>> MultiPoly<'a, F, E> {
    const __: () = assert!(F::Packing::WIDTH.is_power_of_two());

    pub fn base(evals: impl Into<Cow<'a, [F]>>) -> Self {
        Self::Base(evals.into())
    }

    pub fn ext(evals: impl Into<Cow<'a, [E]>>) -> Self {
        let evals = evals.into();
        if E::D == 1 || F::Packing::WIDTH == 1 || evals.len() <= F::Packing::WIDTH {
            Self::Ext(evals)
        } else {
            Self::ExtPacking(E::ExtensionPacking::ext_pack_slice(&evals).into())
        }
    }

    pub fn ext_packing(evals: impl Into<Cow<'a, [E::ExtensionPacking]>>) -> Self {
        Self::ExtPacking(evals.into())
    }

    pub fn eq<'t>(
        x: impl IntoIterator<IntoIter: ExactSizeIterator, Item = &'t E>,
        scalar: E,
    ) -> Self {
        let mut x = x.into_iter();
        let num_vars = x.len();
        if E::D == 1 || F::Packing::WIDTH == 1 || (1 << num_vars) <= F::Packing::WIDTH {
            let mut evals = E::zero_vec(1 << num_vars);
            evals[0] = scalar;
            enumerate(x).for_each(|(i, x_i)| eq_expand(&mut evals, *x_i, i));
            return Self::ext(evals);
        }

        let mut lo = E::zero_vec(F::Packing::WIDTH);
        lo[0] = scalar;
        let x_lo = x.by_ref().take(F::Packing::WIDTH.ilog2() as usize);
        enumerate(x_lo).for_each(|(i, x_i)| eq_expand(&mut lo, *x_i, i));

        let mut evals = E::ExtensionPacking::zero_vec((1 << num_vars) / F::Packing::WIDTH);
        evals[0] = E::ExtensionPacking::ext_pack(&lo);
        let x_hi = x.copied().map(E::ExtensionPacking::ext_broadcast);
        enumerate(x_hi).for_each(|(i, x_i)| eq_expand(&mut evals, x_i, i));
        Self::ext_packing(evals)
    }

    pub fn num_vars(&self) -> usize {
        match self {
            Self::Base(evals) => evals.len().ilog2() as usize,
            Self::Ext(evals) => evals.len().ilog2() as usize,
            Self::ExtPacking(evals) => (E::Packing::WIDTH * evals.len()).ilog2() as usize,
        }
    }

    pub fn len(&self) -> usize {
        let evals = self;
        op_multi_poly!(|evals| evals.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn split_at(&'a self, mid: usize) -> (Self, Self) {
        match self {
            Self::Base(evals) => {
                let (lo, hi) = evals.split_at(mid);
                (Self::base(lo), Self::base(hi))
            }
            Self::Ext(evals) => {
                let (lo, hi) = evals.split_at(mid);
                (Self::ext(lo), Self::ext(hi))
            }
            Self::ExtPacking(evals) => {
                let (lo, hi) = evals.split_at(mid);
                (Self::ext_packing(lo), Self::ext_packing(hi))
            }
        }
    }

    pub fn fix_last_var(&mut self, x_i: E) {
        match self {
            Self::Base(evals) => *self = Self::fix_last_var_from_base(evals, x_i),
            Self::Ext(evals) => fix_last_var(evals.to_mut(), x_i),
            Self::ExtPacking(evals) => {
                let x_i = E::ExtensionPacking::ext_broadcast(x_i);
                fix_last_var(evals.to_mut(), x_i);
                if evals.len() == 1 {
                    *self = Self::ext(evals[0].ext_unpack());
                }
            }
        }
    }

    fn fix_last_var_from_base(evals: &[F], x_i: E) -> Self {
        if E::D == 1 || F::Packing::WIDTH == 1 || evals.len() <= 2 * F::Packing::WIDTH {
            Self::ext(fix_last_var_from_base(evals, x_i))
        } else {
            let evals = F::Packing::pack_slice(evals);
            let x_i = E::ExtensionPacking::ext_broadcast(x_i);
            Self::ext_packing(fix_last_var_from_base(evals, x_i))
        }
    }

    pub fn evaluate(&self, x: &[E]) -> E {
        match self {
            Self::Base(evals) => x
                .split_last()
                .map(|(x_last, x)| Self::fix_last_var_from_base(evals, *x_last).evaluate(x))
                .unwrap_or_else(|| E::from_base(evals[0])),
            Self::Ext(evals) => evaluate(evals, x),
            Self::ExtPacking(evals) => {
                let (x_lo, x_hi) = x.split_at(F::Packing::WIDTH.ilog2() as usize);
                let x_hi = x_hi
                    .iter()
                    .copied()
                    .map(E::ExtensionPacking::ext_broadcast)
                    .collect_vec();
                evaluate::<E, E>(&evaluate(evals, &x_hi).ext_unpack(), x_lo)
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
            (Ext(e), ExtPacking(p)) | (ExtPacking(p), Ext(e)) => {
                debug_assert!(p.len() <= 2);
                ee(&E::ExtensionPacking::ext_unpack_slice(p), e)
            }
        }
    }

    pub fn to_ext(&self) -> Vec<E> {
        match self {
            Self::Base(evals) => evals.iter().copied().map(E::from_base).collect(),
            Self::Ext(evals) => evals.to_vec(),
            Self::ExtPacking(evals) => E::ExtensionPacking::ext_unpack_slice(evals),
        }
    }

    pub fn into_ext(self) -> Vec<E> {
        match self {
            Self::Base(evals) => evals.iter().copied().map(E::from_base).collect(),
            Self::Ext(evals) => evals.into_owned(),
            Self::ExtPacking(evals) => E::ExtensionPacking::ext_unpack_slice(&evals),
        }
    }
}

pub fn fix_last_var<F: FieldAlgebra + Copy>(evals: &mut Vec<F>, x_i: F) {
    let mid = evals.len() / 2;
    let (lo, hi) = evals.split_at_mut(mid);
    zip!(lo, hi).for_each(|(lo, hi)| *lo += x_i * (*hi - *lo));
    evals.truncate(mid);
}

fn fix_last_var_from_base<F: FieldAlgebra + Copy, E: FieldExtensionAlgebra<F> + Copy>(
    evals: &[F],
    x_i: E,
) -> Vec<E> {
    let (lo, hi) = evals.split_at(evals.len() / 2);
    zip!(lo, hi)
        .map(|(lo, hi)| x_i * (*hi - *lo) + *lo)
        .collect()
}

pub fn evaluate<F, E>(evals: &[F], x: &[E]) -> E
where
    F: FieldAlgebra + Copy + Send + Sync,
    E: FieldExtensionAlgebra<F> + Copy + Send + Sync,
{
    debug_assert_eq!(evals.len(), 1 << x.len());
    match x {
        [] => E::from_base(evals[0]),
        &[ref x @ .., x_i] => {
            let (lo, hi) = evals.split_at(evals.len() / 2);
            let (lo, hi) = rayon::join(|| evaluate(lo, x), || evaluate(hi, x));
            x_i * (hi - lo) + lo
        }
    }
}

pub fn eq_expand<F: FieldAlgebra + Copy + Send + Sync>(evals: &mut [F], x_i: F, i: usize) {
    let (lo, hi) = evals[..2 << i].split_at_mut(1 << i);
    par_zip!(lo, hi).for_each(|(lo, hi)| {
        *hi = *lo * x_i;
        *lo -= *hi;
    });
}

pub fn eq_eval<'a, E: Field>(
    x: impl IntoIterator<Item = &'a E>,
    y: impl IntoIterator<Item = &'a E>,
) -> E {
    E::product(zip!(x, y).map(|(&x, &y)| (x * y).double() + E::ONE - x - y))
}

#[macro_export]
macro_rules! op_multi_poly {
    (|$a:ident| $op:expr, |$b_out:ident| $op_b_out:expr, |$p_out:ident| $op_p_out:expr $(,)?) => {
        match $a {
            $crate::poly::multilinear::MultiPoly::Base(a) => {
                let $a = a;
                let $b_out = $op;
                $op_b_out
            }
            $crate::poly::multilinear::MultiPoly::Ext(a) => {
                let $a = a;
                $op
            }
            $crate::poly::multilinear::MultiPoly::ExtPacking(a) => {
                let $a = a;
                let $p_out = $op;
                $op_p_out
            }
        }
    };
    (|$a:ident| $op:expr, $op_b_out:expr, $op_p_out:expr $(,)?) => {
        op_multi_poly!(|$a| $op, |out| $op_b_out(out), |out| $op_p_out(out))
    };
    (|$a:ident| $op:expr $(,)?) => {
        op_multi_poly!(|$a| $op, |out| out, |out| out)
    };
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
    use crate::{
        baby_bear::BabyBear,
        field::{extension::BinomialExtensionField, ExtensionField, Field, FromUniformBytes},
        poly::multilinear::MultiPoly,
    };
    use rand::{rngs::StdRng, SeedableRng};

    mod definition {
        use crate::field::{ExtensionField, Field};
        use util::zip;

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
            zip!(eq_evals(x, E::ONE), evals)
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
                let poly = MultiPoly::base(&evals);
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
                let poly = MultiPoly::ext(&evals);
                assert_eq!(poly.evaluate(&x), definition::evaluate(&evals, &x));
            }
        }

        run::<BabyBear, BabyBear>();
        run::<BabyBear, BinomialExtensionField<BabyBear, 5>>();
    }
}
