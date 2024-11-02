use crate::izip;
use core::{
    array,
    iter::Sum,
    ops::{Add, AddAssign, Deref, DerefMut, Mul, MulAssign},
};
use p3_field::Field;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct FieldArray<F: Field, const N: usize>(pub [F; N]);

impl<F: Field, const N: usize> FieldArray<F, N> {
    pub fn map<T: Field>(self, f: impl FnMut(F) -> T) -> FieldArray<T, N> {
        FieldArray(self.0.map(f))
    }
}

impl<F: Field, const N: usize> Deref for FieldArray<F, N> {
    type Target = [F; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Field, const N: usize> DerefMut for FieldArray<F, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Field, const N: usize> Default for FieldArray<F, N> {
    fn default() -> Self {
        Self(array::from_fn(|_| F::default()))
    }
}

impl<F: Field, const N: usize> AddAssign for FieldArray<F, N> {
    fn add_assign(&mut self, rhs: Self) {
        izip!(&mut self.0, rhs.0).for_each(|(acc, item)| *acc += item);
    }
}

impl<F: Field, const N: usize> Add for FieldArray<F, N> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F: Field, const N: usize> MulAssign<F> for FieldArray<F, N> {
    fn mul_assign(&mut self, rhs: F) {
        self.0.iter_mut().for_each(|lhs| *lhs *= rhs);
    }
}

impl<F: Field, const N: usize> Mul<F> for FieldArray<F, N> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<F: Field, const N: usize> Sum for FieldArray<F, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc + item).unwrap_or_default()
    }
}
